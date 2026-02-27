import os
import uuid
import fitz
from fastapi import APIRouter, Request, Body

from database import SessionLocal, UserChatSession

# Azure SDKs
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

from langchain_text_splitters import RecursiveCharacterTextSplitter

ACCOUNT_URL="https://codesipsdocs.blob.core.windows.net"
CONTAINER_NAME = "agent-docs"

router = APIRouter(
    prefix="/ai_search",
    tags=["ai_search"]
)


def chunk_text(text:str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def get_blob_content(blob_name: str):
    container_name = "agent-docs"
    blob_service_client = BlobServiceClient(ACCOUNT_URL, credential=DefaultAzureCredential())
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    return blob_client.download_blob().readall()

def extract_text_from_blob(blob_name: str):
    pdf_byes = get_blob_content(blob_name)
    if not pdf_byes:
        return None
    doc = fitz.open(stream=pdf_byes, filetype="pdf")
    return ".".join([page.get_text() for page in doc])

def get_search_client():
    return SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"])
    )


def ingest_blob_to_search(blob_name: str, embedding_client):
    # Extract -> Chunking -> Embedding -> Search Index.
    
    # Extraction
    text = extract_text_from_blob(blob_name)
    if not text:
        raise ValueError("Empty document or file not found")

    # Chunking
    chunks = chunk_text(text)

    # Embedding
    res = embedding_client.embeddings.create(
        input= chunks,
        model = os.environ["EMBEDDING_DEPLOYMENT_NAME"]
    )

    embeddings = [item.embedding for item in res.data]

    # Search Index
    search_client = get_search_client()
    batch = []
    for i, chunk in enumerate(chunks):
        batch.append({
            "id": f"chunk_{uuid.uuid4().hex}",
            "content": chunk,
            "metadata": blob_name,
            "content_vector": embeddings[i]
        })
    search_client.upload_documents(documents=batch)
    return len(batch)


@router.post("/ingest")
async def process_to_search(request: Request, blob_name: str = Body(...)):
    embedding_client = request.app.state.embedding_client
    count = ingest_blob_to_search(blob_name, embedding_client)
    return {"message": "Success", "chunks_indexed": count}


def search_docs(query: str, embedding_client):
    search_client = get_search_client()
    res = embedding_client.embeddings.create(
        input = [query],
        model = os.environ["EMBEDDING_DEPLOYMENT_NAME"]
    )
    query_vector = res.data[0].embedding

    vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=3, fields="content_vector")

    results = search_client.search(
        search_text=query, 
        vector_queries=[vector_query],
        top = 3,
        select=["content", "metadata"]
    )

    context_parts = []

    for result in results:
        try: 
            content = result['content']
            if content:
                context_parts.append(str(content))
        except(KeyError, TypeError):
            continue        

    context = "\n\n".join(context_parts) if context_parts else ""        
    return context

@router.post("/chat")
async def chat(request: Request, user_email: str = Body(...), message: str = Body(...)):
    embedding_client = request.app.state.embedding_client
    openai_client = request.app.state.openai_client
    db = SessionLocal()

    context = search_docs(message, embedding_client)

    user_session = db.query(UserChatSession).filter(UserChatSession.user_email == user_email).first()
    if not user_session:
        new_conv = openai_client.conversations.create()
        user_session = UserChatSession(
            user_email = user_email,
            foundry_conversation_id = new_conv.id
        )
        db.add(user_session)
        db.commit()
        db.refresh(user_session)

    agent_name = os.environ["AGENT_NAME"]

    dynamic_context = f"""
    # ROLE
    You are a professional.

    # CONTEXT DATA
    {context}

    # TASK
    Answer the user's question based ONLY on the CONTEXT DATA above.
    If the information is missing, state clearly that it is not present.

    # USER QUESTION
    {message}
    """

    response = openai_client.responses.create(
        conversation = user_session.foundry_conversation_id,
        extra_body={
            "agent": {
                "name": agent_name,
                "type": "agent_reference",
                "instructions": dynamic_context
            }
        },
        input = message
    )

    db.close()
    return {"agent_response": response.output_text}
