
import os
import uuid
import fitz  # PyMuPDF
from fastapi import APIRouter, Request, Body



# Azure SDKs
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery


# AI & Processing
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Database
from database import SessionLocal, UserChatSession


router = APIRouter(
    prefix="/ai_search",
    tags=["ai_search"]
)

ACCOUNT_URL="https://codesipsdocs.blob.core.windows.net"
CONTAINER_NAME = "agent-docs"


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    """Splits text into manageable pieces for Azure AI Search."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)



def get_blob_content(blob_name: str, container_name: str = "agent-docs"):
    """Downloads raw bytes from Azure Blob Storage."""
    try:
        blob_service_client = BlobServiceClient(ACCOUNT_URL, credential=DefaultAzureCredential())
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        return blob_client.download_blob().readall()
    except Exception:
        return None



def extract_text_from_blob(blob_name: str):
    """Converts PDF bytes to a plain text string."""
    pdf_bytes = get_blob_content(blob_name)
    if not pdf_bytes:
        return None
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "".join([page.get_text() for page in doc])


def get_search_client():
    """Get a SearchClient instance."""
    return SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"])
    )


def ingest_blob_to_search(blob_name: str, embedding_client):
    """Orchestrates Extraction -> Chunking -> Embedding -> Search Indexing."""
    text = extract_text_from_blob(blob_name)
    if not text:
        raise ValueError("Empty document or file not found.")
    
    chunks = chunk_text(text)

    res = embedding_client.embeddings.create(
        input=chunks,
        model=os.environ["EMBEDDING_DEPLOYMENT_NAME"]
    )
    embeddings = [item.embedding for item in res.data]

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
async def process_to_search(request:Request, blob_name: str = Body(...)):
    """Triggers the RAG pipeline for a stored blob."""
    embedding_client = request.app.state.embedding_client
    count = ingest_blob_to_search(blob_name, embedding_client)
    return {"message": "Success", "chunks_indexed": count}


# Search indexing and open ai calls

def search_docs(query: str, embedding_client):
    """Search documents using hybrid search (text + vector)."""
    search_client = get_search_client()

    # Vectorize the user's question
    res = embedding_client.embeddings.create(
        input=[query],
        model=os.environ["EMBEDDING_DEPLOYMENT_NAME"]
    )
    query_vector = res.data[0].embedding

    # Hybrid search: text + vector
    vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=3, fields="content_vector")
    
    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        top=3,
        select=["content", "metadata"]
    )

    # Extract content from results
    # Azure Search returns SearchItem objects accessed via dict-style: result['field_name']
    context_parts = []
    for result in results:
        try:
            content = result['content']
            if content:
                context_parts.append(str(content))
        except (KeyError, TypeError):
            continue
    
    context = "\n\n".join(context_parts) if context_parts else ""
    return context


@router.post("/chat/")
async def chat(request: Request, user_email: str = Body(...), message: str = Body(...)):
    # 1. SETUP: Use shared clients from state
    embedding_client = request.app.state.embedding_client
    openai_client = request.app.state.openai_client
    db = SessionLocal()
    
    # 2. RETRIEVAL: Fetch relevant text from Azure Search
    context = search_docs(message, embedding_client)
    
    # 3. SESSION: Fetch or create the Foundry Conversation
    user_session = db.query(UserChatSession).filter(UserChatSession.user_email == user_email).first()
    if not user_session:
        new_conv = openai_client.conversations.create()
        user_session = UserChatSession(user_email=user_email, foundry_conversation_id=new_conv.id)
        db.add(user_session)
        db.commit()
        db.refresh(user_session)

    # 4. Pass context via instructions override
    agent_name = os.environ["AGENT_NAME"]

    # Make the instruction even more aggressive
    dynamic_override = f"""
    # ROLE
    You are a professional.

    # CONTEXT DATA
    {context}

    # TASK
    Answer the user's question based ONLY on the CONTEXT DATA above.
    If the information is missing, state clearly that it is not in the CV.

    # USER QUESTION
    {message}
    """

    response = openai_client.responses.create(
        conversation=user_session.foundry_conversation_id,
        extra_body={
            "agent": {
                "name": agent_name, 
                "type": "agent_reference",
                "instructions": dynamic_override # This is the key
            }
        },
        input=message,
    )
    
    db.close()
    return {"agent_response": response.output_text}
