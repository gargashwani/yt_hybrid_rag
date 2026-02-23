import os, uuid
from fastapi import APIRouter, HTTPException, Body
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from openai import OpenAI
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from database import SessionLocal, UserChatSession


router = APIRouter(
    prefix="/ai_search",
    tags=["ai_search"]
)

_embedding_base_url = os.getenv("AZURE_OPENAI_EMBEDDING_BASE_URL")
_embedding_client = OpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"], base_url=_embedding_base_url)
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

@router.post("/ingest/{blob_name}")
async def process_to_search(blob_name: str):
    """Triggers the RAG pipeline for a stored blob."""
    if _embedding_client is None:
        raise HTTPException(status_code=503, detail="AZURE_OPENAI_API_KEY not set.")
    count = ingest_blob_to_search(blob_name, _embedding_client)
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

from azure.ai.projects import AIProjectClient
project_client = AIProjectClient(
    endpoint=os.environ["PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(),
)
openai_client = project_client.get_openai_client()

@router.post("/chat/")
async def chat(user_email: str = Body(...), message: str = Body(...)):
    db = SessionLocal()
    # 1. RETRIEVAL: Fetch relevant text from Azure Search
    context = search_docs(message, _embedding_client)
    
    # 2. SESSION: Fetch or create the Foundry Conversation
    user_session = db.query(UserChatSession).filter(UserChatSession.user_email == user_email).first()
    if not user_session:
        new_conv = openai_client.conversations.create()
        user_session = UserChatSession(user_email=user_email, foundry_conversation_id=new_conv.id)
        db.add(user_session)
        db.commit()
        db.refresh(user_session)

    # 3. Pass context via instructions override
    agent_name = os.environ["AGENT_NAME"]

    dynamic_override = f"""CRITICAL CONTEXT:
    ---
    {context}
    ---
    USER QUESTION: {message}

    INSTRUCTION: Answer the question using ONLY the context above."""

    response = openai_client.responses.create(
        conversation=user_session.foundry_conversation_id,
        extra_body={
            "agent": {
                "name": agent_name, 
                "type": "agent_reference",
                "instructions": dynamic_override
            }
        },
        input=message,
    )
    
    db.close()
    return {"agent_response": response.output_text}
