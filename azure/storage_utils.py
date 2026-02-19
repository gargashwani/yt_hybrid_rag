from azure.storage.blob import BlobServiceClient
import os, uuid, sys
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from azure.search.documents.models import VectorizedQuery
def list_agent_docs():
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client("agent-docs")
    
    blob_list = container_client.list_blobs()
    
    # Create a clean list of names to return to the frontend
    # We also include the size or last modified date for a "Pro" feel
    return [
        {"name": blob.name, "size": blob.size, "created": blob.last_modified} 
        for blob in blob_list
    ]

def get_blob_content(blob_name: str, container_name: str = "agent-docs"):
    """Fetches the raw bytes of a file from Azure Blob Storage."""
    try:
        connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        # Download as bytes
        downloader = blob_client.download_blob()
        return downloader.readall()
    except Exception:
        return None

import fitz  # PyMuPDF

def extract_text_from_blob(blob_name: str):
    # Reuse your existing function to get bytes
    pdf_bytes = get_blob_content(blob_name)
    if not pdf_bytes:
        return "File not found."
    
    # Open the PDF from memory
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    Splits text into manageable pieces for Azure AI Search.
    chunk_size: How many characters per piece.
    chunk_overlap: To keep context, the end of chunk 1 is the start of chunk 2.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    return chunks

def index_document_chunks(blob_name: str, chunks: list, embeddings: list):
    search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    search_key = os.environ["AZURE_SEARCH_KEY"]
    index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]

    search_client = SearchClient(search_endpoint, index_name, AzureKeyCredential(search_key))

    documents = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        documents.append({
            "id": f"{uuid.uuid4()}", # Unique ID for each chunk
            "parent_id": blob_name, # So you know which file it came from
            "content": chunk,
            "content_vector": vector,
            "metadata": {"filename": blob_name}
        })

    result = search_client.upload_documents(documents)
    return len(result)
# --- BLOB OPERATIONS ---

def get_blob_content(blob_name: str, container_name: str = "agent-docs"):
    """Downloads raw bytes from Azure Blob Storage."""
    try:
        connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        return blob_client.download_blob().readall()
    except Exception:
        return None

def list_agent_docs():
    """Lists all blobs in the container."""
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client("agent-docs")
    return [{"name": b.name, "size": b.size, "created": b.last_modified} for b in container_client.list_blobs()]

# --- TEXT PROCESSING ---

def extract_text_from_blob(blob_name: str):
    """Converts PDF bytes to a plain text string."""
    pdf_bytes = get_blob_content(blob_name)
    if not pdf_bytes: return None
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "".join([page.get_text() for page in doc])

def chunk_text(text: str):
    """Splits text into 1000-character pieces with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

# --- SEARCH INGESTION ---

def ingest_blob_to_search(blob_name: str, openai_client):
    """Orchestrates Extraction -> Chunking -> Embedding -> Search Indexing."""
    text = extract_text_from_blob(blob_name)
    if not text:
        raise ValueError("Empty document or file not found.")
    chunks = chunk_text(text)

    res = openai_client.embeddings.create(
        input=chunks,
        model=os.environ["EMBEDDING_DEPLOYMENT_NAME"]
    )
    embeddings = [item.embedding for item in res.data]

    search_client = SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"])
    )
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


def search_docs(query: str, embedding_client):
    search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    search_key = os.environ["AZURE_SEARCH_KEY"]
    index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]

    search_client = SearchClient(search_endpoint, index_name, AzureKeyCredential(search_key))

    # 1. Vectorize the user's question
    res = embedding_client.embeddings.create(
        input=[query],
        model=os.environ["EMBEDDING_DEPLOYMENT_NAME"]
    )
    query_vector = res.data[0].embedding

    # 2. Search the index
    vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=3, fields="content_vector")
    
    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        top=3
    )

    # 3. Combine findings into one string
    context = "\n".join([doc['content'] for doc in results])
    return context