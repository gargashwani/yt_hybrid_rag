"""Azure Blob Storage operations."""
import os
import fitz  # PyMuPDF
from azure.storage.blob import BlobServiceClient
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_blob_service_client():
    """Get a BlobServiceClient instance."""
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    return BlobServiceClient.from_connection_string(connect_str)


def list_agent_docs():
    """Lists all blobs in the agent-docs container."""
    blob_service_client = get_blob_service_client()
    container_client = blob_service_client.get_container_client("agent-docs")
    blob_list = container_client.list_blobs()
    return [
        {"name": blob.name, "size": blob.size, "created": blob.last_modified}
        for blob in blob_list
    ]


def get_blob_content(blob_name: str, container_name: str = "agent-docs"):
    """Downloads raw bytes from Azure Blob Storage."""
    try:
        blob_service_client = get_blob_service_client()
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


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    """Splits text into manageable pieces for Azure AI Search."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)
