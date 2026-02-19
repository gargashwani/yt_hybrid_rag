"""Azure AI Search operations."""
import os
import uuid
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
import storage_utils


def get_search_client():
    """Get a SearchClient instance."""
    return SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=os.environ["AZURE_SEARCH_INDEX_NAME"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"])
    )


def ingest_blob_to_search(blob_name: str, embedding_client):
    """Orchestrates Extraction -> Chunking -> Embedding -> Search Indexing."""
    text = storage_utils.extract_text_from_blob(blob_name)
    if not text:
        raise ValueError("Empty document or file not found.")
    
    chunks = storage_utils.chunk_text(text)

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
