from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv

# Load environment variables (Endpoint, Key, Index Name)
load_dotenv()

def create_search_index():
    # Initialize the Management Client to create/modify infrastructure
    client = SearchIndexClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"])
    )

    index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]

    # Define the Schema: How your data is structured
    fields = [
        # 'id' must be unique and is the primary key for each text chunk
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        
        # 'content' stores the actual text from the PDF for the LLM to read
        SearchableField(name="content", type=SearchFieldDataType.String),
        
        # 'metadata' stores the source (e.g., filename) for traceability
        SearchableField(name="metadata", type=SearchFieldDataType.String),
        
        # This is the 'Embedding' field. It stores the mathematical meaning of the text.
        # vector_search_dimensions=1536 is the standard for text-embedding-3-small / ada-002
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536, 
            vector_search_profile_name="myHnswProfile"
        )
    ]

    # Configure Vector Search Logic
    # HNSW (Hierarchical Navigable Small World) allows for fast, approximate nearest neighbor search
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="myHnswAlgorithm")
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile", 
                algorithm_configuration_name="myHnswAlgorithm"
            )
        ]
    )

    # Combine Schema and Search Logic into an Index Object
    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    
    try:
        # Create the index in the Azure Cloud
        client.create_index(index)
        print(f"✅ Index '{index_name}' created successfully!")
    except Exception as e:
        # Catch errors like 'Index already exists'
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    create_search_index()