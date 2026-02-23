from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchField, 
    SearchIndex,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile
)

from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv
load_dotenv()

def create_search_index():
    client = SearchIndexClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"])
    )

    index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchableField(name="metadata", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="myHnswProfile"
        )
    ]

    # Configure Vector search logic
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

    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)

    try:
        client.create_index(index)
        print(f"Index '{index_name}' created successfully")
    except Exception as e:
        print(f"Error: {e}")    

if __name__=="__main__":
    create_search_index()        