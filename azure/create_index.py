from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField, 
    SearchFieldDataType, 
    SearchableField,
    CorsOptions,
    SearchIndex,
    ScoringProfile,
    SearchField
)
from azure.core.credentials import AzureKeyCredential

import os
from dotenv import load_dotenv

# Load environment variables (Endpoint, Key, Index Name)
load_dotenv()

from typing import List

key = os.environ["AZURE_SEARCH_KEY"]
service_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]

client = SearchIndexClient(service_endpoint, AzureKeyCredential(key))
name = "hotels"
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="content", type=SearchFieldDataType.String),
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
cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)
scoring_profiles: List[ScoringProfile] = []
index = SearchIndex(name=name, fields=fields, scoring_profiles=scoring_profiles, cors_options=cors_options)

result = client.create_index(index)