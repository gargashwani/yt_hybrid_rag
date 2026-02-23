from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    ComplexField,
    SearchIndex,
    CorsOptions,
    ScoringProfile
)
from azure.core.credentials import AzureKeyCredential

from typing import List

import os
from dotenv import load_dotenv
load_dotenv()

service_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
key  = os.environ["AZURE_SEARCH_KEY"]
client = SearchIndexClient(service_endpoint, AzureKeyCredential(key))
name = "hotels"
fields = [
    SimpleField(name="hotelId", type=SearchFieldDataType.String, key=True),
    SimpleField(name="baseRate", type=SearchFieldDataType.Double),
    SearchableField(name="description", type=SearchFieldDataType.String, collection=True),
    ComplexField(
        name="address",
        fields=[
            SimpleField(name="streetAddress", type=SearchFieldDataType.String),
            SimpleField(name="city", type=SearchFieldDataType.String),
        ],
        collection=True,
    ),
]
cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)
scoring_profiles: List[ScoringProfile] = []
index = SearchIndex(name=name, fields=fields, scoring_profiles=scoring_profiles, cors_options=cors_options)

result = client.create_index(index)