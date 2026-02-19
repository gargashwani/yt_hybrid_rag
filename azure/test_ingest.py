#!/usr/bin/env python3
"""Run ingest pipeline. Usage: from repo root run python azure/test_ingest.py [blob_name]
Requires AZURE_OPENAI_API_KEY in azure/.env for embeddings (Entra ID not supported for v1 embeddings)."""
import os
import sys

# Run from azure/ so imports and .env work
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

import storage_utils

def main():
    # Use API key + v1 base URL for embeddings (same as main.py)
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("Set AZURE_OPENAI_API_KEY in azure/.env (from Foundry resource Keys and endpoint)")
        return 1
    from urllib.parse import urlparse
    base_url = os.getenv("AZURE_OPENAI_EMBEDDING_BASE_URL") or (
        "https://" + urlparse(os.environ["PROJECT_ENDPOINT"]).netloc + "/openai/v1/"
    )
    embedding_client = OpenAI(api_key=api_key, base_url=base_url)

    if len(sys.argv) > 1:
        blob_name = sys.argv[1]
        print(f"Testing ingest for blob: {blob_name!r}")
    else:
        docs = storage_utils.list_agent_docs()
        if not docs:
            print("No blobs in agent-docs. Upload a PDF first or pass blob name: python test_ingest.py <blob_name>")
            return 1
        blob_name = docs[0]["name"]
        print(f"Using first blob: {blob_name!r}")

    try:
        count = storage_utils.ingest_blob_to_search(blob_name, embedding_client)
        print(f"OK: indexed {count} chunks")
        return 0
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
