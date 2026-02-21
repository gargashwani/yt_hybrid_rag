import os
from fastapi import FastAPI, HTTPException, Body
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from openai import OpenAI
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition
from azure.storage.blob import BlobServiceClient

# Local modules
from database import Base, engine, SessionLocal, FoundryAgent, UserChatSession
import ai_search

# Blob Storage Account Imports
from azure.storage.blob import BlobServiceClient

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP LOGIC
    Base.metadata.create_all(bind=engine)
    get_or_create_active_agent()

    yield
    # SHUTDOWN LOGIC
    print("Shutting down, cleaning up resources")

from routes import storage  # Import your new file

app = FastAPI(title="Azure Foundry Agentic API", lifespan = lifespan)
# Register the storage routes
app.include_router(storage.router)

project_client = AIProjectClient(
    endpoint=os.environ["PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(),
)
openai_client = project_client.get_openai_client()
blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))

# Embeddings client (requires API key, Entra ID not supported for embeddings v1 API)
_embedding_client = None
if os.getenv("AZURE_OPENAI_API_KEY"):
    from urllib.parse import urlparse
    _embedding_base_url = os.getenv("AZURE_OPENAI_EMBEDDING_BASE_URL") or (
        "https://" + urlparse(os.environ["PROJECT_ENDPOINT"]).netloc + "/openai/v1/"
    )
    _embedding_client = OpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"], base_url=_embedding_base_url)

# --- ROUTES ---

# 1. First, define the Account URL (based on your Azure Template)
ACCOUNT_URL = "https://codesipsdocs2026.blob.core.windows.net"



@app.post("/ingest/{blob_name}")
async def process_to_search(blob_name: str):
    """Triggers the RAG pipeline for a stored blob."""
    if _embedding_client is None:
        raise HTTPException(status_code=503, detail="AZURE_OPENAI_API_KEY not set.")
    count = ai_search.ingest_blob_to_search(blob_name, _embedding_client)
    return {"message": "Success", "chunks_indexed": count}

# --- AGENT LOGIC ---

def get_or_create_active_agent():
    db = SessionLocal()
    agent_record = db.query(FoundryAgent).first()

    if not agent_record:
        agent = project_client.agents.create_version(
            agent_name=os.environ["AGENT_NAME"],
            definition=PromptAgentDefinition(
                model=os.environ["MODEL_DEPLOYMENT_NAME"],
                instructions="You are an expert document assistant. Use the provided context to answer accurately."
            ),
        )
        agent_record = FoundryAgent(agent_id=agent.id, agent_name=agent.name, agent_model=agent.definition.model)
        db.add(agent_record)
        db.commit()
    return agent_record

@app.post("/chat/")
async def chat(user_email: str = Body(...), message: str = Body(...)):
    db = SessionLocal()
    # 1. RETRIEVAL: Fetch relevant text from Azure Search
    context = ai_search.search_docs(message, _embedding_client)
    
    # 2. SESSION: Fetch or create the Foundry Conversation
    user_session = db.query(UserChatSession).filter(UserChatSession.user_email == user_email).first()
    if not user_session:
        new_conv = openai_client.conversations.create()
        user_session = UserChatSession(user_email=user_email, foundry_conversation_id=new_conv.id)
        db.add(user_session)
        db.commit()
        db.refresh(user_session)

    # 3. Pass context via instructions override
    agent_name = getattr(app.state, "agent_name", os.environ["AGENT_NAME"])

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




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)