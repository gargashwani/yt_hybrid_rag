import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from openai import OpenAI
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition
from azure.storage.blob import BlobServiceClient

# Local utilities and database
from database import Base, engine, SessionLocal, FoundryAgent, UserChatSession
import storage_utils

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    agent_record = get_or_create_active_agent()
    app.state.agent_name = agent_record.agent_name
    yield

app = FastAPI(title="Azure Foundry RAG API")

# Clients
project_client = AIProjectClient(
    endpoint=os.environ["PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(),
)
openai_client = project_client.get_openai_client()
blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))

# Embeddings require API key + /openai/v1/ (Entra ID not supported for embeddings in v1 API).
# Use AZURE_OPENAI_API_KEY from your Foundry resource (Keys and endpoint in Azure portal).
# Base URL: resource-level (host + /openai/v1/) matches the Keys and endpoint in portal.
from urllib.parse import urlparse
_embedding_base_url = os.getenv("AZURE_OPENAI_EMBEDDING_BASE_URL") or (
    "https://" + urlparse(os.environ["PROJECT_ENDPOINT"]).netloc + "/openai/v1/"
)
_embedding_client = None
if os.getenv("AZURE_OPENAI_API_KEY"):
    _embedding_client = OpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"], base_url=_embedding_base_url)

# --- ROUTES ---

@app.post("/upload-doc")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDFs allowed.")
    
    unique_filename = f"{uuid.uuid4()}-{file.filename}"
    blob_client = blob_service_client.get_blob_client(container="agent-docs", blob=unique_filename)
    
    contents = await file.read()
    blob_client.upload_blob(contents)
    return {"filename": unique_filename, "status": "stored"}

@app.post("/ingest/{blob_name}")
async def process_to_search(blob_name: str):
    """Triggers the RAG pipeline for a stored blob."""
    if _embedding_client is None:
        raise HTTPException(status_code=503, detail="AZURE_OPENAI_API_KEY not set.")
    try:
        count = storage_utils.ingest_blob_to_search(blob_name, _embedding_client)
        return {"message": "Success", "chunks_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-docs")
async def list_docs():
    return storage_utils.list_agent_docs()

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
        agent_record = FoundryAgent(agent_id=agent.id, agent_name=agent.name, agent_model=agent.model)
        db.add(agent_record)
        db.commit()
    return agent_record

def _get_agent_name():
    name = getattr(app.state, "agent_name", None)
    if name:
        return name
    db = SessionLocal()
    try:
        r = db.query(FoundryAgent).first()
        return r.agent_name if r else os.environ["AGENT_NAME"]
    finally:
        db.close()

@app.post("/chat/{user_email}")
async def chat(user_email: str, message: str):
    db = SessionLocal()
    try:
        # --- NEW: RETRIEVAL STEP ---
        # Search for context based on the user's message
        context = storage_utils.search_docs(message, _embedding_client)
        
        # Combine the document context with the user's question
        enriched_input = f"""
        Use the following context from uploaded documents to answer the question. 
        If the answer isn't in the context, say you don't know.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {message}
        """
        # ---------------------------

        user_session = db.query(UserChatSession).filter(UserChatSession.user_email == user_email).first()
        if not user_session:
            new_conv = openai_client.conversations.create()
            user_session = UserChatSession(user_email=user_email, foundry_conversation_id=new_conv.id)
            db.add(user_session)
            db.commit()
            db.refresh(user_session)

        agent_name = _get_agent_name()
        response = openai_client.responses.create(
            conversation=user_session.foundry_conversation_id,
            extra_body={"agent": {"name": agent_name, "type": "agent_reference"}},
            input=enriched_input, # Use the enriched input instead of just 'message'
        )
        return {"agent_response": response.output_text}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)