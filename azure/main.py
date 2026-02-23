import os
from fastapi import FastAPI, HTTPException, Body
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from openai import OpenAI
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition

# Local modules
from database import Base, engine, SessionLocal, FoundryAgent, UserChatSession

load_dotenv()

from routes import storage, ai_search

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP LOGIC
    Base.metadata.create_all(bind=engine)
    get_or_create_active_agent()

    yield
    # SHUTDOWN LOGIC
    print("Shutting down, cleaning up resources")

from routes import storage, ai_search  # Import your new file

app = FastAPI(title="Azure Foundry Agentic API", lifespan = lifespan)

app.include_router(storage.router)
app.include_router(ai_search.router)

project_client = AIProjectClient(
    endpoint=os.environ["PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(),
)
openai_client = project_client.get_openai_client()

# 1. First, define the Account URL (based on your Azure Template)
ACCOUNT_URL = "https://codesipsdocs.blob.core.windows.net"



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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)