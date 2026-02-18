import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI
from sqlalchemy.orm import Session

# Import from your new database file
from database import Base, engine, SessionLocal, FoundryAgent, UserChatSession

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition

load_dotenv()

# 1. LIFESPAN: The Bootloader
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print("🚀 App starting up...")
    
    # Ensure tables are ready
    Base.metadata.create_all(bind=engine)
    
    # Initialize Agent and store ID and name in memory
    db = SessionLocal()
    try:
        agent_record = get_or_create_active_agent(db)
        # Store in app.state so it's globally accessible in routes
        app.state.agent_id = agent_record.agent_id
        app.state.agent_name = agent_record.agent_name
        print(f"🤖 Global Agent Loaded: {app.state.agent_name} (ID: {app.state.agent_id})")
    finally:
        db.close()
    
    yield  # Application processes requests here
    
    # --- SHUTDOWN ---
    print("🛑 App shutting down...")

# 2. CLIENT SETUP (Singleton)
project_client = AIProjectClient(
    endpoint=os.environ["PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(),
)
openai_client = project_client.get_openai_client()

# 3. APP INITIALIZATION
app = FastAPI(title="Production Microsoft Foundry API", lifespan=lifespan)

def get_or_create_active_agent(db: Session):
    """
    Checks the local Postgres database for an existing Agent ID.
    If not found, it creates a new Agent in Azure AI Foundry and saves the ID.
    """
    # 1. Check if we already have an agent registered in our local DB
    agent_record = db.query(FoundryAgent).first()
    
    if not agent_record:
        print("🚀 No agent found in DB. Creating a new one in Azure AI Foundry...")
        
        # 2. Create the agent in Azure using the SDK
        # The 'create_version' method is the standard for Foundry Agents
        azure_agent = project_client.agents.create_version(
            agent_name=os.environ["AGENT_NAME"],
            definition=PromptAgentDefinition(
                model=os.environ["MODEL_DEPLOYMENT_NAME"],
                instructions=(
                    "You are a professional research assistant. "
                    "Provide concise, verified answers and cite your sources if possible."
                ),
                # tools=[CodeInterpreterTool()]  # You can add tools here in Week 2
            ),
        )
        
        # 3. Save the Azure Agent ID to our Postgres DB for future use
        agent_record = FoundryAgent(
            agent_id=azure_agent.id,
            agent_name=azure_agent.name,
            agent_model=os.environ["MODEL_DEPLOYMENT_NAME"]
        )
        db.add(agent_record)
        db.commit()
        db.refresh(agent_record)
        print(f"✅ Agent Created & Saved to DB: {agent_record.agent_id}")
    
    else:
        print(f"📡 Found existing Agent in DB: {agent_record.agent_id}")
    
    return agent_record


@app.post("/chat/{user_email}")
async def chat_with_agent(user_email: str, message: str):
    db = SessionLocal()
    # Get agent name from app.state
    agent_name = app.state.agent_name

    user_session = db.query(UserChatSession).filter(UserChatSession.user_email == user_email).first()
    
    if not user_session:
        new_conv = openai_client.conversations.create()
        user_session = UserChatSession(user_email=user_email, foundry_conversation_id=new_conv.id)
        db.add(user_session)
        db.commit()
        db.refresh(user_session)

    # Use agent name instead of ID - this is what the API expects
    response = openai_client.responses.create(
        conversation=user_session.foundry_conversation_id,
        extra_body={"agent": {"name": agent_name, "type": "agent_reference"}},
        input=message,
    )
    return {"agent_response": response.output_text}
