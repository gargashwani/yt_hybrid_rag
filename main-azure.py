import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy_utils import database_exists, create_database

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition

# Load env
load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL" ,"postgresql://ashwanigarg:Test123@localhost:5432/foundry_db")

Base = declarative_base()

class UserChatSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, unique=True, index=True)
    foundry_conversation_id = Column(String) # The "Link" to Microsoft Foundry memory
    
class FoundryAgent(Base):
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String, unique=True, index=True)
    agent_name = Column(String)
    agent_model = Column(String, nullable=True)
    
engine = create_engine(DATABASE_URL)

# This checks if "foundry_db" exists on the server
if not database_exists(engine.url):
    create_database(engine.url)
    print("Database 'foundry_db' created successfully!")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# 2. SETUP: Microsoft Foundry Client (Singleton)
project_client = AIProjectClient(
    endpoint=os.environ["PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

openai_client = project_client.get_openai_client()

# 3. FASTAPI APP
app = FastAPI(title="Production Microsoft Foundry API")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
# 1. Logic to Create Agent Programmatically
def get_or_create_active_agent(db: Session):
    # Check if we already have an agent registered in our DB
    agent_record = db.query(FoundryAgent).first()
    
    if not agent_record:
        print("🚀 No agent found in DB. Creating a new one in Azure AI Foundry...")
        
        # Create the agent in Azure
        # You can add tools like CodeInterpreterTool() or FileSearchTool() here
        azure_agent = project_client.agents.create_version(
            agent_name=os.environ["AGENT_NAME"],
            definition=PromptAgentDefinition(
                model=os.environ["MODEL_DEPLOYMENT_NAME"],
                instructions="You are a professional research assistant. Provide concise and verified answers.",
                # tools=[CodeInterpreterTool()] # Optional: Add tools here
                # temprature
            ),
        )
        
        # Save the Azure Agent ID to our Postgres DB
        agent_record = FoundryAgent(
            agent_id=azure_agent.id,
            agent_name=azure_agent.name,
            agent_model=os.environ["MODEL_DEPLOYMENT_NAME"]
        )
        db.add(agent_record)
        db.commit()
        db.refresh(agent_record)
        print(f"✅ Agent Created & Saved: {agent_record.agent_id}")
    
    return agent_record      
        
        
@app.post("/chat/{user_email}")
async def chat_with_agent(user_email: str, message: str, db: Session = Depends(get_db)):
    # Check if agent is created and present
    # Automatically get or create the agent
    agent_info = get_or_create_active_agent(db)

    # STEP 1: Look for existing conversation in Postgres
    user_session = db.query(UserChatSession).filter(UserChatSession.user_email == user_email).first()

    if not user_session:
        # Create a brand new conversation in Foundry if user is new
        print(f"New user detected: {user_email}. Creating Foundry Conversation...")
        new_conv = openai_client.conversations.create()
        
        user_session = UserChatSession(
            user_email=user_email, 
            foundry_conversation_id=new_conv.id
        )
        db.add(user_session)
        db.commit()
        db.refresh(user_session)

    # STEP 2: Send message to the Foundry Agent using the stored conversation_id
    try:
        response = openai_client.responses.create(
            conversation=user_session.foundry_conversation_id,
            extra_body={"agent": {"name": os.environ["AGENT_NAME"], "type": "agent_reference"}},
            input=message,
        )
        
        return {
            "user": user_email,
            "agent_response": response.output_text,
            "foundry_session_id": user_session.foundry_conversation_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)    
        