import os, uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from dotenv import load_dotenv
from database import Base, engine, SessionLocal, FoundryAgent, UserChatSession

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition
from contextlib import asynccontextmanager

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

agent_name = os.environ["AGENT_NAME"]
openai_client = project_client.get_openai_client()




def get_or_create_active_agent():
    db = SessionLocal()

    # check database
    agent_record = db.query(FoundryAgent).first()

    if not agent_record:
        agent = project_client.agents.create_version(
            agent_name=os.environ["AGENT_NAME"],
            definition=PromptAgentDefinition(
                model=os.environ["MODEL_DEPLOYMENT_NAME"],
                instructions="You are a helpful assistant that answers general questions",
            ),
        )

        agent_record = FoundryAgent(
            agent_id = agent.id, 
            agent_name = agent.name,
            agent_model = os.environ["MODEL_DEPLOYMENT_NAME"]
        )
        db.add(agent_record)
        db.commit()
        db.refresh(agent_record)
        print(f"Agent created successfully")

    else:
        print(f"Agent found in the db: {agent_record.agent_id}")

    return agent_record

@app.post("/chat/{user_email}")
async def chat_with_agent(user_email: str, message: str):
    db = SessionLocal()

    user_session = db.query(UserChatSession).filter(UserChatSession.user_email == user_email).first()

    conversation_id = user_session.foundry_conversation_id

    if not user_session:
        new_conversation =  openai_client.conversations.create()
        user_session = UserChatSession(
            user_email = user_email,
            foundry_conversation_id = new_conversation.id
        )
        conversation_id = new_conversation.id
        db.add(user_session)
        db.commit()
        db.refresh(user_session)

    # Chat with the agent to answer questions
    response = openai_client.responses.create(
        conversation=conversation_id, #Optional conversation context for multi-turn
        extra_body={"agent": {"name": agent_name, "type": "agent_reference"}},
        input=message,
    )
    return {"Agent response": response.output_text}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port="8000")