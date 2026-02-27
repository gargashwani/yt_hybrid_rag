import os, uuid
from fastapi import FastAPI
from dotenv import load_dotenv
from database import Base, engine, SessionLocal, FoundryAgent, UserChatSession

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition
from contextlib import asynccontextmanager
from openai import OpenAI
load_dotenv()

from routes import storage, ai_search


@asynccontextmanager
async def lifespan(app: FastAPI):

    credential = DefaultAzureCredential()

    project_client = AIProjectClient(
        endpoint=os.environ["PROJECT_ENDPOINT"],
        credential=credential
    )

    app.state.project_client = project_client
    app.state.openai_client = project_client.get_openai_client()
    app.state.account_url = "https://codesipsdocs.blob.core.windows.net"

    _embedding_base_url = os.getenv('AZURE_OPENAI_EMBEDDING_BASE_URL')
    app.state.embedding_client = OpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        base_url=_embedding_base_url
    )
    # STARTUP LOGIC
    Base.metadata.create_all(bind=engine)
    get_or_create_active_agent()

    yield
    # SHUTDOWN LOGIC
    project_client.close()
    print("Shutting down, cleaning up resources")

from routes import storage  # Import your new file

app = FastAPI(title="Azure Foundry Agentic API", lifespan = lifespan)

app.include_router(storage.router)
app.include_router(ai_search.router)


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port="8000")