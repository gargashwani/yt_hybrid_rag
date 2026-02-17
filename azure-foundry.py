import os, sys
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition
from database import Base, Thread, SessionLocal, engine, Agent
from pprint import pprint

load_dotenv()
agent_name = os.environ["AGENT_NAME"]

project_client = AIProjectClient(
    endpoint=os.environ["PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

Base.metadata.create_all(bind=engine)  

def get_create_agent():
    db = SessionLocal()

    db_agents = db.query(Agent).all()
    if len(db_agents) > 0:
        return db_agents[0].agent_id
    
    agents = list(project_client.agents.list())

    if len(agents) == 0:
        # Create persistent agent
        agent = project_client.agents.create_version(
            agent_name=os.environ["AGENT_NAME"],
            definition=PromptAgentDefinition(
                model=os.environ["MODEL_DEPLOYMENT_NAME"],
                instructions="You are a helpful assistant that answers general questions",
            ),
        )
   
        db.add(Agent(
            agent_id = agent.id,
            agent_name = agent.name,
            agent_model = agent.versions.latest.definition.model
        ))
        db.commit()
        return agent.id
    else:
        for agent in agents:
            agent_id = agent.id
            db.add(Agent(
                agent_id = agent.id,
                agent_name = agent.name,
                agent_model = agent.versions.latest.definition.model
            ))    
        db.commit()
        return agent_id

agent_id = get_create_agent()
openai_client = project_client.get_openai_client()

def get_create_thread():
    db = SessionLocal()
    db_threads = db.query(Thread).all()
    
    if len(db_threads) == 0:
        conversation = openai_client.conversations.create()
        db.add(Thread(
            thread_id = conversation.id,
        ))
        db.commit()
        return conversation.id
    else:
        for thread in db_threads:
            thread_id = thread.thread_id
        return thread_id

thread_id = get_create_thread()

print(f"Created agent, ID: {agent_id}")

response = openai_client.responses.create(
    conversation=thread_id, #Optional conversation context for multi-turn
    extra_body={"agent": {"name": agent_name, "type": "agent_reference"}},
    input="What is the size of France in square miles?",
)

print(response.output_text)