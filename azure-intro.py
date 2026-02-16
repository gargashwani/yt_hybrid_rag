from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
load_dotenv()
import uuid
from database import Base, Thread, SessionLocal, engine, Agent
project_client = AIProjectClient(
    endpoint="https://codesips-2236-resource.services.ai.azure.com/api/projects/codesips-2236",
    credential=DefaultAzureCredential(),
)

# models = project_client.get_openai_client(api_version="2024-10-21")
# response = models.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful writing assistant"},
#         {"role": "user", "content": "Write me a poem about flowers"},
#     ],
# )

# print(response.choices[0].message.content)

Base.metadata.create_all(bind=engine)  

def get_create_agent():
    db = SessionLocal()

    db_agents = db.query(Agent).all()
    if len(db_agents) > 0:
        return db_agents[0].agent_id
    
    agents = list(project_client.agents.list_agents())
    
    if len(agents) == 0:
        # Create persistent agent
        agent = project_client.agents.create_agent(
            model="gpt-4o-mini",
            name="CodeSips-Assistant",
            instructions="You are a senior backend developer helping with Python and Azure architecture.",
        )
        db.add(Agent(
            agent_id = agent.id,
            agent_name = agent.name,
            agent_model = agent.model 
        ))
        db.commit()
        return agent.id
    else:
        for agent in agents:
            agent_id = agent.id
            db.add(Agent(
                agent_id = agent.id,
                agent_name = agent.name,
                agent_model = agent.model 
            ))    
        db.commit()
        return agent_id

agent_id = get_create_agent()

def get_create_thread():
    db = SessionLocal()
    db_threads = db.query(Thread).all()

    if len(db_threads) == 0:
        # Create persistent thread
        thread = project_client.agents.threads.create()
        db.add(Thread(
            thread_id = thread.id,
        ))
        db.commit()
        return thread.id
    else:
        for thread in db_threads:
            thread_id = thread.thread_id
        return thread_id

thread_id = get_create_thread()


print(f"Created agent, ID: {agent_id}")

# create an agent thread
# thread = project_client.agents.threads.create()
# print(f"thread id: {thread.id}")

# add message to a thread
message = project_client.agents.messages.create(
    thread_id=thread_id,
    role="user",
    content="What are the benefits of using an Agent over a simple LLM call?"
)

# 5. Run the Agent on the Thread
# 'create_and_process' is the standard method for the Foundry SDK to wait for completion
run = project_client.agents.runs.create_and_process(
    thread_id=thread_id, 
    agent_id=agent_id
)

# print(f"Run Status: {run.status}")

# 6. Retrieve and print the response
if run.status == "completed":
    # Get the messages (this returns an iterator)
    messages = project_client.agents.messages.list(thread_id=thread_id)
    
    # Convert the iterator to a list to access by index
    messages_list = list(messages)
    
    # The newest message is at index 0
    last_msg = messages_list[0]
    
    if last_msg.role == "assistant":
        # Access the text content properly
        # Content is a list of MessageContent objects
        print(f"\nAgent Response: {last_msg.content[0].text.value}")

