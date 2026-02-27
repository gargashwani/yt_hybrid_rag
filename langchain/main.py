from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

load_dotenv()

# 1. Initialize the model using your Azure credentials
# This replaces the need for an Anthropic API Key
model = init_chat_model(
    model="gpt-4o-mini",              # Your Azure deployment name
    model_provider="azure_openai",
    api_version="2024-02-15-preview",
    temperature=0
)

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's rainy in {city}!"

# 2. The Agent stays exactly the same!
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# 3. Run it
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

print(response["messages"][-1].content)