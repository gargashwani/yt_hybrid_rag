import streamlit as st
from typing import Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# --- 1. SCHEMAS ---
class RouteQuery(BaseModel):
    """Classification schema for routing."""
    task_type: Literal["simple", "complex"] = Field(
        description="Classify if the query is simple chat/info (simple) or deep coding/logic (complex)."
    )

# --- 2. INITIALIZE MODELS ---
# Using low temperature (0) for the router to ensure consistent decisions
router_model = ChatOllama(model="phi3:latest", temperature=0)

# Workers
cheap_model = ChatOllama(model="llama3.2:3b", temperature=0.7)
power_model = ChatOllama(model="qwen2.5-coder:32b", temperature=0.7)

# --- 3. UI SETUP ---
st.set_page_config(page_title="Streaming Agent Router", layout="wide")
st.title("🌊 Streaming Agentic Router")

# --- 4. ROUTING LOGIC ---
def get_selected_model(user_input: str):
    """Step 1: The Router makes a decision"""
    # Force the router to use the Pydantic schema
    structured_router = router_model.with_structured_output(RouteQuery)
    decision = structured_router.invoke(user_input)
    
    if decision.task_type == "complex":
        return power_model, "Qwen 2.5 (32B)"
    else:
        return cheap_model, "Llama 3.2 (3B)"

# --- 5. STREAMLIT UI ---
user_query = st.text_area("What can I help you build today?", placeholder="Enter your prompt here...")

if st.button("Generate Response"):
    if user_query:
        # Step A: Identify the model
        with st.status("🤖 AI Orchestrator at work...", expanded=True) as status:
            st.write("Analyzing query complexity...")
            selected_llm, model_name = get_selected_model(user_query)
            st.write(f"Decision: Query is {'complex' if 'Qwen' in model_name else 'simple'}.")
            st.write(f"Selected Model: **{model_name}**")
            status.update(label=f"✅ Routed to {model_name}", state="complete", expanded=False)
        
        # Step B: Create the Worker Chain
        prompt = ChatPromptTemplate.from_template("{query}")
        # Note: StrOutputParser is vital here to turn ChatMessages into raw strings for st.write
        worker_chain = prompt | selected_llm | StrOutputParser()
        
        # Step C: Stream the response
        st.markdown(f"### Response from {model_name}:")
        
        # We use a placeholder to update the text as it streams
        placeholder = st.empty()
        full_response = ""
        
        # Directly iterating over the stream is often more reliable than a separate generator function
        for chunk in worker_chain.stream({"query": user_query}):
            full_response += chunk
            placeholder.markdown(full_response + "▌") # Add a cursor effect
        
        placeholder.markdown(full_response) # Final update to remove cursor
            
    else:
        st.error("Please enter a query.")