import streamlit as st
import time
from typing import TypedDict, Literal
from pydantic import BaseModel, Field

# LangChain & LangGraph Imports
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# --- 1. SCHEMAS & STATE ---
class AgentState(TypedDict):
    query: str
    category: str
    response: str

class RouteDecision(BaseModel):
    decision: Literal["simple", "complex"] = Field(
        description="Classify query as 'simple' or 'complex'."
    )

# --- 2. INITIALIZE MODELS ---
router_llm = ChatOllama(model="phi3:latest", temperature=0)
cheap_llm = ChatOllama(model="llama3.2:3b", temperature=0.7)
power_llm = ChatOllama(model="qwen2.5-coder:32b", temperature=0.7)

# --- 3. DEFINE NODES ---

def router_node(state: AgentState):
    start_time = time.time()
    structured_router = router_llm.with_structured_output(RouteDecision)
    result = structured_router.invoke(state["query"])
    duration = time.time() - start_time
    # We store the decision in state
    return {"category": result.decision}

# Note: We will handle the Worker "Streaming" logic inside the Streamlit loop 
# to ensure the UI updates in real-time.

# --- 4. CONSTRUCT THE GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)

# In a pure streaming UI, we often branch the logic in the UI loop 
# after the router node completes its work.
workflow.set_entry_point("router")
workflow.add_edge("router", END)
app = workflow.compile()

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="Streaming Performance Agent", layout="wide")
st.title("🌊 Real-Time Streaming & Latency Monitor")

user_query = st.text_area("Input your request:", height=100)

if st.button("Execute Agent"):
    if user_query:
        current_state = {"query": user_query, "category": "", "response": ""}
        
        start_total = time.time()
        
        # --- PHASE 1: ROUTING ---
        with st.status("🚦 Orchestrating Workflow...", expanded=True) as status:
            st.write("Step 1: Analyzing Intent...")
            
            # Run the router node
            router_start = time.time()
            output = app.invoke(current_state)
            router_end = time.time()
            
            category = output["category"]
            router_dur = router_end - router_start
            
            st.write(f"✅ Router Finished in **{router_dur:.2f}s**")
            st.write(f"Decision: **{category.upper()}**")
            
            # Select the model based on graph decision
            selected_llm = power_llm if category == "complex" else cheap_llm
            model_name = "Qwen 2.5 (32B)" if category == "complex" else "Llama 3.2 (3B)"
            
            status.update(label=f"✅ Routed to {model_name} in {router_dur:.2f}s", state="complete")

        # --- PHASE 2: STREAMING WORKER ---
        st.markdown(f"### 🤖 Response from {model_name}")
        
        # Setup Streaming UI
        placeholder = st.empty()
        full_response = ""
        
        worker_start = time.time()
        
        # Simple LCEL for the streaming worker part
        prompt = ChatPromptTemplate.from_template("{query}")
        chain = prompt | selected_llm | StrOutputParser()
        
        # Streaming loop
        for chunk in chain.stream({"query": user_query}):
            full_response += chunk
            placeholder.markdown(full_response + "▌")
        
        worker_end = time.time()
        worker_dur = worker_end - worker_start
        total_dur = worker_end - start_total
        
        placeholder.markdown(full_response) # Remove cursor
        
        # --- PHASE 3: PERFORMANCE METRICS ---
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Router Latency", f"{router_dur:.2f}s")
        col2.metric("Worker Latency", f"{worker_dur:.2f}s")
        col3.metric("Total Wall Time", f"{total_dur:.2f}s")
            
    else:
        st.error("Please provide a prompt.")

# Footer info for interview talking points
st.sidebar.info("""
**2026 Architectural Highlights:**
- **Asynchronous Handoff:** Router finishes, then Worker streams.
- **TTFT Optimization:** User sees routing decision < 500ms.
- **Observability:** Granular timing per execution block.
""")