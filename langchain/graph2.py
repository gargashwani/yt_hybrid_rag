import streamlit as st
import time
from typing import TypedDict, Literal
from pydantic import BaseModel, Field

# LangChain & LangGraph Imports
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# --- 1. SCHEMAS & STATE ---
class AgentState(TypedDict):
    query: str
    category: str
    response: str

class RouteDecision(BaseModel):
    """Router's classification schema."""
    decision: Literal["simple", "complex", "research"] = Field(
        description="Classify query: 'simple' (general), 'complex' (logic/code), or 'research' (live data/prices/news)."
    )

# --- 2. INITIALIZE MODELS & TOOLS ---
# Set temperature to 0 for the router to prevent "thinking" delays
router_llm = ChatOllama(model="phi3:latest", temperature=0)
cheap_llm = ChatOllama(model="llama3.2:3b", temperature=0.3)
power_llm = ChatOllama(model="qwen2.5-coder:32b", temperature=0.5)
ddg_search = DuckDuckGoSearchRun()

# --- 3. DEFINE NODES ---

def router_node(state: AgentState):
    """Few-Shot Routing to ensure 'Price' triggers Research."""
    system_instructions = """You are a precision router. Classify based on these rules:
    - 'research': ONLY for real-time info, current stock/crypto prices, weather, or news.
    - 'complex': For writing code, math problems, or deep logic.
    - 'simple': For general knowledge (e.g. 'Who is Einstein'), greetings, or chat.
    
    Examples:
    'Price of Bitcoin' -> research
    'Nvidia stock today' -> research
    'Write a FastAPI app' -> complex
    'How are you?' -> simple
    """
    
    structured_router = router_llm.with_structured_output(RouteDecision)
    # Combining instructions with query
    result = structured_router.invoke(f"{system_instructions}\n\nUser Query: {state['query']}")
    return {"category": result.decision}

# --- 4. CONSTRUCT THE GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.set_entry_point("router")
workflow.add_edge("router", END)
app = workflow.compile()

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="Lead Agentic Research", layout="wide")
st.title("🚀 Professional Research Agent")

user_query = st.text_input("Ask me anything:", placeholder="e.g., 'What is the current BTC price?' or 'Write a merge sort in Python'")

if st.button("Execute"):
    if user_query:
        start_time = time.time()
        initial_state = {"query": user_query, "category": "", "response": ""}
        
        # PHASE 1: ROUTING & OPTIONAL SEARCH
        with st.status("🧠 Analyzing Intent & Gathering Data...", expanded=True) as status:
            # 1. Determine Intent
            graph_output = app.invoke(initial_state)
            category = graph_output["category"]
            st.write(f"✅ Category identified: **{category.upper()}**")
            
            # 2. Logic for Node selection and context injection
            if category == "research":
                st.write("🔍 Searching DuckDuckGo for live data...")
                search_results = ddg_search.run(user_query)
                selected_llm = cheap_llm 
                model_name = "Llama 3.2 (Grounded in Web Search)"
                # Crucial: This prompt forces the model to use the web data
                prompt_input = f"Using this LIVE DATA, answer the query accurately: {user_query}\n\nLive Data: {search_results}"
            elif category == "complex":
                selected_llm = power_llm
                model_name = "Qwen 2.5 (32B)"
                prompt_input = user_query
            else:
                selected_llm = cheap_llm
                model_name = "Llama 3.2 (3B)"
                prompt_input = user_query
            
            status.update(label=f"✅ Routing Complete: {model_name}", state="complete")

        # PHASE 2: STREAMING THE RESULT
        st.markdown(f"### 🤖 Output from {model_name}")
        
        output_placeholder = st.empty()
        full_content = ""
        
        # Re-creating a streaming chain for the worker
        prompt = ChatPromptTemplate.from_template("{input}")
        chain = prompt | selected_llm | StrOutputParser()
        
        for chunk in chain.stream({"input": prompt_input}):
            full_content += chunk
            output_placeholder.markdown(full_content + "▌")
        
        output_placeholder.markdown(full_content)
        
        # PHASE 3: METRICS
        total_latency = time.time() - start_time
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Latency", f"{total_latency:.2f}s")
        col2.metric("Routing Decision", category.capitalize())
        col3.metric("Search Active", "Yes" if category == "research" else "No")
            
    else:
        st.error("Please enter a query.")

st.sidebar.markdown("""
### 🏗️ Technical Architecture
- **Router:** Phi-3 (Few-Shot Prompted)
- **Web Retrieval:** DuckDuckGo
- **Inference:** Ollama (Local)
- **State Management:** LangGraph
""")