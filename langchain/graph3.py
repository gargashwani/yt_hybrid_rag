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
    decision: Literal["simple", "complex", "research"] = Field(...)

# --- 2. INITIALIZE MODELS WITH REPETITION PENALTY ---
# repeat_penalty=1.2 prevents the model from getting stuck in loops
router_llm = ChatOllama(model="phi3:latest", temperature=0, repeat_penalty=1.2)

cheap_llm = ChatOllama(
    model="llama3.2:3b", 
    temperature=0.1, 
    repeat_penalty=1.3, # Stronger penalty to stop loops
    stop=["Some notable", "\n\n\n"] # Stop tokens to kill hallucinations early
)

power_llm = ChatOllama(model="qwen2.5-coder:32b", temperature=0.1, repeat_penalty=1.2)
ddg_search = DuckDuckGoSearchRun()

# --- 3. NODES ---
def router_node(state: AgentState):
    system = "Route to 'research' for prices/news, 'complex' for code/logic, 'simple' for chat."
    structured = router_llm.with_structured_output(RouteDecision)
    res = structured.invoke(f"{system}\nQuery: {state['query']}")
    return {"category": res.decision}

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="Stable Agent v7", layout="wide")
st.title("🛡️ Anti-Loop Research Agent")

# Sidebar to clear state if it ever loops again
if st.sidebar.button("Clear Cache & Reset"):
    st.rerun()

user_query = st.text_input("Query:", placeholder="e.g., '1 BTC to INR price'")

if st.button("Execute"):
    if user_query:
        start_time = time.time()
        
        with st.status("🧠 Processing...", expanded=True) as status:
            # Step 1: Route
            route_res = router_node({"query": user_query})
            category = route_res["category"]
            st.write(f"✅ Path: **{category.upper()}**")

            if category == "research":
                st.write("🔍 Fetching specific market variables...")
                # Targeted search expansion
                price_data = ddg_search.run(f"current {user_query} price USD")
                fx_data = ddg_search.run("current 1 USD to INR exchange rate live")
                
                model_name = "Llama 3.2 (Grounded)"
                selected_llm = cheap_llm
                prompt_input = f"""
                You are a Financial Data Analyst. Use the data below:
                
                DATA 1 (Asset): {price_data}
                DATA 2 (Forex): {fx_data}
                
                TASK:
                1. State the Asset Price in USD.
                2. State the USD/INR rate.
                3. Calculate the INR value (Price * Rate).
                
                If data is missing, say 'Incomplete data found'. Do not repeat yourself.
                """
            elif category == "complex":
                selected_llm = power_llm
                model_name = "Qwen 22b"
                prompt_input = user_query
            else:
                selected_llm = cheap_llm
                model_name = "Llama 3.2"
                prompt_input = user_query
            
            status.update(label="✅ Analysis Ready", state="complete")

        # --- PHASE 2: STREAMING ---
        st.markdown(f"### 🤖 Output ({model_name})")
        placeholder = st.empty()
        full_text = ""
        
        chain = ChatPromptTemplate.from_template("{input}") | selected_llm | StrOutputParser()
        
        for chunk in chain.stream({"input": prompt_input}):
            # Protection: If the model starts repeating common loop phrases, stop it manually
            if "Some notable frameworks" in full_text:
                break
            full_text += chunk

            # Display streaming output with a cursor-like indicator "▌"
            st.markdown(full_text + "▌")
        
        # Final output without the cursor indicator after stream is complete
        st.markdown(full_text)
        st.caption(f"Latency: {time.time() - start_time:.2f}s")