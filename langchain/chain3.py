import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

st.set_page_config(page_title="FinOps Router", layout="wide")
st.header("Smart Model Router (Llama 3b vs Qwen 32b)")

# 1. Initialize our models
cheap_model = ChatOllama(model="llama3.2:3b", temperature=0)
power_model = ChatOllama(model="qwen2.5-coder:32b", temperature=0)

# 2. Define the Router Logic
def route_task(input_text):
    # For a real project, you'd use a small LLM to classify this.
    # For this test, let's use a keyword-based heuristic.
    complex_keywords = ["code", "python", "algorithm", "architect", "debug", "complex", "lcel"]
    
    if any(word in input_text.lower() for word in complex_keywords):
        st.info("🧠 Complex task detected. Routing to **Qwen 32b**...")
        return power_model
    else:
        st.success("⚡ Simple task detected. Routing to **Llama 3b** (Saving VRAM)...")
        return cheap_model

# 3. Create the LCEL Chain
# We use RunnableLambda to wrap our routing function
chain = RunnableLambda(route_task) | StrOutputParser()

user_input = st.text_area("Enter your request (e.g., 'Say hello' vs 'Write a Python script')")

if st.button("Run Optimized Chain"):
    if user_input:
        with st.spinner("Processing..."):
            response = chain.invoke(user_input)
            st.markdown(response)
    else:
        st.warning("Please enter some text.")