from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

st.header("LCEL Research Tool (Ollama)")

# 1. Initialize Components
model = ChatOllama(model="qwen2.5-coder:32b", temperature=0.5)
parser = StrOutputParser()

# 2. Define a Prompt Template (The "Input" stage of LCEL)
prompt = ChatPromptTemplate.from_template(
    "You are a professional researcher. Provide a concise summary of: {topic}"
)

# 3. CONSTRUCT THE LCEL CHAIN (The "Pipe" Logic)
# Input -> Prompt -> Model -> Parser -> Output
chain = prompt | model | parser

user_input = st.text_input("Enter a topic to summarize")

if st.button("Summarise"):
    if user_input:
        with st.spinner("Thinking..."):
            # 4. EXECUTE THE CHAIN
            # Note: We pass a dictionary because the prompt expects {topic}
            # result = chain.invoke({"topic": user_input})
            # st.write(result)
            # Streaming output token-by-token
            response_placeholder = st.empty()
            full_response = ""

            for chunk in chain.stream({"topic": user_input}):
                full_response += chunk
                response_placeholder.markdown(full_response)
    else:
        st.warning("Please enter a topic.")