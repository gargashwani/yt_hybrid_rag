# RAG in LangChain is just a retriever composed into an LCEL pipeline. 
# Everything else is preprocessing.

# 1. Document Loader (Source → Documents)
from langchain_community.document_loaders import TextLoader
from typing import List
loader = TextLoader("data/rag_intro.txt")
documents = loader.load()
# Each Document has:
# page_content
# metadata


# print([Document.page_content for Document in documents])

# 2. Text Splitter (Chunking Strategy)
# Chunk size controls retrieval recall
# Overlap preserves semantic continuity
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)
chunks = text_splitter.split_documents(documents)
print([chunk for chunk in chunks])

# 3. Embeddings (Vector Representation)
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(
    model="qwen3-embedding"
)


# 4. Vector Store (FAISS)
# This step:
# Builds index
# Stores vectors
# Enables similarity search
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
) 

# 5. Retriever (THIS is the R in RAG)
# Retriever is a Runnable in LCEL.
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 6. Prompt (RAG-Aware)
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
You are a senior AI engineer.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""")


# 7. LLM + Parser
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.5,
    streaming=True
)

parser = StrOutputParser()


# 8. LCEL RAG PIPELINE (MOST IMPORTANT PART)
from langchain_core.runnables import RunnablePassthrough
rag_chain = (
    {
        "context": retriever,          # 👈 retrieval happens here
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)

# THIS IS THE KEY LINE
# retriever runs automatically
# Retrieved docs → injected into {context}
# LCEL handles dataflow
# Fully streamable
# Fully async
# Fully traceable

# 9. Streaming RAG Answer
print("RAG Streaming Output:\n")

for token in rag_chain.stream(
    "What is Retrieval-Augmented Generation?"
):
    print(token, end="", flush=True)