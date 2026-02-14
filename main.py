from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from typing import List, Optional
from urllib.parse import unquote
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import ollama

import time
def get_latency(start_time):
    return round((time.time() - start_time) * 1000, 2) # Returns milliseconds


import time
def get_latency(start_time):
    return round((time.time()-start_time)*1000, 2)


# Global Variables
documents:List[str] = []
bm25_index: Optional[BM25Okapi] = None
faiss_index: Optional[faiss.IndexFlatL2] = None

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

from database import Base, SessionLocal, engine, DocumentChunk

from contextlib import asynccontextmanager
import os
INDEX_FILE = "vector_index.faiss"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP LOGIC
    global faiss_index, bm25_index, documents

    Base.metadata.create_all(bind=engine)

    if os.path.exists(INDEX_FILE):
        faiss_index = faiss.read_index(INDEX_FILE)
        rebuild_indexes()
    else:
        print("No local index found, Initializing empty state")    
    
    # SHUTDOWN LOGIC
    # print("Shutting down, cleaning up resources")

    yield

app = FastAPI(lifespan = lifespan)
def chunk_text(text, size=500, overlap = 100):
    chunks = []
    start = 0

    while start < len(text):
        end = start+size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = start+size-overlap
    return chunks

# Now, modify index rebuilding to fetch from the database and handle FAISS file persistence.
# Rebuild indexes, whenever any doc or text gets uploaded.
def rebuild_indexes():
    global bm25_index, faiss_index, documents

    db = SessionLocal()
    db_chunks = db.query(DocumentChunk).all()
    db.close()

    if not db_chunks:
        return

    all_texts = [chunk.content for chunk in db_chunks]
    documents = all_texts
    ##################################
    # BM25 Index
    ##################################
    # Tokenization
    tokenize_docs = [doc.lower().split() for doc in all_texts]
    bm25_index = BM25Okapi(tokenize_docs)

    # FAISS Index 
    ################################## 
    doc_embeddings = embed_model.encode(all_texts).astype("float32")
    faiss_index = faiss.IndexFlatL2(384)
    faiss_index.add(doc_embeddings)

    # Persis FAISS index to disk
    faiss.write_index(faiss_index, INDEX_FILE)
    print(f"index persisted to {INDEX_FILE}")

# Route to upload text or file
@app.post("/upload")
async def upload(text: Optional[str] = Body(None), file: Optional[UploadFile] = File(None)):
    if file:
        content = await file.read()
        text = content.decode('utf-8')

    if not text or not text.strip():
        raise HTTPException(400, "Text or text file is required")

    if text:
        text = unquote(text)

    # Chunking documents
    chunks = chunk_text(text)

    # documents.extend(chunks)
    db=SessionLocal()
    try:
        for  chunk in chunks:
            db.add(DocumentChunk(content=chunk))
        db.commit()
    finally:
        db.close()
    # Rebuild indexes
    rebuild_indexes()

    return {"added chunks": len(chunks), "total documents":len(documents)}

@app.get("/delete")
async def delete_docs():
    doc_length = len(documents)
    documents.clear()
    return {
        "total cleared docs": doc_length
    }

@app.get("/list/documents")
async def get_documents():
    db = SessionLocal()
    document_chunks = db.query(DocumentChunk).all()
    return {
        "total length": len(document_chunks),
        "document_chunks": document_chunks
    }



def hybrid_search(query: str = Body(...)) -> List[str]:
    if not documents or bm25_index is None or faiss_index is None:
        return []
    
    # Sparse using BM25 Index
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)

    top_indexes = np.argsort(bm25_scores)[-2:][::-1]
    bm25_results = [documents[i] for i in top_indexes]

    # Dense Indexing using FAISS
    query_embedding = embed_model.encode(query).astype("float32").reshape(1, -1)
    _,dense_indexes = faiss_index.search(query_embedding, 2)

    dense_results = [documents[i] for i in dense_indexes[0]]

    # Combine for bm25 and faiss results
    return list(set(bm25_results+dense_results))


# Rerank function for Cross Encoding
def rerank(query:str, docs: List[str], top_k=2):
    if not docs:
        return []

    # create pairs
    pairs = [[query, doc] for doc in docs]

    # Get relavent scores
    scores = reranker.predict(pairs)

    # Sory by score(highest first) and return top_k
    return [doc for doc, _ in sorted(zip(docs, scores), key= lambda x:x[1], reverse=True)][:top_k]

# RAG Search API
@app.post("/rag/search")
async def rag_search(query: str = Body(...)):
    total_start = time.time()
    if not query.strip():
        raise HTTPException(400, "query is required")

    if not bm25_index or not faiss_index:
        raise HTTPException(400, "Upload document first")

    # Step 1
    retrieval_start = time.time()
    hybrid_candidates = hybrid_search(query)
    retrieval_ms = get_latency(retrieval_start)

    # Step 2 Reranking using CrossEncoder
    rerank_start = time.time()
    top_docs = rerank(query, hybrid_candidates, 2)
    rerank_ms = get_latency(rerank_start)

    # Step 3 - Prompt building with context for the LLM
    llm_start = time.time()
    context = "\n".join(f"{i+1}.{doc}" for i, doc in enumerate(top_docs))
    prompt = f"""Answer based on the context below

Context:
{context}   

Question: {query}

Answer:"""

    # Step 4 - send prompt to local llm or open ai
    response = ollama.chat(model="phi3:latest", messages=[{'role': 'user', 'content': prompt}])
    llm_ms = get_latency(llm_start)
    total_ms = get_latency(total_start)
    return {
        "query":query,
        "answer": response['message']['content'],
        # "context": top_docs,
        "metrices" : {
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "llm_ms": llm_ms,
            "total_ms": total_ms
        }
    }




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port="8000")