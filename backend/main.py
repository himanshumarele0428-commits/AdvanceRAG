import os
import shutil
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid

# LangChain and RAG components
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq
from flashrank import Ranker, RerankRequest

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DB_DIR = "vector_db"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
embeddings = None
vector_db = None
ranker = None

class QueryRequest(BaseModel):
    query: str
    api_key: Optional[str] = None
    model: Optional[str] = "llama-3.3-70b-versatile"

class ConnectionRequest(BaseModel):
    api_key: str
    model: Optional[str] = "llama-3.3-70b-versatile"

class QueryResponse(BaseModel):
    answer: str
    original_chunks: List[dict]
    reranked_chunks: List[dict]
    stats: dict

@app.get("/")
async def root():
    return {"status": "online", "message": "Advanced RAG Backend is running"}

@app.on_event("startup")
async def startup_event():
    print("--- Advanced RAG Backend Started ---")
    if not GROQ_API_KEY:
        print("Note: GROQ_API_KEY not found in environment.")

def get_embeddings():
    global embeddings
    if embeddings is None:
        print("Loading Embedding model (all-MiniLM-L6-v2)...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

def get_ranker():
    global ranker
    if ranker is None:
        print("Loading Ranker model (ms-marco-MiniLM-L-12-v2)...")
        ranker = Ranker()
    return ranker

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vector_db
    emb = get_embeddings()
    
    # Save file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(temp_path)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(temp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Deduplicate rows to prevent duplicate chunks
        df = df.drop_duplicates().reset_index(drop=True)

        # Convert DF to string content for RAG
        # We assume there's a column or we combine all columns
        df['text_content'] = df.apply(lambda x: ' | '.join([f"{col}: {val}" for col, val in x.items() if pd.notna(val)]), axis=1)
        
        loader = DataFrameLoader(df, page_content_column="text_content")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(documents)
        
        # Clear existing DB if any (optional, user might want to append)
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
            
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=emb,
            persist_directory=DB_DIR
        )
        
        # Prepare preview data
        preview_chunks = []
        for i, chunk in enumerate(chunks): 
            preview_chunks.append({
                "id": i,
                "content": chunk.page_content, # Return full content for the right panel
                "metadata": chunk.metadata
            })

        return {
            "status": "success",
            "total_rows": len(df),
            "total_chunks": len(chunks),
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "preview": preview_chunks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/models")
async def get_models():
    return [
        {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B Versatile"},
        {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B Instant"},
        {"id": "llama3-70b-8192", "name": "Llama 3 70B"},
        {"id": "llama3-8b-8192", "name": "Llama 3 8B"},
        {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B"},
        {"id": "gemma2-9b-it", "name": "Gemma 2 9B IT"},
        {"id": "deepseek-r1-distill-llama-70b", "name": "Deepseek R1 Llama 70B"}
    ]

@app.post("/test-connection")
async def test_connection(request: ConnectionRequest):
    try:
        client = Groq(api_key=request.api_key)
        # Simple test call
        client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model=request.model or "llama-3.3-70b-versatile",
            max_tokens=5
        )
        return {"status": "success", "message": "Connection successful"}
    except Exception as e:
        print(f"Connection Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    global vector_db
    emb = get_embeddings()
    rnk = get_ranker()
    
    if not vector_db:
        if os.path.exists(DB_DIR):
            vector_db = Chroma(persist_directory=DB_DIR, embedding_function=emb)
        else:
            raise HTTPException(status_code=400, detail="No data ingested yet")
    
    current_key = request.api_key or GROQ_API_KEY
    if not current_key:
        raise HTTPException(
            status_code=500, 
            detail="GROQ_API_KEY not found. Please provide it in the settings."
        )

    # 1. Retrieval
    docs = vector_db.similarity_search(request.query, k=20) # Get more initially
    
    # Deduplicate docs by content
    seen_content = set()
    unique_docs = []
    for d in docs:
        if d.page_content not in seen_content:
            unique_docs.append(d)
            seen_content.add(d.page_content)
    
    original_chunks = [
        {"content": d.page_content, "metadata": d.metadata} for d in unique_docs[:10]
    ]
    
    # 2. Re-ranking
    passages = [
        {"id": i, "text": d["content"], "meta": d["metadata"]} for i, d in enumerate(original_chunks)
    ]
    rerank_request = RerankRequest(query=request.query, passages=passages)
    results = rnk.rerank(rerank_request)
    
    reranked_chunks = [
        {"content": r['text'], "score": float(r['score']), "metadata": r['meta']} for r in results[:5]
    ]
    
    # 3. Generation
    try:
        client = Groq(api_key=current_key)
        context = "\n\n".join([c["content"] for c in reranked_chunks])
        
        prompt = f"""You are an advanced RAG assistant. Use the following context to answer the user's question. 
Context:
{context}

Question: {request.query}
Answer:"""

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert Test Case Management assistant. Analyze the provided test cases and answer the user's query comprehensively. If the user asks for specific types of test cases (like 'MFA' or 'Login'), summarize the relevant ones you find in the context. If no direct match exists, provide the most related information available in the data. Always be helpful and avoid saying you cannot find information unless the context is truly empty."},
                {"role": "user", "content": prompt},
            ],
            model=request.model or "llama-3.3-70b-versatile",
        )
        answer = chat_completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

    return {
        "answer": answer,
        "original_chunks": original_chunks,
        "reranked_chunks": reranked_chunks,
        "stats": {
            "initial_retrieval_count": len(original_chunks),
            "reranked_count": len(reranked_chunks),
            "model_used": request.model or "llama-3.3-70b-versatile"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
