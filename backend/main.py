import os
import shutil
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# LangChain and RAG components
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq
from flashrank import Ranker, RerankRequest

load_dotenv()

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


# Feature keywords for diverse retrieval
FEATURE_KEYWORDS = [
    "login", "logout", "password", "forgot password", "reset",
    "profile", "account", "settings",
    "search", "filter",
    "cart", "add to cart", "remove", "checkout", "payment", "order",
    "registration", "signup", "sign up",
    "navigation", "menu", "dashboard",
    "notification", "email", "message",
    "upload", "download", "import", "export",
]


def extract_search_queries(user_query: str) -> List[str]:
    """Extract targeted search terms from the user query to improve retrieval diversity."""
    q = user_query.lower()

    # If the query asks for "each module/feature" or "per feature", use feature keywords
    if any(phrase in q for phrase in ["each module", "each feature", "per module", "per feature",
                                        "1 test case for each", "one test case for each",
                                        "all modules", "all features", "list all", "group by"]):
        return FEATURE_KEYWORDS + [user_query]

    # Check if query mentions specific features
    mentioned = [kw for kw in FEATURE_KEYWORDS if kw in q]
    if mentioned:
        return mentioned + [user_query]

    # Default: use the query as-is
    return [user_query]


RELEVANCE_THRESHOLD = 0.08

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

    # 1. Retrieval - use extracted search queries for diversity
    search_queries = extract_search_queries(request.query)
    
    all_docs = []
    seen_content = set()
    
    # For "per feature" queries, search with feature keywords to get broad coverage
    if len(search_queries) > 1:
        for sq in search_queries[:10]:  # limit to first 10 queries
            batch = vector_db.similarity_search(sq, k=5)
            for d in batch:
                if d.page_content not in seen_content:
                    all_docs.append(d)
                    seen_content.add(d.page_content)
    else:
        all_docs = vector_db.similarity_search(search_queries[0], k=30)
        # Deduplicate
        uniq = []
        seen = set()
        for d in all_docs:
            if d.page_content not in seen:
                uniq.append(d)
                seen.add(d.page_content)
        all_docs = uniq
    
    original_chunks = [
        {"content": d.page_content, "metadata": d.metadata} for d in all_docs
    ]
    
    # 2. Re-ranking
    passages = [
        {"id": i, "text": d["content"], "meta": d["metadata"]} for i, d in enumerate(original_chunks)
    ]
    rerank_request = RerankRequest(query=request.query, passages=passages)
    results = rnk.rerank(rerank_request)
    
    # Filter by relevance threshold and keep top results
    filtered_results = [r for r in results if float(r['score']) >= RELEVANCE_THRESHOLD]
    if not filtered_results:
        # Fallback: keep at least the top 3 even if below threshold
        filtered_results = results[:3]
    
    reranked_chunks = [
        {"content": r['text'], "score": float(r['score']), "metadata": r['meta']} for r in filtered_results
    ]
    
    # 3. Generation
    try:
        client = Groq(api_key=current_key)
        context = "\n\n---\n\n".join([c["content"] for c in reranked_chunks])
        
        prompt = f"""You are a test case management expert. Analyze the provided test cases and answer the user's question.

Context (test cases from our database):
{context}

User Question: {request.query}

Instructions:
- Each test case has fields: Jira ID, Summary, Description, Steps, Expected Result, Priority
- Based on the Summary and Description, identify which feature/module each test case belongs to (e.g., Login, Search, Cart, Checkout, Profile, Settings, Registration, Payment, etc.)
- If the user asks for "one test case per feature/module" or similar, group ALL test cases by their inferred feature, then pick the best representative for each group
- Format your answer with clear headings for each feature group
- Include Jira ID, Summary, Priority, and Expected Result for each test case listed
- If you cannot determine a feature for a specific test case, list it under "Other / Uncategorized"
- Use ALL test cases from the context — do not leave any out — do not invent test cases
- If no test cases are provided in context, say "No test cases found in the database. Please upload a file first."

Answer:"""

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert Test Case Management assistant. Your job is to analyze the provided test case context and answer queries comprehensively. Always group test cases by their inferred feature/module when the user asks for organization by feature. Infer features from the Summary field (e.g., 'Login' → Login feature, 'Add to cart' → Cart feature). Always include Jira IDs in your answer. If the context is empty, politely state that no test cases are available. Never invent test cases."},
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
            "initial_retrieval_count": len(all_docs),
            "unique_chunks": len(original_chunks),
            "reranked_count": len(reranked_chunks),
            "model_used": request.model or "llama-3.3-70b-versatile",
            "search_queries_used": search_queries[:5]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
