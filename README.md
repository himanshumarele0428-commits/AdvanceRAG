# 🚀 Advanced RAG Explorer

Advanced RAG Explorer is a high-performance, production-ready **Retrieval-Augmented Generation (RAG)** application specifically designed for managing and exploring regression test cases. It leverages state-of-the-art semantic search, cross-encoder re-ranking, and high-speed LLM synthesis to provide transparent and accurate insights into your testing data.

## ✨ Key Features

- **Multi-Stage RAG Pipeline**: Explicitly separates Ingestion, Semantic Retrieval, Re-ranking, and LLM Synthesis for maximum accuracy.
- **Smart Ingestion**: Supports CSV and Excel files with automatic deduplication and chunking.
- **Advanced Retrieval**: Uses **ChromaDB** with `all-MiniLM-L6-v2` embeddings for deep semantic understanding.
- **AI Re-ranking**: Integrates **FlashRank** (`ms-marco-MiniLM-L-12-v2`) to prioritize the most relevant context before LLM generation.
- **Groq Integration**: High-speed inference using the latest models like **Llama 3.3 70B**, **Deepseek R1**, and **Gemma 2**.
- **Process Transparency**: Real-time "Process Trace" allows users to see exactly which chunks were retrieved and how they were ranked.
- **Knowledge Base Browser**: A dedicated side panel to browse all ingested data chunks in real-time.

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI
- **Orchestration**: LangChain
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **Re-ranker**: FlashRank (`ms-marco-MiniLM-L-12-v2`)
- **LLM API**: Groq Cloud

### Frontend
- **Framework**: React + Vite
- **Styling**: Vanilla CSS (Custom Design System)
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **API Client**: Axios

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+
- Groq API Key ([Get it here](https://console.groq.com/))

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python main.py
```
The backend will run on `http://localhost:8000`.

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
The frontend will run on `http://localhost:5173`.

## 📂 Project Structure

```text
Project18_AdvanceRAG/
├── backend/            # FastAPI Server
│   ├── main.py         # Application logic & RAG pipeline
│   ├── requirements.txt
│   └── vector_db/      # Persisted ChromaDB storage
├── frontend/           # React Application
│   ├── src/
│   │   ├── App.jsx     # Main UI logic
│   │   └── index.css   # Custom styling
│   └── package.json
└── sample_test_cases.csv # Sample data for testing
```

## 📝 License
This project is for educational and professional demonstration purposes.

---
Built with ❤️ by Antigravity.
