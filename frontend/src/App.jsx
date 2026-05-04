import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { 
  Upload, 
  Send, 
  Database, 
  Search, 
  Layers, 
  ChevronDown, 
  ChevronUp, 
  FileText, 
  Loader2,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE = "http://localhost:8000";

const TraceStep = ({ title, icon: Icon, children, defaultOpen = false }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  return (
    <div className="trace-container">
      <div className="trace-header" onClick={() => setIsOpen(!isOpen)}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Icon size={16} />
          <span>{title}</span>
        </div>
        {isOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </div>
      <AnimatePresence>
        {isOpen && (
          <motion.div 
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="trace-body"
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

function App() {
  const [file, setFile] = useState(null);
  const [ingestionStats, setIngestionStats] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isQuerying, setIsQuerying] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [selectedModel, setSelectedModel] = useState("llama-3.3-70b-versatile");
  const [models, setModels] = useState([
    { id: "llama-3.3-70b-versatile", name: "Llama 3.3 70B Versatile" },
    { id: "llama-3.1-8b-instant", name: "Llama 3.1 8B Instant" },
    { id: "llama3-70b-8192", name: "Llama 3 70B" },
    { id: "llama3-8b-8192", name: "Llama 3 8B" },
    { id: "mixtral-8x7b-32768", name: "Mixtral 8x7B" },
    { id: "gemma2-9b-it", name: "Gemma 2 9B IT" },
    { id: "deepseek-r1-distill-llama-70b", name: "Deepseek R1 Llama 70B" }
  ]);
  const [isTestingKey, setIsTestingKey] = useState(false);
  const [keyStatus, setKeyStatus] = useState(null); // 'success', 'error', null
  const [showAllChunks, setShowAllChunks] = useState(false);
  
  const fileInputRef = useRef(null);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isQuerying]);

  const testConnection = async () => {
    if (!apiKey) return;
    setIsTestingKey(true);
    setKeyStatus(null);
    try {
      await axios.post(`${API_BASE}/test-connection`, { 
        api_key: apiKey,
        model: selectedModel 
      });
      setKeyStatus('success');
    } catch (err) {
      setKeyStatus('error');
      alert("Connection failed: " + (err.response?.data?.detail || err.message));
    } finally {
      setIsTestingKey(false);
    }
  };

  const handleFileUpload = async (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;
    
    setFile(uploadedFile);
    setIsUploading(true);
    
    const formData = new FormData();
    formData.append("file", uploadedFile);
    
    try {
      const res = await axios.post(`${API_BASE}/upload`, formData);
      setIngestionStats(res.data);
    } catch (err) {
      console.error(err);
      alert("Upload failed: " + (err.response?.data?.detail || err.message));
    } finally {
      setIsUploading(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isQuerying) return;
    
    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsQuerying(true);
    
    try {
      const res = await axios.post(`${API_BASE}/query`, { 
        query: input,
        api_key: apiKey,
        model: selectedModel
      });
      const aiMsg = { 
        role: 'ai', 
        content: res.data.answer,
        trace: {
          original_chunks: res.data.original_chunks,
          reranked_chunks: res.data.reranked_chunks,
          stats: res.data.stats
        }
      };
      setMessages(prev => [...prev, aiMsg]);
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { 
        role: 'ai', 
        content: "Sorry, I encountered an error: " + (err.response?.data?.detail || err.message),
        error: true 
      }]);
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar - Ingestion Panel */}
      <aside className="sidebar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '32px' }}>
          <div style={{ background: 'var(--accent)', padding: '8px', borderRadius: '8px', color: 'white' }}>
            <Layers size={24} />
          </div>
          <h2 style={{ fontSize: '1.2rem', fontWeight: '700' }}>RAG Explorer</h2>
        </div>

        {/* API Settings Section */}
        <div style={{ marginBottom: '24px', padding: '16px', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius)' }}>
          <h3 style={{ fontSize: '0.75rem', fontWeight: '700', marginBottom: '12px', color: 'var(--text-secondary)' }}>LLM CONNECTION</h3>
          <input 
            type="password" 
            placeholder="Enter Groq API Key" 
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            style={{ width: '100%', padding: '10px', borderRadius: '8px', border: '1px solid var(--border)', fontSize: '0.8rem', marginBottom: '10px' }}
          />
          
          <h4 style={{ fontSize: '0.65rem', fontWeight: '700', marginBottom: '8px', color: 'var(--text-secondary)' }}>SELECT MODEL</h4>
          <select 
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            style={{ width: '100%', padding: '10px', borderRadius: '8px', border: '1px solid var(--border)', fontSize: '0.8rem', marginBottom: '12px', background: 'white' }}
          >
            {models.map(m => (
              <option key={m.id} value={m.id}>{m.name}</option>
            ))}
          </select>

          <button 
            className="send-btn" 
            onClick={testConnection} 
            disabled={isTestingKey || !apiKey}
            style={{ width: '100%', fontSize: '0.8rem', padding: '10px' }}
          >
            {isTestingKey ? "Testing..." : "Test Connection"}
          </button>
          {keyStatus === 'success' && <p style={{ fontSize: '0.7rem', color: '#10b981', marginTop: '8px' }}>✓ Connected to Groq</p>}
        </div>

        <div style={{ marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent)' }}></div>
          <h3 style={{ fontSize: '0.85rem', fontWeight: '700' }}>STAGE 1: INGESTION</h3>
        </div>

        <div className="upload-zone" onClick={() => fileInputRef.current.click()} style={{ border: file ? '2px solid var(--accent)' : '2px dashed var(--border)' }}>
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileUpload} 
            style={{ display: 'none' }}
            accept=".csv, .xlsx, .xls"
          />
          <Upload size={32} style={{ color: file ? 'var(--accent)' : 'var(--text-secondary)', marginBottom: '12px' }} />
          <p style={{ fontSize: '0.9rem', fontWeight: '600' }}>
            {file ? "File Selected" : "Upload Test Cases"}
          </p>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            {file ? file.name : "Click to browse CSV/Excel"}
          </p>
        </div>

        {file && !isUploading && !ingestionStats && (
          <button 
            className="send-btn" 
            onClick={() => handleFileUpload({ target: { files: [file] } })}
            style={{ width: '100%', marginBottom: '20px', padding: '12px' }}
          >
            Start Ingestion
          </button>
        )}

        {isUploading && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'center', margin: '20px 0' }}>
            <Loader2 className="animate-spin" size={18} />
            <span style={{ fontSize: '0.85rem' }}>Processing Chunks...</span>
          </div>
        )}

        {ingestionStats && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <h3 style={{ fontSize: '0.85rem', fontWeight: '600', marginBottom: '16px', color: 'var(--text-secondary)' }}>
              INGESTION STATUS
            </h3>
            
            <div className="stats-card">
              <h4>Total Rows</h4>
              <p>{ingestionStats.total_rows}</p>
            </div>
            
            <div className="stats-card">
              <h4>Vector Chunks</h4>
              <p>{ingestionStats.total_chunks}</p>
            </div>

            <div style={{ marginTop: '20px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', color: '#10b981', marginBottom: '8px' }}>
                <CheckCircle2 size={14} />
                <span>Indexed in ChromaDB</span>
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                Chunk Size: {ingestionStats.chunk_size} | Overlap: {ingestionStats.chunk_overlap}
              </div>
            </div>

          </motion.div>
        )}
      </aside>

      {/* Main Content - Chat Area */}
      <main className="main-content">
        <header className="header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Search size={20} color="var(--text-secondary)" />
            <h1>Advanced RAG Pipeline</h1>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Model: <span style={{ color: 'var(--text-primary)', fontWeight: '500' }}>Llama 3 (Groq)</span>
            </div>
          </div>
        </header>

        <div className="chat-container" ref={scrollRef}>
          {messages.length === 0 && !isQuerying && (
            <div style={{ textAlign: 'center', marginTop: '100px', opacity: 0.6 }}>
              <Database size={48} style={{ margin: '0 auto 16px' }} />
              <h3>Ready to explore your test cases</h3>
              <p style={{ fontSize: '0.9rem' }}>Upload a file to start querying your data</p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role} fade-in`}>
              <div className="message-content">
                {msg.role === 'ai' && (
                  <div style={{ fontWeight: '600', marginBottom: '8px', color: 'var(--accent)', fontSize: '0.8rem' }}>
                    CLAUDE RAG ASSISTANT
                  </div>
                )}
                <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                
                {msg.trace && (
                  <div style={{ marginTop: '20px', borderTop: '1px solid var(--border)', paddingTop: '20px' }}>
                    <div style={{ fontSize: '0.75rem', fontWeight: '800', color: 'var(--text-secondary)', marginBottom: '16px', letterSpacing: '0.05em' }}>PROCESS TRACE</div>
                    
                    <TraceStep title={`STAGE 2: INITIAL RETRIEVAL (${msg.trace.stats.initial_retrieval_count} chunks)`} icon={Database}>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        {msg.trace.original_chunks.map((c, i) => (
                          <div key={i} style={{ padding: '12px', background: 'white', borderRadius: '8px', border: '1px solid var(--border)', fontSize: '0.8rem' }}>
                            <div style={{ fontWeight: '700', fontSize: '0.65rem', marginBottom: '6px', color: 'var(--accent)' }}>RETRIEVED CHUNK {i+1}</div>
                            {c.content}
                          </div>
                        ))}
                      </div>
                    </TraceStep>
                    
                    <div style={{ height: '12px' }}></div>
                    
                    <TraceStep title={`STAGE 3: AI RE-RANKING (Top ${msg.trace.stats.reranked_count} selected)`} icon={Layers}>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        {msg.trace.reranked_chunks.map((c, i) => (
                          <div key={i} style={{ padding: '12px', background: 'white', borderRadius: '8px', border: '1px solid var(--accent)', fontSize: '0.8rem', borderLeft: '4px solid var(--accent)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                              <span style={{ fontWeight: '700', fontSize: '0.65rem', color: 'var(--accent)' }}>RE-RANKED CHUNK {i+1}</span>
                              <span style={{ color: 'var(--accent)', fontWeight: '800', fontSize: '0.65rem', background: '#f5f3ff', padding: '2px 6px', borderRadius: '4px' }}>RELEVANCE: {c.score.toFixed(4)}</span>
                            </div>
                            {c.content}
                          </div>
                        ))}
                      </div>
                    </TraceStep>

                    <div style={{ marginTop: '16px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                      <CheckCircle2 size={14} color="#10b981" />
                      <span>Stage 4: LLM Synthesis complete ({msg.trace.stats.model_used})</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}

          {isQuerying && (
            <div className="message ai fade-in">
              <div className="message-content">
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <Loader2 className="animate-spin" size={20} color="var(--accent)" />
                    <span style={{ fontWeight: '600', color: 'var(--accent)' }}>Processing Request...</span>
                  </div>
                  <div style={{ paddingLeft: '32px', display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div className="animate-pulse" style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--accent)' }}></div>
                      <span>Stage 2: Semantic Retrieval</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div className="animate-pulse" style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--accent)' }}></div>
                      <span>Stage 3: Cross-Encoder Re-ranking</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div className="animate-pulse" style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--accent)' }}></div>
                      <span>Stage 4: LLM Context Synthesis (Groq)</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="input-container">
          <div className="input-wrapper">
            <input 
              type="text" 
              placeholder="Ask about test cases, or create a new one for a Jira ID..." 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            />
            <button className="send-btn" onClick={handleSend} disabled={isQuerying}>
              <Send size={18} />
            </button>
          </div>
          <p style={{ textAlign: 'center', fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '12px' }}>
            Advanced RAG Explorer • Powered by Groq & Open Source Embeddings
          </p>
        </div>
      </main>

      {/* Right Panel - All Chunks */}
      <aside className="right-panel">
        <h2>
          <Database size={20} color="var(--accent)" />
          Knowledge Base Chunks
        </h2>
        {!ingestionStats ? (
          <div style={{ textAlign: 'center', marginTop: '40px', opacity: 0.5 }}>
            <p style={{ fontSize: '0.85rem' }}>Ingest data to view chunks here</p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '8px' }}>
              Showing all {ingestionStats.total_chunks} chunks
            </p>
            {ingestionStats.preview.map((chunk, idx) => (
              <div key={idx} className="chunk-card" style={{ maxHeight: 'none', overflow: 'visible' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontWeight: '700', fontSize: '0.65rem', color: 'var(--accent)' }}>CHUNK #{idx + 1}</span>
                  <span style={{ fontSize: '0.6rem', color: 'var(--text-secondary)' }}>ID: {chunk.id}</span>
                </div>
                <div style={{ fontSize: '0.8rem', whiteSpace: 'pre-wrap', lineHeight: '1.4' }}>
                  {chunk.content}
                </div>
              </div>
            ))}
          </div>
        )}
      </aside>

      <style>{`
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

export default App;
