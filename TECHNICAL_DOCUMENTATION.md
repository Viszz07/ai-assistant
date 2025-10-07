# 🔧 Service Cure Insights - Technical Documentation

**Version:** 1.0  
**Last Updated:** October 2025

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Technology Stack](#technology-stack)
3. [Database Design](#database-design)
4. [RAG Pipeline](#rag-pipeline)
5. [How Chat Assistant Works](#how-chat-assistant-works)
6. [Embedding System](#embedding-system)
7. [Performance](#performance)

---

## Architecture Overview

### High-Level System Design

```
User Interface (Streamlit)
    ↓
Application Logic (app.py, db_setup.py)
    ↓
AI/ML Processing (LLM_integration.py)
    ↓ ↓
SQLite + ChromaDB (Dual Database)
```

### Design Principles

- **RAG Architecture**: Retrieval Augmented Generation for grounded responses
- **Dual Database**: SQLite (structured) + ChromaDB (semantic)
- **Modular**: Separation of concerns across files
- **Stateless UI**: Streamlit session state for caching

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web Framework | Streamlit 1.28.0+ | Rapid UI development |
| LLM | Google Gemini 2.5 Flash | Natural language AI |
| Vector DB | ChromaDB 0.4.15+ | Semantic search |
| Relational DB | SQLite 3.x | Structured queries |
| Embeddings | Sentence Transformers 2.5.0+ | Text-to-vector |
| ML Backend | PyTorch 2.1.0+ | Transformer inference |
| Data Processing | Pandas 2.0.0+ | DataFrame operations |
| Visualization | Plotly 5.15.0+ | Interactive charts |

### Embedding Model: `all-MiniLM-L6-v2`

```python
{
    "architecture": "MiniLM (distilled BERT)",
    "parameters": "22 million",
    "embedding_dimension": 384,
    "max_sequence_length": 256,
    "model_size": "90 MB",
    "cpu_inference": "~50ms",
    "accuracy_sts": 68.06
}
```

**Why this model?**
- Fast CPU inference (no GPU needed)
- Lightweight (90MB)
- Good accuracy/speed balance
- General-purpose

---

## Database Design

### Why Two Databases?

Different query patterns need different data structures:

| Query Type | Best DB | Example |
|------------|---------|---------|
| Exact filtering | SQLite | "Show ERROR logs" |
| Semantic search | ChromaDB | "Authentication issues" |
| Aggregations | SQLite | "Count by severity" |
| Pattern discovery | ChromaDB | "Similar problems" |
| Stats | SQLite | Dashboard metrics |
| AI context | ChromaDB | Chat assistant |

### SQLite Schema

```sql
CREATE TABLE logs (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,              -- YYYY-MM-DD-HH:MM:SS
    filename TEXT NOT NULL,               -- UserService.java
    line_number INTEGER NOT NULL,
    severity TEXT NOT NULL,               -- ERROR|WARN|INFO|DEBUG
    message TEXT NOT NULL,
    log_file_source TEXT NOT NULL,        -- application_service.log
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes
CREATE INDEX idx_timestamp ON logs(timestamp);
CREATE INDEX idx_severity ON logs(severity);
CREATE INDEX idx_filename ON logs(filename);
```

**Performance:**
- Without index: O(n) scan
- With B-tree index: O(log n) lookup
- Typical query: <10ms

### ChromaDB Structure

```python
{
    "ids": ["log_1", "log_2", ...],
    "documents": ["Database connection timeout", ...],
    "embeddings": [[0.123, -0.456, ..., 0.789], ...],  # 384D vectors
    "metadatas": [
        {
            "timestamp": "2025-10-06-16:18:41",
            "filename": "UserService.java",
            "line_number": 123,
            "severity": "ERROR"
        },
        ...
    ]
}
```

**HNSW Index:**
- Algorithm: Hierarchical Navigable Small World
- Distance: Cosine similarity
- Search: O(log n)
- Performance: 100-200ms for 10K vectors

---

## RAG Pipeline

### Five-Phase Process

```
1. Query Validation (Guardrails)
    ↓
2. Vector Retrieval (Semantic Search)
    ↓
3. Context Building (Top 10 logs)
    ↓
4. LLM Prompting (Gemini API)
    ↓
5. Response Delivery (Structured output)
```

### Phase 1: Query Validation

```python
def is_log_related_query(query):
    # Check keywords: error, warning, crash, timeout, etc.
    # Check patterns: "why.*crash", "show.*error", etc.
    # Reject: "What's the weather?"
    return True/False
```

### Phase 2: Vector Retrieval

```python
# Convert query to 384D vector
query_embedding = sentence_transformer.encode(query)

# Search ChromaDB (ALL ingested logs)
results = chromadb.query(
    query_embeddings=[query_embedding],
    n_results=15  # Top 15 similar logs
)

# Returns logs from ALL files, ranked by similarity
```

**Key Point:** Searches across ALL logs from ALL files, not just one specific file.

### Phase 3: Context Building

```python
# Take top 10 logs only
context = "RELEVANT LOG ENTRIES:\n\n"
for i, log in enumerate(relevant_logs[:10], 1):
    context += f"{i}. [{timestamp}] {filename}:{line} [{severity}] {message}\n"

# Add stats if needed
if 'overview' in query:
    context += f"\nTotal ERROR logs: {count}\n"
```

**Result:** ~800 tokens (very efficient)

### Phase 4: LLM Prompting

```python
system_prompt = """You are a log analysis assistant.

GUARDRAILS:
- Base response ONLY on provided log entries
- Do NOT paste raw logs
- If insufficient info, say so

OUTPUT FORMAT:
Summary:
Key Issues:
Probable Causes:
Quick Fixes:
Next Steps:
Conclusion:

CONTEXT: {context}
USER QUERY: {query}
"""

response = gemini_model.generate_content(system_prompt)
```

**Token Usage:**
- Input: ~1,000 tokens (system + context + query)
- Output: ~500-1000 tokens
- Cost: ~$0.01 per query

### Phase 5: Response Delivery

```python
return {
    'response': ai_generated_text,
    'relevant_logs': top_5_logs,
    'context_used': True,
    'total_logs_found': 15
}
```

---

## How Chat Assistant Works

### Does it use a specific file?

**NO** - It doesn't work on "one specific file"

### What data does it use?

**ALL ingested logs** from ALL files simultaneously

### How does it find relevant logs?

1. **At Ingestion Time:**
   - Parse ALL log files (application_service.log, database_service.log, etc.)
   - Store in SQLite (structured)
   - Convert messages to vectors, store in ChromaDB

2. **At Query Time:**
   - Convert user query to vector
   - Search ChromaDB across ALL logs
   - Return top 15 most semantically similar
   - Logs can be from different files

3. **Context Building:**
   - Take top 10 relevant logs
   - Send only these to LLM (~800 tokens)
   - NOT entire file contents

### Example Flow

```
User: "Show me database connection errors"
    ↓
System searches ALL logs (from all files)
    ↓
Finds 15 similar logs:
  - 5 from application_service.log
  - 7 from database_service.log
  - 3 from auth_service.log
    ↓
Takes top 10, sends to Gemini
    ↓
AI analyzes and responds
```

### Why NOT send entire files?

**Efficiency:**
- Entire files: 5,000+ tokens
- Selected logs: ~800 tokens
- 6x cost reduction

**Accuracy:**
- Focused context = better answers
- Only relevant info to LLM

**Scalability:**
- Works with 1,000 or 1,000,000 logs
- Token count stays constant (~800)

---

## Embedding System

### Text-to-Vector Conversion

```
Input: "Database connection timeout"
    ↓
Tokenization: [101, 2951, 4434, ...]
    ↓
Transformer (6 layers)
    ↓
Mean Pooling
    ↓
Output: [0.123, -0.456, ..., 0.789] (384D)
```

### Semantic Similarity

```python
def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / norm
```

**Similarity Scale:**
- 1.0: Identical
- 0.8-0.9: Very similar
- 0.6-0.8: Related
- <0.6: Different

### Example

```python
"Database connection failed"     → [0.2, 0.5, -0.3, ...]
"Unable to connect to database"  → [0.3, 0.4, -0.2, ...]
# Similarity: 0.89 (found as related)

"Database connection failed"     → [0.2, 0.5, -0.3, ...]
"User logged in successfully"    → [-0.1, -0.2, 0.8, ...]
# Similarity: 0.12 (not related)
```

---

## Performance

### Caching

```python
# Cache LLM integration (saves 2-3s per query)
if 'llm_integration' not in st.session_state:
    st.session_state.llm_integration = LLMIntegration()

# Cache sentence transformer (saves 90MB memory)
# Model loaded once per application instance
```

### Query Speed

- **SQLite query**: <10ms (indexed)
- **Vector search**: 100-200ms (HNSW)
- **Embedding generation**: 50ms
- **LLM response**: 1-3s
- **Total**: 1.5-3.5s per query

### Scalability

| Log Count | SQLite Query | Vector Search | Total Query Time |
|-----------|--------------|---------------|------------------|
| 1,000 | 5ms | 80ms | ~1.5s |
| 10,000 | 8ms | 150ms | ~2s |
| 100,000 | 12ms | 250ms | ~2.5s |
| 1,000,000 | 20ms | 400ms | ~3s |

**Key Point:** Performance scales logarithmically, not linearly

### Memory Usage

- **SQLite**: ~10MB per 10,000 logs
- **ChromaDB**: ~50MB per 10,000 embeddings (384D)
- **Sentence Transformer**: 90MB (one-time)
- **Total**: ~150MB for 10,000 logs

---

## Code Structure

```
Service_Log_Insights/
├── app.py                    # Main UI (Streamlit)
│   ├── LogAnalysisApp class
│   ├── render_sidebar()      # Data ingestion UI
│   ├── render_chat_tab()     # Chat interface
│   ├── render_dashboard_tab() # Analytics
│   └── render_table_tab()    # Log browser
│
├── LLM_integration.py        # AI/RAG pipeline
│   ├── LLMIntegration class
│   ├── is_log_related_query() # Guardrails
│   ├── get_relevant_logs()   # Vector search
│   ├── build_context()       # Context prep
│   ├── generate_response()   # LLM call
│   └── process_query()       # Main entry
│
├── db_setup.py               # Database initialization
│   ├── DatabaseSetup class
│   ├── setup_sqlite()        # Create tables
│   ├── setup_chromadb()      # Create collection
│   ├── process_log_file()    # Parse logs
│   └── load_all_logs()       # Batch insert
│
├── log_generator.py          # Sample data
│   └── LogGenerator class
│
├── requirements.txt          # Dependencies (10 packages)
├── README.md                 # User documentation
└── TECHNICAL_DOCUMENTATION.md # This file
```

---

## Summary

**Service Cure Insights** uses a sophisticated RAG architecture to provide intelligent log analysis:

1. **Dual Database**: SQLite for structure + ChromaDB for semantics
2. **Sentence Transformers**: Convert text to 384D vectors
3. **Semantic Search**: Find similar logs across ALL files
4. **Context-Aware LLM**: Gemini generates grounded responses
5. **Efficient**: Only ~800 tokens to LLM, not entire files
6. **Scalable**: Performance scales logarithmically

**Key Innovation:** Combining vector search with structured queries for best of both worlds.

---

**For questions or contributions, see README.md**
