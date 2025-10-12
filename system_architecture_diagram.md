# Service Cure Insights - System Architecture Diagram

```mermaid
graph TB
    %% Data Sources
    subgraph "Data Sources"
        A1[4G Network Microservices]
        A2[5G Network Microservices]
        A3[Raw Log Files<br/>(.log, .txt)]
    end

    %% Log Processing Layer
    subgraph "Log Processing Layer"
        B1[Log Ingestion & Parsing<br/>LogGenerator.py]
        B2[Data Structuring<br/>db_setup.py]
    end

    %% Storage Layer
    subgraph "Storage Layer"
        C1[(SQLite3 Database<br/>Structured Data)]
        C2[(ChromaDB<br/>Vector Embeddings)]
        C3[Sentence Transformers<br/>Vectorization Engine]
    end

    %% AI Processing Layer
    subgraph "AI Processing Layer"
        D1[Semantic Search<br/>ChromaDB Query]
        D2[Context Building<br/>Log Correlation]
        D3[Gemini 2.5 Flash API<br/>LLM Integration]
    end

    %% User Interface Layer
    subgraph "User Interface Layer - Streamlit App"
        E1[🤖 Chat Assistant<br/>Natural Language Queries]
        E2[📊 Summary Dashboard<br/>System Health & Trends]
        E3[📋 Error Frequency Table<br/>Pattern Analysis]
    end

    %% User
    F[👤 Network Engineer/Developer]

    %% Data Flow Connections
    A1 --> A3
    A2 --> A3
    A3 --> B1
    B1 --> B2
    B2 --> C1
    B2 --> C3
    C3 --> C2

    %% Query Flow
    F --> E1
    E1 --> D1
    D1 --> C2
    D1 --> D2
    D2 --> C1
    D2 --> D3
    D3 --> E1

    %% Dashboard Flow
    C1 --> E2
    C1 --> E3
    E2 --> F
    E3 --> F

    %% Styling
    classDef dataSource fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef storage fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef ai fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef ui fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef user fill:#f1f8e9,stroke:#33691e,stroke-width:3px

    class A1,A2,A3 dataSource
    class B1,B2 processing
    class C1,C2,C3 storage
    class D1,D2,D3 ai
    class E1,E2,E3 ui
    class F user
```

## Technical Workflow Summary

### 1. Data Ingestion Flow
```
4G/5G Network Microservices → Raw Log Files → Log Ingestion & Parsing → SQLite3 Database
```

### 2. Vectorization Flow
```
SQLite3 Data → Sentence Transformers → Vector Embeddings → ChromaDB Storage
```

### 3. AI Query Processing Flow
```
User Query → Vector Search (ChromaDB) → Context Building → Gemini API → Chat Response
```

### 4. Analytics Flow
```
SQLite3 Database → Summary Dashboard & Error Frequency Table → User Interface
```

## Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Log Ingestion** | Python, LogGenerator | Parse and structure raw log files |
| **Relational Storage** | SQLite3 | Fast queries and analytics |
| **Vector Storage** | ChromaDB | Semantic similarity search |
| **Embeddings** | Sentence Transformers | Convert text to vectors |
| **AI Processing** | Gemini 2.5 Flash API | Generate insights and responses |
| **User Interface** | Streamlit | Interactive web application |

## Data Flow Architecture

1. **Input**: Raw logs from network microservices
2. **Processing**: Parsing, structuring, and vectorization
3. **Storage**: Dual database architecture (SQL + Vector)
4. **Intelligence**: AI-powered semantic search and analysis
5. **Output**: Interactive dashboards and chat interface
