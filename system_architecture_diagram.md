# Service Cure Insights - System Architecture Diagram

```mermaid
graph TB
    %% Phase 1: Log Ingestion of Microservices
    subgraph "🔄 PHASE 1: Log Ingestion of Microservices"
        direction TB
        subgraph "Data Sources"
            A1[4G Network Microservices]
            A2[5G Network Microservices]
            A3[Raw Log Files<br/>(.log, .txt)]
        end
        
        subgraph "Log Processing"
            B1[Log Ingestion & Parsing<br/>LogGenerator.py]
            B2[Data Structuring<br/>db_setup.py]
        end
        
        subgraph "Storage Infrastructure"
            C1[(SQLite3 Database<br/>Structured Data)]
            C2[(ChromaDB<br/>Vector Embeddings)]
            C3[Sentence Transformers<br/>Vectorization Engine]
        end
    end

    %% Phase 2: Tool Utilization
    subgraph "🛠️ PHASE 2: Tool Utilization & Interface"
        direction TB
        subgraph "AI Processing Engine"
            D1[Semantic Search<br/>ChromaDB Query]
            D2[Context Building<br/>Log Correlation]
            D3[Gemini 2.5 Flash API<br/>LLM Integration]
        end
        
        subgraph "User Interface Tools - Streamlit App"
            E1[🤖 Chat Assistant<br/>Natural Language Queries]
            E2[📊 Summary Dashboard<br/>System Health & Trends]
            E3[📋 Error Frequency Table<br/>Pattern Analysis]
        end
    end

    %% Phase 3: Insights & Solutions
    subgraph "💡 PHASE 3: Insights & Solutions for Smooth Analysis"
        direction TB
        F1[🔍 Key Issue Identification<br/>Pattern Recognition & Root Cause Analysis]
        F2[⚡ Solution Recommendations<br/>Automated Troubleshooting Guidance]
        F3[📈 Performance Optimization<br/>Proactive Monitoring & Alerts]
        F4[👤 Network Engineer/Developer<br/>Decision Making & Implementation]
    end

    %% Phase 1 Internal Flow
    A1 --> A3
    A2 --> A3
    A3 --> B1
    B1 --> B2
    B2 --> C1
    B2 --> C3
    C3 --> C2

    %% Phase 1 to Phase 2 Flow
    C1 --> D2
    C2 --> D1
    C1 --> E2
    C1 --> E3

    %% Phase 2 Internal Flow
    D1 --> D2
    D2 --> D3
    D3 --> E1

    %% Phase 2 to Phase 3 Flow
    E1 --> F1
    E2 --> F1
    E3 --> F1
    F1 --> F2
    F2 --> F3
    F3 --> F4

    %% User Interaction Flow
    F4 --> E1
    F4 --> E2
    F4 --> E3

    %% Styling
    classDef phase1 fill:#e3f2fd,stroke:#1565c0,stroke-width:3px
    classDef phase2 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
    classDef phase3 fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px
    classDef dataSource fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef storage fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef ai fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef ui fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef insights fill:#f1f8e9,stroke:#33691e,stroke-width:2px

    class A1,A2,A3 dataSource
    class B1,B2 processing
    class C1,C2,C3 storage
    class D1,D2,D3 ai
    class E1,E2,E3 ui
    class F1,F2,F3,F4 insights
```

## Three-Phase Architecture Summary

### 🔄 Phase 1: Log Ingestion of Microservices
**Objective**: Capture, process, and store log data from network microservices
```
4G/5G Network Microservices → Raw Log Files → Log Ingestion & Parsing → Data Structuring → 
SQLite3 Database + Vector Embeddings (ChromaDB)
```
**Key Components**: LogGenerator.py, db_setup.py, Sentence Transformers

### 🛠️ Phase 2: Tool Utilization & Interface
**Objective**: Provide interactive tools for log analysis and monitoring
```
Stored Data → AI Processing (Semantic Search + Context Building + Gemini API) → 
User Interface Tools (Chat Assistant + Summary Dashboard + Error Frequency Table)
```
**Key Tools**: 
- **Chat Assistant**: Natural language queries for specific log investigations
- **Summary Dashboard**: System health trends and performance metrics  
- **Error Frequency Table**: Pattern analysis and error categorization

### 💡 Phase 3: Insights & Solutions for Smooth Analysis
**Objective**: Generate actionable insights and solutions for microservice optimization
```
Tool Outputs → Key Issue Identification → Solution Recommendations → 
Performance Optimization → Network Engineer Decision Making
```
**Key Outcomes**:
- **Pattern Recognition**: Automated identification of recurring issues
- **Root Cause Analysis**: Deep dive into system bottlenecks
- **Proactive Solutions**: Preventive measures and optimization recommendations
- **Smooth Operations**: Enhanced microservice reliability and performance

## Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Log Ingestion** | Python, LogGenerator | Parse and structure raw log files |
| **Relational Storage** | SQLite3 | Fast queries and analytics |
| **Vector Storage** | ChromaDB | Semantic similarity search |
| **Embeddings** | Sentence Transformers | Convert text to vectors |
| **AI Processing** | Gemini 2.5 Flash API | Generate insights and responses |
| **User Interface** | Streamlit | Interactive web application |

## Phase-Based Data Flow Architecture

### Phase 1 Flow: Log Ingestion Pipeline
1. **Input**: Raw logs from 4G/5G network microservices
2. **Processing**: Automated parsing, structuring, and vectorization
3. **Storage**: Dual database architecture (SQLite3 + ChromaDB)

### Phase 2 Flow: Tool Utilization Pipeline  
4. **AI Processing**: Semantic search, context building, and LLM integration
5. **Interface**: Interactive tools (Chat Assistant, Dashboard, Frequency Table)

### Phase 3 Flow: Insights & Solutions Pipeline
6. **Analysis**: Pattern recognition and root cause identification
7. **Recommendations**: Automated troubleshooting and optimization guidance
8. **Implementation**: Network engineer decision-making for smooth microservice operations
