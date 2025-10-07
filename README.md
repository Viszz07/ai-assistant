# 💊 Service Cure Insights

An intelligent log analysis platform powered by AI that helps diagnose and cure service issues. Built with Streamlit, Google's Gemini 2.5 Flash API, and vector databases for semantic log search.

## 🌟 Features

### 🤖 AI-Powered Chat Assistant
- Natural language queries about your logs
- Context-aware responses using RAG (Retrieval Augmented Generation)
- Guardrails to prevent hallucination - answers only from log data
- Powered by Google Gemini 2.5 Flash API

### 📊 Interactive Dashboard
- Real-time log statistics and health metrics
- Visual charts for error distribution and trends
- Timeline analysis of log events
- Recent critical issues monitoring

### 📋 Advanced Log Table
- Searchable and filterable log entries
- Multi-criteria filtering (severity, filename, content)
- Export functionality for filtered data
- Color-coded severity levels

### 🗄️ Dual Database Architecture
- **SQLite**: Structured storage for fast queries and filtering
- **ChromaDB**: Vector embeddings for semantic search
- **Sentence Transformers**: High-quality text embeddings

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key

### Setup Steps

1. **Clone or download the project**
   ```bash
   cd Service_Log_Insights
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Gemini API key**
   
   **Option A: Environment Variable**
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```
   
   **Option B: Streamlit Secrets**
   Create `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your_api_key_here"
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Ingest logs from the UI (manual ingestion)**
   - In the sidebar, open the "📥 Data Ingestion" section:
     - Click "🧪 Generate Sample Logs and Ingest" to auto-generate and load logs, or
     - Upload `.log`/`.txt` files and click "📦 Ingest Uploaded Logs", or
     - Select existing files within `logs/` using "➡️ Ingest Selected Existing Logs".
   - Once ingested, the Chat, Dashboard, and Table tabs are enabled.

## 📁 Project Structure

```
Service_Log_Insights/
├── app.py                 # Main Streamlit application
├── log_generator.py       # Generates sample log files
├── db_setup.py           # Database initialization
├── LLM_integration.py    # Gemini API integration with guardrails
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── logs/                # Generated log files
│   ├── application_service.log
│   └── database_service.log
├── chroma_db/           # ChromaDB vector storage
└── logs_database.db     # SQLite database
```

## 🚀 Usage Guide

### 1. Log Generation
You can generate logs directly from the app via the sidebar (no scripts required). The generator creates realistic log files with:
- **Format**: `YYYY-MM-DD-HH:MM:SS filename line_number SEVERITY message`
- **Severities**: ERROR (15%), WARN (10%), INFO (50%), DEBUG (25%)
- **50 entries per file** across 2 log files
- **Realistic messages** for different components

### 2. Chat Assistant Examples

Ask natural language questions like:
- *"Show me database connection errors"*
- *"What are the most common errors in the system?"*
- *"How can I fix authentication issues?"*
- *"Analyze the recent performance issues"*

Responses are strictly structured and based on ingested logs only, using the sections:
Summary, Key Issues, Probable Causes, Quick Fixes, Next Steps, Conclusion.

### 3. Dashboard Features

- **Health Score**: Calculated based on error/warning percentages
- **Severity Distribution**: Pie chart of log levels
- **Error Timeline**: Temporal analysis of issues
- **Top Error Files**: Files with most errors/warnings

### 4. Table Filtering

- **Severity Filter**: ERROR, WARN, INFO, DEBUG
- **Filename Filter**: Select specific files
- **Text Search**: Search within log messages
- **Export**: Download filtered results as CSV

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Your Google Gemini API key (required)

### Streamlit Configuration
Create `.streamlit/config.toml` for custom settings:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"

[server]
port = 8501
```

## 🧠 AI Integration Details

### RAG Pipeline
1. **Query Processing**: Validates log-related queries
2. **Vector Search**: Finds semantically similar log entries
3. **Context Building**: Assembles relevant logs for LLM
4. **Response Generation**: Gemini API generates contextual answers
5. **Guardrails**: Ensures responses are based only on log data

### Guardrail Keywords
The system recognizes log-related queries using keywords like:
- error, warning, debug, info, log, crash, fail, exception
- timeout, connection, database, service, api, authentication
- memory, performance, network, ssl, certificate, backup
- pod, container, deployment, server, application, system

## 📊 Database Schema

### SQLite Tables
```sql
CREATE TABLE logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    filename TEXT NOT NULL,
    line_number INTEGER NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    log_file_source TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### ChromaDB Collections
- **Collection**: `log_embeddings`
- **Documents**: Formatted log messages
- **Metadata**: Timestamp, filename, severity, line number
- **Embeddings**: Generated using `all-MiniLM-L6-v2`

## 🔁 Reset and Maintenance

Use the sidebar under "🧹 Maintenance":
- **🧽 Soft Reset (Truncate Data)**: Empties databases but keeps database files. Clears chat history. Best for reloading fresh logs quickly.
- **♻️ Reset Databases**: Completely deletes database files and optionally log files (if checkbox is selected). Requires re-ingestion after reset.
- **Also delete log files**: Checkbox option that only affects the Reset Databases button.

## 🔍 Troubleshooting

### Common Issues

1. **"No such table: logs" error after reset**
   - This is expected after Reset Databases
   - Simply ingest logs from the sidebar to recreate databases

2. **"Gemini API key missing" error**
   - Set `GEMINI_API_KEY` environment variable
   - Or add to `.streamlit/secrets.toml`

3. **"Cannot copy out of meta tensor" error**
   - This is a PyTorch/sentence-transformers compatibility issue
   - The app now handles this with fallback loading
   - Restart the app if you see this error

4. **Slow embedding generation on first run**
   - First run downloads the sentence transformer model (~90MB)
   - Subsequent runs will be much faster

### Performance Tips

- **Large log files**: Increase `n_results` parameter in vector search (default: 15)
- **Memory usage**: ChromaDB handles embeddings efficiently
- **Response time**: LLM integration is cached in session state automatically
- **Rate limits**: Gemini API free tier has request limits - wait a few minutes if you hit them

## 🧪 Testing

### Manual Testing (optional)
```bash
# Test LLM integration
python LLM_integration.py
```

### Sample Queries for Testing
1. "Show me all database errors"
2. "What's causing authentication failures?"
3. "Analyze system performance issues"
4. "Why are there so many timeouts?"
5. "Summarize the current system health"

## 🔮 Future Enhancements

- **Real-time log monitoring** with file watchers
- **Alert system** for critical errors
- **Log pattern detection** using ML
- **Multi-file log correlation** analysis
- **Custom dashboard widgets**
- **Integration with external monitoring tools**

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration settings
3. Ensure all dependencies are installed
4. Verify your Gemini API key is valid

---

**Built with ❤️ using Streamlit, Google Gemini AI, ChromaDB, and Sentence Transformers**
