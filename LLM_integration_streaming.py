import os
from dotenv import load_dotenv
import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Generator
import re
import json
import time

class LLMIntegration:
    """
    Handles LLM integration with Gemini 2.5 Flash API for log analysis.
    Implements RAG pipeline with guardrails to prevent hallucination.
    Now supports streaming responses for real-time chat experience.
    """
    
    def __init__(self, db_path="logs_database.db", chroma_path="./chroma_db"):
        # Load environment variables from .env if present (useful for standalone runs)
        load_dotenv()
        self.db_path = db_path
        self.chroma_path = chroma_path
        
        # Load sentence transformer with device specification to avoid meta tensor issues
        print("Loading sentence transformer model...")
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            print(f"Model loaded successfully on {device}!")
        except Exception as e:
            print(f"Error loading model with device specification: {e}")
            # Fallback: try loading without device specification
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Model loaded successfully (fallback method)!")
            except Exception as e2:
                raise Exception(f"Failed to load sentence transformer model: {e2}")
        
        # Initialize databases
        # Use check_same_thread=False because Streamlit may access this object across threads
        self.sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        
        try:
            self.chroma_collection = self.chroma_client.get_collection("log_embeddings")
        except:
            raise Exception("ChromaDB collection not found. Please run db_setup.py first.")
        
        # Initialize Gemini API
        self.setup_gemini_api()
        
        # Define guardrail keywords for log-related queries
        self.log_related_keywords = [
            'error', 'warning', 'debug', 'info', 'log', 'crash', 'fail', 'exception',
            'timeout', 'connection', 'database', 'service', 'api', 'authentication',
            'memory', 'performance', 'network', 'ssl', 'certificate', 'backup',
            'cache', 'validation', 'permission', 'transaction', 'query', 'response',
            'pod', 'container', 'deployment', 'server', 'application', 'system'
        ]
    
    def setup_gemini_api(self):
        """Initialize Gemini API with API key"""
        # Try to get API key from environment variable or streamlit secrets
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GEMINI_API_KEY")
            except:
                pass
        
        if not api_key:
            raise Exception(
                "Gemini API key not found. Please set GEMINI_API_KEY environment variable "
                "or add it to streamlit secrets."
            )
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        print("Gemini API initialized successfully!")
    
    def is_log_related_query(self, query: str) -> bool:
        """
        Check if the query is related to log analysis.
        Implements guardrails to prevent non-log related questions.
        """
        query_lower = query.lower()
        
        # Check for log-related keywords
        for keyword in self.log_related_keywords:
            if keyword in query_lower:
                return True
        
        # Check for question patterns that might be log-related
        log_patterns = [
            r'why.*(?:crash|fail|error|down)',
            r'what.*(?:wrong|error|problem|issue)',
            r'how.*(?:fix|resolve|solve)',
            r'when.*(?:error|fail|crash)',
            r'show.*(?:error|log|warning)',
            r'find.*(?:error|issue|problem)',
            r'analyze.*(?:log|error|performance)',
            r'troubleshoot',
            r'diagnose'
        ]
        
        for pattern in log_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def get_relevant_logs_from_vector_db(self, query: str, n_results: int = 10) -> List[Dict]:
        """
        Retrieve relevant logs using vector similarity search.
        """
        try:
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            relevant_logs = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    
                    relevant_logs.append({
                        'document': doc,
                        'metadata': metadata,
                        'similarity_score': 1 - distance  # Convert distance to similarity
                    })
            
            return relevant_logs
        
        except Exception as e:
            print(f"Error retrieving from vector database: {e}")
            return []
    
    def get_logs_by_severity(self, severity: str, limit: int = 20) -> List[Dict]:
        """
        Get logs filtered by severity level from SQLite.
        """
        cursor = self.sqlite_conn.cursor()
        
        query = """
            SELECT timestamp, filename, line_number, severity, message, log_file_source
            FROM logs 
            WHERE severity = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        
        cursor.execute(query, (severity, limit))
        results = cursor.fetchall()
        
        logs = []
        for row in results:
            logs.append({
                'timestamp': row[0],
                'filename': row[1],
                'line_number': row[2],
                'severity': row[3],
                'message': row[4],
                'log_file_source': row[5]
            })
        
        return logs
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """
        Get statistical information about logs from SQLite.
        """
        cursor = self.sqlite_conn.cursor()
        
        # Total count
        cursor.execute("SELECT COUNT(*) FROM logs")
        total_count = cursor.fetchone()[0]
        
        # Severity distribution
        cursor.execute("SELECT severity, COUNT(*) FROM logs GROUP BY severity")
        severity_stats = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Recent errors (last 10)
        cursor.execute("""
            SELECT timestamp, filename, message 
            FROM logs 
            WHERE severity IN ('ERROR', 'WARN') 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        recent_errors = cursor.fetchall()
        
        # Most common error files
        cursor.execute("""
            SELECT filename, COUNT(*) as error_count
            FROM logs 
            WHERE severity = 'ERROR'
            GROUP BY filename 
            ORDER BY error_count DESC 
            LIMIT 5
        """)
        error_files = cursor.fetchall()
        
        return {
            'total_logs': total_count,
            'severity_distribution': severity_stats,
            'recent_errors': recent_errors,
            'top_error_files': error_files
        }
    
    def build_context_from_logs(self, relevant_logs: List[Dict], query: str) -> str:
        """
        Build context string from relevant logs for LLM prompt.
        """
        if not relevant_logs:
            return "No relevant logs found for this query."
        
        context = "RELEVANT LOG ENTRIES:\\n\\n"
        
        for i, log in enumerate(relevant_logs[:30], 1):  # Increased from 10 to 30 for more comprehensive analysis
            metadata = log['metadata']
            context += f"{i}. [{metadata['timestamp']}] {metadata['filename']}:{metadata['line_number']} "
            context += f"[{metadata['severity']}] {log['document']}\\n"
        
        # Add statistical context if query seems to ask for overview
        if any(word in query.lower() for word in ['overview', 'summary', 'statistics', 'how many']):
            stats = self.get_log_statistics()
            context += f"\\n\\nLOG STATISTICS:\\n"
            context += f"Total logs: {stats['total_logs']}\\n"
            context += f"Severity distribution: {stats['severity_distribution']}\\n"
        
        return context
    
    def generate_streaming_response(self, query: str, context: str, conversation_history: str = "") -> Generator[str, None, None]:
        """
        Generate streaming response for real-time chat experience.
        Yields chunks of text as they are generated by the model.
        """
        # Create a concise prompt that produces shorter, more focused responses
        system_prompt = """You are a Network Assurance Expert AI. Provide **concise, actionable insights** about network systems and logs.

**Response Style Guidelines:**
- **Keep it brief**: 200-400 words max, focus on key insights only
- **Complete responses**: Always finish your response completely, never cut off mid-sentence
- **Visual & Scannable**: Use emojis, bold text, charts and bar graphs whenever necessary, bullet points, and simple tables
- **Action-oriented**: Prioritize actionable information over lengthy explanations
- **Context-aware**: Reference conversation history when relevant

**IMPORTANT - Context Awareness:**
- If the user asks about "previous response", "last answer", "that", "it", or similar references, USE THE CONVERSATION HISTORY
- When user asks to "explain in tabular manner" or "simplify", reformat the PREVIOUS ASSISTANT response
- When user asks follow-up questions like "can you explain that better", refer to the ASSISTANT's last response
- DO NOT say "I only answer based on logs" when user is asking about YOUR previous response
- You can discuss and reformat your own previous responses without needing log context

**Context Available:**
{context}

**Previous Conversation:**
{conversation_history}

**Query:** {query}

**Response Structure (Choose based on query type):**

**For LOG ANALYSIS & STACK TRACES:**
🔍 **Key Issue:** [SPECIFIC error type, file, and line number]
📊 **Impact:** [Visual breakdown - ERROR: 20% | WARN: 5%]
🎯 **Root Cause:** [Explain EXACTLY what's failing and why - be detailed, not vague]

✅ **Detailed Fix Steps:** [SPECIFIC actions, not generic advice]
   1. [Exact change needed - include file names, config keys, line numbers]
   2. [Specific command or code modification]
   3. [How to verify it worked]

💻 **Diagnostic Commands:** [Provide 3-5 copy-paste ready commands]
   Network issues: `ping <actual-host>`, `traceroute <host>`, `netstat -an | grep <port>`, `tcpdump -i any port <port>`
   Timeouts: `curl -v -w "Time: %%{{time_total}}s\n" <url>`, `telnet <host> <port>`
   Memory: `free -h`, `top -o %%MEM`, `ps aux --sort=-%%mem | head -10`
   Connections: `ss -tuln | grep <port>`, `lsof -i :<port>`, `netstat -tulpn`

🔧 **Config Changes:** [Show EXACT before/after if applicable]
   Example: "In vnfm_client.properties: change `timeout=30000` to `timeout=60000`"

🧪 **Test Commands:** [How to verify the fix]
   Example: `curl -X GET http://vnfm-endpoint:8080/health -v`

**For EXPLANATIONS:**
📚 **[Topic]:** [Brief definition - 1 sentence]
🎯 **Key Points:** [3-5 bullet points max]
💡 **Relevance:** [How it connects to logs - 1 sentence]

**For COMPARISONS:**
⚖️ **Comparison:** [Simple table or bullets]
📈 **Key Difference:** [1 sentence highlight]

**MANDATORY: Always end with EXACTLY 2 specific follow-up questions.**
**MANDATORY: Always try to create bar graphs, charts or visually appealing response possible.**

**CRITICAL REQUIREMENTS:**
- ALWAYS complete your entire response and include follow-up questions
- Generate EXACTLY 2 follow-up questions, numbered 1, 2
- Each follow-up question must be specific and answerable from the available data
- Never cut off mid-response or mid-question

**FOR STACK TRACES/ERRORS - USER WANTS DETAILED HELP:**
- Provide DETAILED, SPECIFIC troubleshooting (not brief/vague explanations)
- Include 3-5 diagnostic commands users can copy-paste immediately
- Explain EXACTLY what to change (file names, config keys, values)
- Show actual command examples with real values from the logs
- For each command, briefly explain what it checks
- If suggesting code/config changes, show EXACT before/after
- Be prescriptive and actionable - user wants to solve the problem NOW"""

        prompt = system_prompt.format(context=context, query=query, conversation_history=conversation_history)

        try:
            # Use Gemini's streaming API
            response_stream = self.model.generate_content(prompt, stream=True)
            
            accumulated_text = ""
            for chunk in response_stream:
                if chunk.text:
                    accumulated_text += chunk.text
                    yield chunk.text
                    
            # Ensure the response is complete
            if accumulated_text and not accumulated_text.strip().endswith(('?', '.', '!')):
                completion = "."
                yield completion
                
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            yield error_msg
    
    def generate_safe_response(self, query: str, context: str, conversation_history: str = "", stream: bool = False):
        """
        Generate concise, visually appealing responses for network assurance queries.
        Returns a generator if stream=True, otherwise returns a complete string.
        """
        if stream:
            return self.generate_streaming_response(query, context, conversation_history)
        
        # Original non-streaming implementation
        system_prompt = """You are a Network Assurance Expert AI. Provide **concise, actionable insights** about network systems and logs.

**Response Style Guidelines:**
- **Keep it brief**: 200-400 words max, focus on key insights only
- **Complete responses**: Always finish your response completely, never cut off mid-sentence
- **Visual & Scannable**: Use emojis, bold text, charts and bar graphs whenever necessary, bullet points, and simple tables
- **Action-oriented**: Prioritize actionable information over lengthy explanations
- **Context-aware**: Reference conversation history when relevant

**IMPORTANT - Context Awareness:**
- If the user asks about "previous response", "last answer", "that", "it", or similar references, USE THE CONVERSATION HISTORY
- When user asks to "explain in tabular manner" or "simplify", reformat the PREVIOUS ASSISTANT response
- When user asks follow-up questions like "can you explain that better", refer to the ASSISTANT's last response
- DO NOT say "I only answer based on logs" when user is asking about YOUR previous response
- You can discuss and reformat your own previous responses without needing log context

**Context Available:**
{context}

**Previous Conversation:**
{conversation_history}

**Query:** {query}

**Response Structure (Choose based on query type):**

**For LOG ANALYSIS & STACK TRACES:**
🔍 **Key Issue:** [SPECIFIC error type, file, and line number]
📊 **Impact:** [Visual breakdown - ERROR: 20% | WARN: 5%]
🎯 **Root Cause:** [Explain EXACTLY what's failing and why - be detailed, not vague]

✅ **Detailed Fix Steps:** [SPECIFIC actions, not generic advice]
   1. [Exact change needed - include file names, config keys, line numbers]
   2. [Specific command or code modification]
   3. [How to verify it worked]

💻 **Diagnostic Commands:** [Provide 3-5 copy-paste ready commands]
   Network issues: `ping <actual-host>`, `traceroute <host>`, `netstat -an | grep <port>`, `tcpdump -i any port <port>`
   Timeouts: `curl -v -w "Time: %%{{time_total}}s\n" <url>`, `telnet <host> <port>`
   Memory: `free -h`, `top -o %%MEM`, `ps aux --sort=-%%mem | head -10`
   Connections: `ss -tuln | grep <port>`, `lsof -i :<port>`, `netstat -tulpn`

🔧 **Config Changes:** [Show EXACT before/after if applicable]
   Example: "In vnfm_client.properties: change `timeout=30000` to `timeout=60000`"

🧪 **Test Commands:** [How to verify the fix]
   Example: `curl -X GET http://vnfm-endpoint:8080/health -v`

**For EXPLANATIONS:**
📚 **[Topic]:** [Brief definition - 1 sentence]
🎯 **Key Points:** [3-5 bullet points max]
💡 **Relevance:** [How it connects to logs - 1 sentence]

**For COMPARISONS:**
⚖️ **Comparison:** [Simple table or bullets]
📈 **Key Difference:** [1 sentence highlight]

**MANDATORY: Always end with EXACTLY 2 specific follow-up questions.**
**MANDATORY: Always try to create bar graphs, charts or visually appealing response possible.**

**CRITICAL REQUIREMENTS:**
- ALWAYS complete your entire response and include follow-up questions
- Generate EXACTLY 2 follow-up questions, numbered 1, 2
- Each follow-up question must be specific and answerable from the available data
- Never cut off mid-response or mid-question

**FOR STACK TRACES/ERRORS - USER WANTS DETAILED HELP:**
- Provide DETAILED, SPECIFIC troubleshooting (not brief/vague explanations)
- Include 3-5 diagnostic commands users can copy-paste immediately
- Explain EXACTLY what to change (file names, config keys, values)
- Show actual command examples with real values from the logs
- For each command, briefly explain what it checks
- If suggesting code/config changes, show EXACT before/after
- Be prescriptive and actionable - user wants to solve the problem NOW"""

        prompt = system_prompt.format(context=context, query=query, conversation_history=conversation_history)

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text

            # Ensure response is complete (remove excessive truncation)
            if len(response_text) > 2000:
                # Find a good breaking point at the end of a sentence
                last_period = response_text.rfind('.')
                if last_period > 1000:
                    response_text = response_text[:last_period + 1]
                else:
                    response_text = response_text[:2000] + "..."

            # Ensure follow-up questions are complete
            if "❓" in response_text:
                # Find the last question mark to ensure completeness
                last_question = response_text.rfind('?')
                if last_question > 0 and last_question < len(response_text) - 50:
                    # Check if response ends abruptly after a question
                    after_last_question = response_text[last_question + 1:].strip()
                    if len(after_last_question) < 10 and not after_last_question.endswith('.'):
                        # Response might be cut off, don't truncate
                        pass
                elif last_question == -1:
                    # No question marks found, ensure we have a complete response
                    if not response_text.endswith('.'):
                        response_text += "."

            return response_text

        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def process_query_with_history(self, query: str, conversation_history: str = "", stream: bool = False) -> Dict[str, Any]:
        """
        Process user queries with conversation history context.
        Supports both streaming and non-streaming responses.
        """
        # Step 1: Check if query is log-related (guardrail)
        if not self.is_log_related_query(query):
            error_response = "I can only answer questions related to network assurance, logs, or telecom systems. Please ask about network components, service flows, or log analysis."
            if stream:
                def error_generator():
                    yield error_response
                return {
                    'response': error_generator(),
                    'relevant_logs': [],
                    'context_used': False,
                    'is_streaming': True
                }
            else:
                return {
                    'response': error_response,
                    'relevant_logs': [],
                    'context_used': False,
                    'is_streaming': False
                }

        # Step 2: Retrieve relevant logs using vector search
        relevant_logs = self.get_relevant_logs_from_vector_db(query, n_results=30)

        # Step 3: Build context with conversation history
        context = self.build_context_from_logs(relevant_logs, query)

        # Step 4: Generate response with conversation context
        response = self.generate_safe_response(query, context, conversation_history, stream=stream)

        return {
            'response': response,
            'relevant_logs': relevant_logs[:5],
            'context_used': len(relevant_logs) > 0,
            'total_logs_found': len(relevant_logs),
            'is_streaming': stream
        }
    
    def close_connections(self):
        """Close database connections"""
        if self.sqlite_conn:
            self.sqlite_conn.close()

# Example usage for testing streaming
def test_streaming():
    """Test the streaming functionality"""
    try:
        llm = LLMIntegration()
        
        query = "Show me recent errors in the system"
        print(f"Query: {query}")
        print("Streaming response:")
        
        result = llm.process_query_with_history(query, stream=True)
        
        if result['is_streaming']:
            for chunk in result['response']:
                print(chunk, end='', flush=True)
                time.sleep(0.1)  # Simulate real-time display
        
        print(f"\\n\\nLogs found: {result['total_logs_found']}")
        
        llm.close_connections()
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_streaming()
