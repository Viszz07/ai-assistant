import os
from dotenv import load_dotenv
import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import re
import json

class LLMIntegration:
    """
    Handles LLM integration with Gemini 2.5 Flash API for log analysis.
    Implements RAG pipeline with guardrails to prevent hallucination.
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
        
        context = "RELEVANT LOG ENTRIES:\n\n"
        
        for i, log in enumerate(relevant_logs[:10], 1):  # Limit to top 10 for context size
            metadata = log['metadata']
            context += f"{i}. [{metadata['timestamp']}] {metadata['filename']}:{metadata['line_number']} "
            context += f"[{metadata['severity']}] {log['document']}\n"
        
        # Add statistical context if query seems to ask for overview
        if any(word in query.lower() for word in ['overview', 'summary', 'statistics', 'how many']):
            stats = self.get_log_statistics()
            context += f"\n\nLOG STATISTICS:\n"
            context += f"Total logs: {stats['total_logs']}\n"
            context += f"Severity distribution: {stats['severity_distribution']}\n"
        
        return context
    
    def generate_safe_response(self, query: str, context: str) -> str:
        """
        Generate response using Gemini API with safety guardrails.
        """
        # Create a comprehensive prompt with guardrails and structured output
        system_prompt = """You are a log analysis assistant. Your role is to analyze the provided log context and answer ONLY using that context.

GUARDRAILS:
1. Only answer questions related to the provided log data and the user's query.
2. Base your response EXCLUSIVELY on the log entries provided in CONTEXT.
3. DO NOT paste raw log lines in the output. Summarize insights instead.
4. If the question cannot be answered from the logs, reply: "I can only answer based on the logs provided, and I don't have enough information to answer this question."
5. Avoid assumptions and generic advice that are not grounded in the context.

OUTPUT FORMAT (use these exact section headings):
### **Root Cause / Main Issue** (Short, precise explanation of what caused the issue.)
### **Analysis** (A detailed breakdown of patterns, trends, and observations.)
### **Solution** (Specific, actionable steps to resolve or mitigate the problem.)
### **Summary** (Short, precise explanation)

CONTEXT (Log Entries):
{context}

USER QUERY: {query}

Now provide a concise, structured response following the required headings. Do not include raw logs."""

        prompt = system_prompt.format(context=context, query=query)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Main method to process user queries with full RAG pipeline.
        """
        # Step 1: Check if query is log-related (guardrail)
        if not self.is_log_related_query(query):
            return {
                'response': "I can only answer questions based on the log data provided. Your query doesn't seem to relate to log analysis or system troubleshooting.",
                'relevant_logs': [],
                'context_used': False
            }
        
        # Step 2: Retrieve relevant logs using vector search
        relevant_logs = self.get_relevant_logs_from_vector_db(query, n_results=15)
        
        # Step 3: Build context from relevant logs
        context = self.build_context_from_logs(relevant_logs, query)
        
        # Step 4: Generate response using Gemini API
        response = self.generate_safe_response(query, context)
        
        return {
            'response': response,
            'relevant_logs': relevant_logs[:5],  # Return top 5 for display
            'context_used': len(relevant_logs) > 0,
            'total_logs_found': len(relevant_logs)
        }
    
    def get_error_summary(self) -> str:
        """
        Generate a summary of current system health based on logs.
        """
        stats = self.get_log_statistics()
        error_logs = self.get_logs_by_severity('ERROR', 10)
        
        context = f"""
LOG SUMMARY:
Total logs: {stats['total_logs']}
Severity distribution: {stats['severity_distribution']}

RECENT ERROR LOGS:
"""
        for log in error_logs:
            context += f"[{log['timestamp']}] {log['filename']} - {log['message']}\n"
        
        query = "Provide a summary of the current system health and main issues based on these logs"
        response = self.generate_safe_response(query, context)
        
        return response
    
    def close_connections(self):
        """Close database connections"""
        if self.sqlite_conn:
            self.sqlite_conn.close()

# Example usage and testing
def test_llm_integration():
    """Test the LLM integration with sample queries"""
    try:
        llm = LLMIntegration()
        
        test_queries = [
            "Show me database connection errors",
            "What are the most common errors in the system?",
            "What's the weather like today?",  # Should be rejected
            "How can I fix authentication issues?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            result = llm.process_query(query)
            print(f"Response: {result['response'][:200]}...")
            print(f"Logs found: {result['total_logs_found']}")
            print("-" * 50)
        
        llm.close_connections()
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_llm_integration()