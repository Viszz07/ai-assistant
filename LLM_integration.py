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
    
    def generate_safe_response(self, query: str, context: str, conversation_history: str = "") -> str:
        """
        Generate response using Gemini API with safety guardrails.
        """
        # Create a comprehensive prompt with guardrails and structured output
        system_prompt = """You are a log analysis assistant. Your role is to analyze the provided log context and answer ONLY using that context.

GUARDRAILS:
1. Only answer questions related to the provided log data and the user's query.
2. Base your response EXCLUSIVELY on the log entries provided in CONTEXT.
3. DO NOT paste raw log lines in the output. Summarize insights instead.
4. If the question cannot be answered from the logs, provide a contextual response based on what's actually available in the logs.
5. Avoid assumptions and generic advice that are not grounded in the context.

OUTPUT FORMAT (use these exact section headings):
### **Root Cause / Main Issue** (Short, precise explanation of what caused the issue.)
### **Analysis** (A detailed breakdown of patterns, trends, and observations.)
### **Solution** (Specific, actionable steps to resolve or mitigate the problem.)
### **Summary** (Short, precise explanation)

For root cause analysis queries, focus on identifying the primary cause and provide structured analysis.
For solution requests, provide specific, actionable steps based on the error patterns observed.
For top errors requests, list and categorize errors with clear prioritization.

CONTEXT (Log Entries):
{context}

CONVERSATION HISTORY (previous Q&A context):
{conversation_history}

USER QUERY: {query}

RESPONSE REQUIREMENTS:
- Provide a structured response using the exact section headings specified.
- After the Summary section, add a "Suggested Follow-ups" section with 2-3 relevant questions that can be answered based on the available log data.
- Make sure follow-up questions are specific and can be answered with the current dataset.
- Enhance responses with visual elements where appropriate:
  - Use markdown tables for comparing data or showing distributions
  - Include progress bars or visual indicators for metrics (e.g., [████████░░] 80%)
  - Add color-coded severity indicators (🔴 ERROR, 🟡 WARN, 🔵 INFO, ⚫ DEBUG)
  - Use bullet points with emojis for better readability (✅, ⚠️, 📊, etc.)
  - Include simple ASCII charts for trends when relevant

VISUAL ENHANCEMENT EXAMPLES:
- For error counts: Create a simple bar chart using markdown
- For severity distribution: Show as a visual breakdown
- For trends: Use arrows (↗️ ↘️) to indicate increases/decreases
- For status: Use status indicators (🟢 Healthy, 🟡 Warning, 🔴 Critical)

Now provide a concise, structured response following the required headings. Focus on what's actually in the context provided."""

        prompt = system_prompt.format(context=context, query=query, conversation_history=conversation_history)
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def process_query(self, query: str, conversation_history: str = "") -> Dict[str, Any]:
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

        # Step 3: Enhanced fallback strategy - if vector search doesn't find enough, try broader approaches
        context = self.build_context_from_logs(relevant_logs, query)
        used_fallback = False

        # If vector search found very few results, try broader approaches
        if len(relevant_logs) < 3 or (context.strip() == "No relevant logs found for this query."):
            # Fallback 1: Get recent ERROR and WARN logs
            fallback_logs = self.get_logs_by_severity('ERROR', 20) + self.get_logs_by_severity('WARN', 20)
            if fallback_logs:
                fallback_context = self.build_context_from_logs(fallback_logs[:30], query + " (recent errors and warnings)")
                if fallback_context and fallback_context != "No relevant logs found for this query.":
                    context = fallback_context
                    used_fallback = True

            # Fallback 2: If still no good results, get a broader set of recent logs
            if not used_fallback or len(relevant_logs) < 2:
                try:
                    cur = self.sqlite_conn.cursor()
                    cur.execute("""
                        SELECT timestamp, filename, line_number, severity, message
                        FROM logs
                        WHERE timestamp >= datetime('now', '-24 hours')
                        ORDER BY timestamp DESC
                        LIMIT 50
                    """)
                    recent_rows = cur.fetchall()
                    if recent_rows:
                        lines = ["Recent logs (last 24 hours) for context:"]
                        for i, row in enumerate(recent_rows, 1):
                            ts, fn, ln, sev, msg = row
                            short_msg = (msg[:140] + '…') if len(msg) > 140 else msg
                            lines.append(f"{i}. [{ts}] {sev} in {fn}:{ln} — {short_msg}")
                        context = "\n".join(lines)
                        used_fallback = True
                except Exception:
                    pass

        # Step 4: Generate response using Gemini API
        response = self.generate_safe_response(query, context, conversation_history)

        # Check if response indicates insufficient data and enhance it
        if "I don't have enough information" in response:
            # Get actual statistics to provide better context
            stats = self.get_log_statistics()

            # Enhanced handling for specific query types
            query_lower = query.lower()

            # Handle root cause requests
            if any(term in query_lower for term in ['root cause', 'cause of', 'why', 'reason']):
                error_logs = self.get_logs_by_severity('ERROR', 10)
                if error_logs:
                    response = self._generate_root_cause_analysis(error_logs, stats)
                else:
                    response = "No ERROR logs found. The system appears to be running without critical issues that require root cause analysis."

            # Handle top errors requests
            elif 'top' in query_lower and any(term in query_lower for term in ['error', 'warning', 'issue']):
                response = self._generate_top_errors_response(stats, query_lower)

            # Handle solution requests
            elif any(term in query_lower for term in ['solution', 'fix', 'resolve', 'how to']):
                error_logs = self.get_logs_by_severity('ERROR', 10)
                if error_logs:
                    response = self._generate_solutions_response(error_logs, stats)
                else:
                    response = "No errors found that require solutions. The system appears to be stable."

            # Handle general statistics
            elif stats['total_logs'] < 10:
                response = f"I have access to {stats['total_logs']} total logs. Here's what I can tell you about the system:\n\n"
                response += "**Log Distribution:**\n"
                for severity, count in stats['severity_distribution'].items():
                    percentage = (count / stats['total_logs'] * 100) if stats['total_logs'] > 0 else 0
                    response += f"• {severity}: {count} ({percentage:.1f}%)\n"

        return {
            'response': response,
            'relevant_logs': relevant_logs[:5],  # Return top 5 for display
            'context_used': len(relevant_logs) > 0,
            'total_logs_found': len(relevant_logs)
        }
    
    def _generate_root_cause_analysis(self, error_logs: List[Dict], stats: Dict[str, Any]) -> str:
        """Generate a structured root cause analysis response."""
        response = f"### **🔍 Root Cause Analysis**\n\n"
        response += f"**System Overview:** Found {stats['severity_distribution'].get('ERROR', 0)} errors across {stats['total_logs']} total logs.\n\n"

        # Group errors by type and frequency
        error_patterns = {}
        for log in error_logs[:10]:  # Analyze top 10 errors
            message = log['message'].lower()
            if 'connection' in message or 'timeout' in message:
                error_patterns['Connection/Network'] = error_patterns.get('Connection/Network', 0) + 1
            elif 'authentication' in message or 'auth' in message:
                error_patterns['Authentication'] = error_patterns.get('Authentication', 0) + 1
            elif 'database' in message or 'db' in message:
                error_patterns['Database'] = error_patterns.get('Database', 0) + 1
            elif 'memory' in message or 'out of memory' in message:
                error_patterns['Memory'] = error_patterns.get('Memory', 0) + 1
            elif 'permission' in message or 'access denied' in message:
                error_patterns['Permissions'] = error_patterns.get('Permissions', 0) + 1
            else:
                error_patterns['Other'] = error_patterns.get('Other', 0) + 1

        # Find the most common error pattern
        if error_patterns:
            primary_issue = max(error_patterns.items(), key=lambda x: x[1])
            response += f"**🔴 Primary Issue:** {primary_issue[0]} errors (occurring {primary_issue[1]} times)\n\n"

        response += f"### **📊 Analysis**\n"
        response += f"**Error Distribution:**\n"
        for pattern, count in error_patterns.items():
            percentage = (count / len(error_logs[:10])) * 100
            response += f"• {pattern}: {count} ({percentage:.1f}%)\n"

        response += f"\n**Most Frequent Errors:**\n"
        for i, log in enumerate(error_logs[:5], 1):
            response += f"{i}. [{log['timestamp']}] {log['filename']}:{log['line_number']} - {log['message']}\n"

        response += f"\n### **🔧 Solution**\n"
        if 'Connection/Network' in error_patterns:
            response += f"**Network Solutions:**\n"
            response += f"• Check network connectivity and firewall settings\n"
            response += f"• Verify service endpoints and ports are accessible\n"
            response += f"• Review connection timeout configurations\n"

        if 'Authentication' in error_patterns:
            response += f"\n**Authentication Solutions:**\n"
            response += f"• Verify API keys and credentials are valid\n"
            response += f"• Check authentication service availability\n"
            response += f"• Review authentication token expiration\n"

        if 'Database' in error_patterns:
            response += f"\n**Database Solutions:**\n"
            response += f"• Check database connectivity and credentials\n"
            response += f"• Verify database server is running\n"
            response += f"• Review query performance and indexing\n"

        response += f"\n### **📋 Summary**\n"
        response += f"**{stats['severity_distribution'].get('ERROR', 0)} errors detected.** "
        if error_patterns:
            response += f"Most critical: {primary_issue[0]} issues affecting system stability."
        else:
            response += f"System requires immediate attention to resolve error conditions."

        return response

    def _generate_top_errors_response(self, stats: Dict[str, Any], query_lower: str) -> str:
        """Generate response for top errors/warnings requests."""
        error_count = stats['severity_distribution'].get('ERROR', 0)
        warn_count = stats['severity_distribution'].get('WARN', 0)

        response = f"### **📊 Top Errors & Issues Analysis**\n\n"

        if error_count == 0 and warn_count == 0:
            response += f"✅ **System Health: Excellent** - No errors or warnings found!\n\n"
            response += f"**Current Status:** 🟢 System running smoothly with {stats['total_logs']} total logs recorded.\n"
        else:
            response += f"**System Overview:** {error_count} errors, {warn_count} warnings across {stats['total_logs']} logs.\n\n"

            # Top errors
            if error_count > 0:
                error_logs = self.get_logs_by_severity('ERROR', min(5, error_count))
                response += f"### **🔴 Top {min(5, error_count)} ERROR logs:**\n"
                for i, log in enumerate(error_logs, 1):
                    response += f"**{i}.** `{log['filename']}:{log['line_number']}` - {log['message']}\n"

            # Top warnings
            if warn_count > 0:
                warn_logs = self.get_logs_by_severity('WARN', min(5, warn_count))
                response += f"\n### **🟡 Top {min(5, warn_count)} WARNING logs:**\n"
                for i, log in enumerate(warn_logs, 1):
                    response += f"**{i}.** `{log['filename']}:{log['line_number']}` - {log['message']}\n"

        response += f"\n### **📋 Summary**\n"
        if error_count == 0:
            response += f"✅ **Excellent system health** - No critical errors detected."
        elif error_count < 3:
            response += f"🟡 **Good system health** - Only {error_count} minor error(s) found."
        else:
            response += f"🔴 **System needs attention** - {error_count} error(s) require investigation."

        return response

    def _generate_solutions_response(self, error_logs: List[Dict], stats: Dict[str, Any]) -> str:
        """Generate response with specific solutions for errors."""
        response = f"### **🔧 Error Solutions & Recommendations**\n\n"
        response += f"**System Status:** {stats['severity_distribution'].get('ERROR', 0)} errors detected across {stats['total_logs']} logs.\n\n"

        # Group errors and provide targeted solutions
        error_types = {
            'connection': {'count': 0, 'solutions': [
                '• Check network connectivity and firewall settings',
                '• Verify service endpoints and port accessibility',
                '• Review connection timeout configurations',
                '• Test with different network conditions'
            ]},
            'authentication': {'count': 0, 'solutions': [
                '• Verify API keys and credentials are current',
                '• Check authentication service availability',
                '• Review token expiration policies',
                '• Validate user permissions and roles'
            ]},
            'database': {'count': 0, 'solutions': [
                '• Verify database server connectivity',
                '• Check database credentials and permissions',
                '• Review query performance and optimize slow queries',
                '• Monitor database resource usage'
            ]},
            'memory': {'count': 0, 'solutions': [
                '• Monitor memory usage patterns',
                '• Review and optimize memory-intensive operations',
                '• Check for memory leaks in application code',
                '• Consider increasing memory allocation if needed'
            ]}
        }

        # Categorize errors
        for log in error_logs[:10]:
            message = log['message'].lower()
            if any(term in message for term in ['connection', 'timeout', 'network']):
                error_types['connection']['count'] += 1
            elif any(term in message for term in ['auth', 'authentication', 'permission', 'access denied']):
                error_types['authentication']['count'] += 1
            elif any(term in message for term in ['database', 'db', 'sql', 'query']):
                error_types['database']['count'] += 1
            elif any(term in message for term in ['memory', 'out of memory', 'heap']):
                error_types['memory']['count'] += 1

        # Provide solutions for detected error types
        solutions_provided = False
        for error_type, data in error_types.items():
            if data['count'] > 0:
                solutions_provided = True
                response += f"### **{error_type.title()} Issues ({data['count']} found)**\n"
                for solution in data['solutions']:
                    response += f"{solution}\n"
                response += "\n"

        if not solutions_provided:
            response += f"**General Troubleshooting Steps:**\n"
            response += f"• Review system logs for detailed error messages\n"
            response += f"• Check system resource utilization (CPU, Memory, Disk)\n"
            response += f"• Verify all dependent services are running\n"
            response += f"• Review recent configuration changes\n"

        response += f"\n### **📋 Summary**\n"
        response += f"**Priority Level:** "
        if stats['severity_distribution'].get('ERROR', 0) > 5:
            response += f"🔴 **High** - Multiple errors require immediate attention"
        elif stats['severity_distribution'].get('ERROR', 0) > 0:
            response += f"🟡 **Medium** - Some errors need investigation"
        else:
            response += f"🟢 **Low** - No critical errors detected"

        return response

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