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
        
        for i, log in enumerate(relevant_logs[:30], 1):  # Increased from 10 to 30 for more comprehensive analysis
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
        Generate concise, visually appealing responses for network assurance queries.
        """
        # Create a concise prompt that produces shorter, more focused responses
        system_prompt = """You are a Network Assurance Expert AI. Provide **concise, actionable insights** about network systems and logs.

**Response Style Guidelines:**
- **Keep it brief**: 200-400 words max, focus on key insights only
- **Complete responses**: Always finish your response completely, never cut off mid-sentence
- **Visual & Scannable**: Use emojis, bold text, charts and bar graphs whenever necessary,bullet points, and simple tables
- **Action-oriented**: Prioritize actionable information over lengthy explanations
- **Context-aware**: Reference conversation history when relevant

**Context Available:**
{context}

**Previous Conversation:**
{conversation_history}

**Query:** {query}

**Response Structure (Choose based on query type):**

**For LOG ANALYSIS:**
🔍 **Key Issue:** [1-sentence summary]
📊 **Impact:** [Visual breakdown - use simple bars like ERROR: 20% | WARN: 5%]
✅ **Quick Fix:** [2-3 bullet points max]

**For EXPLANATIONS:**
📚 **[Topic]:** [Brief definition - 1 sentence]
🎯 **Key Points:** [3-5 bullet points max]
💡 **Relevance:** [How it connects to logs - 1 sentence]

**For COMPARISONS:**
⚖️ **Comparison:** [Simple table or bullets]
📈 **Key Difference:** [1 sentence highlight]

**MANDATORY: Always end with EXACTLY 3 specific follow-up questions.**
**MANDATORY: Always try to create bar graphs, charts o visually applealing response possible.**

**Examples of Good Responses:**
- "🔍 **Network Congestion:** High packet loss detected in core network components. 📊 **Impact:** ERROR: 15% | WARN: 8% | INFO: 77%. ✅ **Quick Fix:** • Check bandwidth allocation • Review QoS policies • Scale VNF instances.**❓ **Follow-ups:** • What specific components show the highest packet loss? • How has network performance trended over the last week? • Can you show me the VNF deployment status?**"
- "📚 **MME (Mobility Management Entity):** Core 4G component handling user authentication and mobility. 🎯 **Key Points:** • Manages UE connections • Handles handovers • Tracks user location. 💡 **Relevance:** Your logs show MME authentication failures affecting user connectivity.**❓ **Follow-ups:** • Show me MME error patterns • Explain AMF vs MME differences • What VNFs are currently deployed for MME?**"

**CRITICAL REQUIREMENTS:**
- ALWAYS complete your entire response and include follow-up questions
- Generate EXACTLY 3 follow-up questions, numbered 1, 2, and 3.
- Each follow-up question must be specific and answerable from the available data
- Never cut off mid-response or mid-question"""

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

    def generate_visual_log_summary(self, logs_data):
        """Generate visual summary of log data with charts and graphs"""
        if not logs_data:
            return ""

        # Analyze severity distribution
        severity_counts = {}
        component_counts = {}
        module_counts = {}

        for log in logs_data:
            # Count severities
            severity = log.get('severity', 'INFO')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

            # Count components
            component = log.get('component', 'Unknown')
            component_counts[component] = component_counts.get(component, 0) + 1

            # Count modules
            module = log.get('module', 'Unknown')
            module_counts[module] = module_counts.get(module, 0) + 1

    def generate_visual_log_summary(self, logs_data):
        """Generate concise visual summary of log data"""
        if not logs_data:
            return ""

        # Analyze severity distribution
        severity_counts = {}
        component_counts = {}

        for log in logs_data:
            # Count severities
            severity = log.get('severity', 'INFO')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

            # Count components
            component = log.get('component', 'Unknown')
            component_counts[component] = component_counts.get(component, 0) + 1

        # Create concise visual elements
        visual_summary = ""

        # Simple severity overview
        if severity_counts:
            total = sum(severity_counts.values())
            error_pct = (severity_counts.get('ERROR', 0) / total) * 100
            warn_pct = (severity_counts.get('WARN', 0) / total) * 100

            visual_summary += f"\n📊 **Status:** 🔴 {error_pct:.0f}% errors | 🟡 {warn_pct:.0f}% warnings"

        # Most active component (if significant)
        if component_counts:
            top_comp = max(component_counts.items(), key=lambda x: x[1])
            if top_comp[1] > 1:  # Only show if meaningful activity
                visual_summary += f"\n🏆 **Most Active:** {top_comp[0]} ({top_comp[1]} events)"

        return visual_summary

    def create_ascii_bar_chart(self, data_dict, title="", max_bar_length=12):
        """Create an ASCII bar chart from a dictionary of values"""
        if not data_dict:
            return ""

        # Find max value for scaling
        max_value = max(data_dict.values()) if data_dict else 1

        chart = f"\n{title}\n" if title else "\n"
        for label, value in sorted(data_dict.items(), key=lambda x: x[1], reverse=True):
            # Scale the bar length
            bar_length = int((value / max_value) * max_bar_length)
            bar = "█" * bar_length + "░" * (max_bar_length - bar_length)
            chart += f"{label:8s} {bar} ({value})\n"

        return chart

    def create_trend_chart(self, time_series_data, title=""):
        """Create a trend chart with time series data"""
        if not time_series_data:
            return ""

        chart = f"\n{title}\n" if title else "\n"

        for time_point, (value, trend) in time_series_data.items():
            # Scale bar length based on value
            max_value = max(v[0] for v in time_series_data.values()) if time_series_data else 1
            bar_length = int((value / max_value) * 10) if max_value > 0 else 0
            bar = "█" * bar_length + "░" * (10 - bar_length)

            trend_emoji = "↗️" if trend == "up" else "↘️" if trend == "down" else "➡️"
            trend_text = "(increasing)" if trend == "up" else "(decreasing)" if trend == "down" else "(stable)"

            chart += f"{time_point:5s} {bar} {trend_emoji} {trend_text}\n"

        return chart

    def create_progress_bar(self, percentage, label="", bar_length=12):
        """Create a progress bar for percentages"""
        filled_length = int((percentage / 100) * bar_length)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        return f"{label:12s} [{bar}] {percentage:.1f}%"
    
    def process_query_with_history(self, query: str, conversation_history: str = "") -> Dict[str, Any]:
        """
        Process user queries with conversation history context.
        """
        # Step 1: Check if query is log-related (guardrail)
        if not self.is_log_related_query(query):
            return {
                'response': "I can only answer questions related to network assurance, logs, or telecom systems. Please ask about network components, service flows, or log analysis.",
                'relevant_logs': [],
                'context_used': False
            }

        # Step 2: Retrieve relevant logs using vector search
        relevant_logs = self.get_relevant_logs_from_vector_db(query, n_results=30)

        # Step 3: Build context with conversation history
        context = self.build_context_from_logs(relevant_logs, query)

        # Step 4: Generate response with conversation context
        response = self.generate_safe_response(query, context, conversation_history)

        # Enhance response with visual elements if we have relevant logs
        if relevant_logs and len(relevant_logs) > 0:
            visual_summary = self.generate_visual_log_summary(relevant_logs)

            # Insert visual summary into response if it doesn't already contain visual elements
            if visual_summary and "📊" not in response and len(response) < 1000:
                # Find a good place to insert the visual summary (after the first section)
                lines = response.split('\n')
                insert_index = 0

                # Find the first major section break
                for i, line in enumerate(lines):
                    if line.startswith('###') and i > 0:
                        insert_index = i + 1
                        break

                if insert_index > 0:
                    lines.insert(insert_index, visual_summary)
                    response = '\n'.join(lines)

        return {
            'response': response,
            'relevant_logs': relevant_logs[:5],
            'context_used': len(relevant_logs) > 0,
            'total_logs_found': len(relevant_logs)
        }
    
    def _generate_root_cause_analysis(self, error_logs: List[Dict], stats: Dict[str, Any]) -> str:
        """Generate a structured root cause analysis response."""
        response = f"### **🔍 Root Cause Analysis**\n\n"
        response += f"**System Overview:** Found {stats['severity_distribution'].get('ERROR', 0)} errors across {stats['total_logs']} total logs.\n\n"

        # Group errors by type and frequency
        error_patterns = {}
        for log in error_logs[:20]:  # Analyze top 20 errors instead of 10
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
            percentage = (count / len(error_logs[:20])) * 100  # Updated to use 20
            response += f"• {pattern}: {count} ({percentage:.1f}%)\n"

        response += f"\n**Most Frequent Errors:**\n"
        for i, log in enumerate(error_logs[:8], 1):  # Show top 8 instead of 5
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
                error_logs = self.get_logs_by_severity('ERROR', min(10, error_count))  # Show more errors
                response += f"### **🔴 Top {min(10, error_count)} ERROR logs:**\n"
                for i, log in enumerate(error_logs, 1):
                    response += f"**{i}.** `{log['filename']}:{log['line_number']}` - {log['message']}\n"

            # Top warnings
            if warn_count > 0:
                warn_logs = self.get_logs_by_severity('WARN', min(8, warn_count))  # Show more warnings
                response += f"\n### **🟡 Top {min(8, warn_count)} WARNING logs:**\n"
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
        for log in error_logs[:20]:  # Analyze more errors for better categorization
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