import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from LLM_integration import LLMIntegration
import time
from db_setup import DatabaseSetup
from log_generator import LogGenerator
import shutil
import re
import json

# Import network assurance knowledge base
from network_assurance_kb import NETWORK_ASSURANCE_KNOWLEDGE, get_component_context, analyze_query_intent

# Load environment variables from a .env file if present
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Service Cure Insights",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Ensure main page content starts below the navbar */
    .block-container {
        padding-top: 4.5rem; /* increase if navbar still overlaps */
    }
    .tab-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .error-card {
        background-color: #fff5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e53e3e;
        margin: 0.5rem 0;
        color: #2d3748;
    }
    .warning-card {
        background-color: #fffbf0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dd6b20;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

class LogAnalysisApp:
    """Main Streamlit application for log analysis"""
    
    def __init__(self):
        self.db_path = "logs_database.db"
        self.setup_session_state()
        
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'llm_integration' not in st.session_state:
            st.session_state.llm_integration = None
        # Default to manual ingestion mode: require user to ingest logs explicitly
        if 'db_initialized' not in st.session_state:
            st.session_state.db_initialized = False
        if 'uploaded_logs' not in st.session_state:
            st.session_state.uploaded_logs = []
        if 'ingest_message' not in st.session_state:
            st.session_state.ingest_message = ""
    
    def check_database_exists(self):
        """Check if the database and required files exist"""
        return (os.path.exists(self.db_path) and 
                os.path.exists("./chroma_db") and
                os.path.exists("logs"))
    
    def get_database_connection(self):
        """Get SQLite database connection"""
        return sqlite3.connect(self.db_path)
    
    def load_llm_integration(self):
        """Load LLM integration with error handling"""
        try:
            if st.session_state.llm_integration is None:
                with st.spinner("Initializing AI assistant..."):
                    st.session_state.llm_integration = LLMIntegration()
            return st.session_state.llm_integration
        except Exception as e:
            st.error(f"Failed to initialize AI assistant: {str(e)}")
            st.info("Please ensure your Gemini API key is set in environment variables or Streamlit secrets.")
            return None
    
    def render_sidebar(self):
        """Render the sidebar with log ingestion controls"""
        with st.sidebar:
            st.title("💊 Service Cure Insights")

            if st.session_state.db_initialized:
                st.success("✅ Database Connected")
            else:
                st.info("💡 Load logs to get started")

            st.subheader("📥 Data Ingestion")

            # Generate sample logs and ingest
            if st.button("🧪 Generate Sample Logs and Ingest"):
                with st.spinner("Generating sample logs and ingesting into databases..."):
                    try:
                        # Generate logs
                        gen = LogGenerator()
                        gen.generate_all_logs()

                        # Setup and load databases
                        dbs = DatabaseSetup()
                        dbs.setup_sqlite()
                        dbs.setup_chromadb()
                        dbs.load_all_logs()
                        dbs.close_connections()

                        st.session_state.db_initialized = True
                        st.session_state.ingest_message = "✅ Sample logs generated and ingested successfully."
                    except Exception as e:
                        st.session_state.ingest_message = f"❌ Ingestion failed: {e}"

            # Upload and ingest custom logs
            uploaded = st.file_uploader("Upload .log files", type=["log", "txt"], accept_multiple_files=True)
            if uploaded:
                st.session_state.uploaded_logs = uploaded
            if st.session_state.uploaded_logs:
                if st.button("📦 Ingest Uploaded Logs"):
                    with st.spinner("Saving uploaded files and ingesting into databases..."):
                        try:
                            os.makedirs("logs", exist_ok=True)
                            # Save uploaded files
                            saved_files = []
                            for f in st.session_state.uploaded_logs:
                                save_path = os.path.join("logs", f.name)
                                with open(save_path, "wb") as out:
                                    out.write(f.getbuffer())
                                saved_files.append(save_path)

                            # Setup and (re)load databases
                            dbs = DatabaseSetup()
                            dbs.setup_sqlite()
                            dbs.setup_chromadb()
                            # If you only want to load just uploaded files, we can process then insert
                            all_entries = []
                            for lf in saved_files:
                                all_entries.extend(dbs.process_log_file(lf))
                            if all_entries:
                                dbs.insert_logs_to_sqlite(all_entries)
                                dbs.insert_logs_to_chromadb(all_entries)
                                dbs.print_database_summary()
                            dbs.close_connections()

                            st.session_state.db_initialized = True
                            st.session_state.ingest_message = f"✅ Ingested {len(saved_files)} uploaded files successfully."
                        except Exception as e:
                            st.session_state.ingest_message = f"❌ Ingestion failed: {e}"

            if st.session_state.ingest_message:
                st.info(st.session_state.ingest_message)

            # Ingest only selected existing logs (service-specific ingest)
            if os.path.exists("logs"):
                try:
                    existing_logs = [f for f in os.listdir("logs") if f.lower().endswith((".log", ".txt"))]
                except Exception:
                    existing_logs = []
                if existing_logs:
                    selected_logs = st.multiselect("Select logs to ingest", options=existing_logs, default=[])
                    if st.button("➡️ Ingest Selected Existing Logs") and selected_logs:
                        with st.spinner("Ingesting selected logs..."):
                            try:
                                dbs = DatabaseSetup()
                                dbs.setup_sqlite()
                                dbs.setup_chromadb()
                                all_entries = []
                                for name in selected_logs:
                                    path = os.path.join("logs", name)
                                    all_entries.extend(dbs.process_log_file(path))
                                if all_entries:
                                    dbs.insert_logs_to_sqlite(all_entries)
                                    dbs.insert_logs_to_chromadb(all_entries)
                                    dbs.print_database_summary()
                                dbs.close_connections()
                                st.session_state.db_initialized = True
                                st.session_state.ingest_message = f"✅ Ingested {len(selected_logs)} selected log file(s)."
                            except Exception as e:
                                st.session_state.ingest_message = f"❌ Ingestion failed: {e}"

            # Maintenance / Reset controls
            st.subheader("🧹 Maintenance")
            delete_logs = st.checkbox("Also delete log files", value=False)
            if st.button("🧽 Soft Reset (Truncate Data)"):
                with st.spinner("Truncating data without deleting files..."):
                    try:
                        # Close cached llm instance to avoid locks
                        llm_instance = st.session_state.get('llm_integration')
                        if llm_instance is not None:
                            try:
                                llm_instance.close_connections()
                            except Exception:
                                pass
                            finally:
                                st.session_state.llm_integration = None

                        # Truncate SQLite logs table
                        if os.path.exists(self.db_path):
                            try:
                                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                                cur = conn.cursor()
                                cur.execute("DELETE FROM logs")
                                conn.commit()
                                conn.close()
                            except Exception as e:
                                st.warning(f"SQLite truncate warning: {e}")

                        # Reset ChromaDB collection
                        try:
                            import chromadb
                            client = chromadb.PersistentClient(path="./chroma_db")
                            try:
                                client.delete_collection("log_embeddings")
                            except Exception:
                                pass
                            client.create_collection(name="log_embeddings", metadata={"description": "Log message embeddings for semantic search"})
                        except Exception as e:
                            st.warning(f"ChromaDB reset warning: {e}")

                        st.session_state.ingest_message = "✅ Soft reset complete. Databases are empty; please ingest logs."
                        st.session_state.db_initialized = os.path.exists(self.db_path)
                        st.session_state.chat_history = []
                        st.success("Soft reset complete.")
                    except Exception as e:
                        st.error(f"Soft reset failed: {e}")
            if st.button("♻️ Reset Databases"):
                with st.spinner("Resetting databases..."):
                    try:
                        # Close any existing LLM integration (to release DB file handle on Windows)
                        llm_instance = st.session_state.get('llm_integration')
                        if llm_instance is not None:
                            try:
                                llm_instance.close_connections()
                            except Exception:
                                pass
                            finally:
                                st.session_state.llm_integration = None

                        # Delete SQLite DB
                        if os.path.exists(self.db_path):
                            try:
                                os.remove(self.db_path)
                            except Exception as e:
                                st.error(f"Failed to delete SQLite database: {e}")
                                raise

                        # Reset ChromaDB collection (don't delete directory - Windows file lock issue)
                        # Instead, delete and recreate the collection
                        try:
                            import chromadb
                            client = chromadb.PersistentClient(path="./chroma_db")
                            try:
                                client.delete_collection("log_embeddings")
                            except Exception:
                                pass  # Collection might not exist
                            # Recreate empty collection
                            client.create_collection(
                                name="log_embeddings",
                                metadata={"description": "Log message embeddings for semantic search"}
                            )
                        except Exception as e:
                            st.error(f"Failed to reset ChromaDB collection: {e}")
                            raise
                        # Optionally delete logs
                        if delete_logs and os.path.exists("logs"):
                            shutil.rmtree("logs", ignore_errors=True)
                        # Update state
                        st.session_state.db_initialized = False
                        st.session_state.chat_history = []
                        st.session_state.ingest_message = "✅ Databases reset successfully. Please ingest logs to continue."
                        st.success("Reset complete.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Reset failed: {e}")
    
    def render_chat_tab(self):
        """Render the Chat Assistant tab"""
        st.markdown('<div class="tab-header">🤖 AI-Powered Chat Assistant</div>', unsafe_allow_html=True)

        if not st.session_state.db_initialized:
            st.info("No data available yet. Please use the sidebar to ingest logs (generate or upload) to get started.")
            return

        # Load LLM integration
        llm = self.load_llm_integration()
        if not llm:
            return

        # Chat interface
        st.markdown("Ask questions about your logs, such as:")
        st.markdown("- *What are the most common errors?*")
        st.markdown("- *Explain the boot sequence of the 5G RAN*")
        st.markdown("- *What is MME and what does it do?*")

        # Chat input at the top
        user_input = st.chat_input("Ask a question about your network logs...")
        if user_input:
            # Analyze query intent
            query_intent = analyze_query_intent(user_input)

            # Process query and get response
            with st.spinner("🔍 Analyzing logs..."):
                response = self.process_query_with_context(user_input, llm, query_intent)

            # Add both user message and assistant response to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            # Show contextual visuals based on query intent
            if response and not response.startswith("❌ Error"):
                if any(k in user_input.lower() for k in [
                    'error','warn','issue','problem','failure','alarm','critical'
                ]):
                    self.render_error_correlation_analysis()
                elif any(k in user_input.lower() for k in [
                    'flow','sequence','process','boot','initialization','monitoring','recovery'
                ]):
                    self.render_service_flow_visualization()
                elif any(k in user_input.lower() for k in [
                    'component','MME','SGW','AMF','SMF','VNF','CNF','network'
                ]):
                    self.render_component_interaction_view()
                elif any(k in user_input.lower() for k in [
                    'trend','pattern','timeline','over time','evolution'
                ]):
                    self.render_advanced_timeline_analysis()
                elif any(k in user_input.lower() for k in [
                    'performance','latency','throughput','metrics','kpi'
                ]):
                    self.render_performance_analysis()

            # Clear the input and rerun once
            st.session_state.user_input = ""
            st.rerun()

        # Display chat history in reverse order (newest first)
        for message in reversed(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message.get("content", ""))

        # Quick actions row at the bottom
        qa_col1, qa_col2 = st.columns([1,1])
        with qa_col1:
            if st.button("🧾 Summarize Current Logs"):
                try:
                    with st.spinner("Generating summary from current logs..."):
                        summary = self.process_query_with_context("Summarize the current logs", llm, "log_analysis")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": summary
                    })
                    st.rerun()
                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Could not generate summary: {e}"
                    })
                    st.rerun()
        with qa_col2:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

    
    def render_dashboard_tab(self):
        """Render the Summary Dashboard tab"""
        st.markdown('<div class="tab-header">📊 Summary Dashboard</div>', unsafe_allow_html=True)
        
        if not st.session_state.db_initialized or not os.path.exists(self.db_path):
            st.info("No data available yet. Please ingest logs from the sidebar to view the dashboard.")
            return
        
        try:
            conn = self.get_database_connection()
            
            # Load data
            df = pd.read_sql_query("SELECT * FROM logs", conn)
            if df.empty:
                st.info("No data available. Please ingest logs from the sidebar.")
                conn.close()
                return
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d-%H:%M:%S')
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.info("Please ingest logs from the sidebar to view the dashboard.")
            if 'conn' in locals():
                conn.close()
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_logs = len(df)
            st.metric("Total Logs", total_logs)
        
        with col2:
            error_count = len(df[df['severity'] == 'ERROR'])
            error_percentage = (error_count / total_logs * 100) if total_logs > 0 else 0
            st.metric("Errors", error_count, f"{error_percentage:.1f}%")
        
        with col3:
            warning_count = len(df[df['severity'] == 'WARN'])
            warning_percentage = (warning_count / total_logs * 100) if total_logs > 0 else 0
            st.metric("Warnings", warning_count, f"{warning_percentage:.1f}%")
        
        with col4:
            health_score = max(0, 100 - error_percentage - warning_percentage)
            st.metric("Health Score", f"{health_score:.1f}%")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Severity distribution pie chart
            severity_counts = df['severity'].value_counts()
            fig_pie = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Log Severity Distribution",
                color_discrete_map={
                    'ERROR': '#e53e3e',
                    'WARN': '#dd6b20',
                    'INFO': '#38a169',
                    'DEBUG': '#3182ce'
                }
            )
            st.plotly_chart(fig_pie, width='stretch')
        
        with col2:
            # Error frequency by file
            error_df = df[df['severity'].isin(['ERROR', 'WARN'])]
            file_errors = error_df['filename'].value_counts().head(10)
            if len(file_errors) == 0:
                st.info("No ERROR/WARN logs to summarize by file.")
            else:
                bar_df = pd.DataFrame({
                    'Filename': file_errors.index,
                    'Count': file_errors.values
                })
                fig_bar = px.bar(
                    bar_df,
                    x='Count',
                    y='Filename',
                    orientation='h',
                    title="Top Files with Errors/Warnings",
                    labels={'Count': 'Count', 'Filename': 'Filename'}
                )
                fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_bar, width='stretch')
        
        # Timeline chart
        st.subheader("Error Distribution Over Time")

        # Group by hour for timeline
        df['hour'] = df['timestamp'].dt.floor('h')
        timeline_data = df.groupby(['hour', 'severity']).size().reset_index(name='count')

        # Enhanced debug information and error handling
        st.info(f"📊 **Timeline Analysis:** {len(timeline_data)} data points from {len(df)} total logs across {df['hour'].nunique()} time periods")

        # Show raw data for debugging
        if st.checkbox("🔍 Show Timeline Debug Data", key="timeline_debug"):
            st.write("**Raw Timeline Data:**")
            st.dataframe(timeline_data.head(10))
            st.write(f"**Hour Range:** {df['hour'].min()} to {df['hour'].max()}")
            st.write(f"**Severity Counts:** {df['severity'].value_counts().to_dict()}")

        if timeline_data.empty:
            st.warning("📈 **No Timeline Data:** No data available for timeline chart. Please ensure logs are properly ingested.")
        elif len(timeline_data) < 2:
            st.warning(f"📈 **Insufficient Data:** Only {len(timeline_data)} data point(s) available. Timeline charts need at least 2 points across different time periods.")
            
            # Show what data we have
            if not timeline_data.empty:
                st.write("**Available Data:**")
                for _, row in timeline_data.iterrows():
                    st.write(f"- {row['hour'].strftime('%Y-%m-%d %H:%M')}: {row['severity']} = {row['count']}")
        else:
            try:
                # Enhanced chart with better formatting
                fig_timeline = px.line(
                    timeline_data,
                    x='hour',
                    y='count',
                    color='severity',
                    title="Log Events Over Time (Hourly Aggregation)",
                    color_discrete_map={
                        'ERROR': '#e53e3e',
                        'WARN': '#dd6b20',
                        'INFO': '#38a169',
                        'DEBUG': '#3182ce'
                    },
                    markers=True  # Add markers to make lines more visible
                )
                
                # Enhanced layout with better formatting
                fig_timeline.update_layout(
                    xaxis_title="Time (Hourly)",
                    yaxis_title="Event Count",
                    showlegend=True,
                    hovermode='x unified',
                    xaxis=dict(
                        tickformat='%H:%M\n%m-%d',
                        tickangle=45
                    ),
                    height=400
                )
                
                # Add grid lines for better readability
                fig_timeline.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                fig_timeline.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                
                st.plotly_chart(fig_timeline, use_container_width=True)

                # Enhanced insights with more details
                try:
                    time_span_hours = (df['hour'].max() - df['hour'].min()).total_seconds() / 3600
                    peak_hour = timeline_data.loc[timeline_data['count'].idxmax()]
                    total_events = timeline_data['count'].sum()
                    
                    st.success(f"""
                    📊 **Timeline Insights:**
                    - **Time Span:** {df['hour'].min().strftime('%H:%M %m-%d')} to {df['hour'].max().strftime('%H:%M %m-%d')} ({time_span_hours:.1f} hours)
                    - **Peak Activity:** {peak_hour['hour'].strftime('%H:%M %m-%d')} with {peak_hour['count']} {peak_hour['severity']} events
                    - **Total Events:** {total_events} across all time periods
                    - **Average per Hour:** {total_events / df['hour'].nunique():.1f} events
                    """)
                except Exception as insight_error:
                    st.info(f"📊 **Timeline Insights:** Data spans {df['hour'].nunique()} unique time periods (insight calculation error: {str(insight_error)})")

            except Exception as chart_error:
                st.error(f"📈 **Chart Rendering Error:** {str(chart_error)}")
                st.info("**Possible causes:**")
                st.info("- Timestamp formatting issues")
                st.info("- Insufficient data variety")
                st.info("- Memory or processing constraints")
                
                # Fallback: Show simple table
                st.write("**Fallback - Timeline Data Table:**")
                st.dataframe(timeline_data)
        
        # Recent critical issues
        st.subheader("🚨 Recent Critical Issues")
        recent_errors = df[df['severity'] == 'ERROR'].sort_values('timestamp', ascending=False).head(5)

        if recent_errors.empty:
            st.info("✅ **System Health:** No critical errors found in the current dataset. All systems appear to be running smoothly!")
        else:
            for _, row in recent_errors.iterrows():
                st.markdown(f"""
                <div class="error-card">
                    <strong>{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</strong> |
                    <code>{row['filename']}:{row['line_number']}</code><br>
                    {row['message']}
                </div>
                """, unsafe_allow_html=True)
        
        conn.close()

    def process_query_with_context(self, user_input, llm, query_intent):
        """Process user query with network assurance context and conversation history"""
        try:
            # Get conversation history from session state
            conversation_history = self.get_conversation_history()

            # Get relevant logs from database
            conn = self.get_database_connection()

            # Use semantic search if available, otherwise use text search
            try:
                # Try semantic search first
                relevant_logs = self.get_semantic_search_results(user_input, conn)
            except Exception:
                # Fallback to text search
                relevant_logs = self.get_text_search_results(user_input, conn)

            conn.close()

            # Generate context-aware prompt
            if query_intent == "log_analysis":
                # For log-specific questions, include log data and flow context
                prompt = self.generate_log_analysis_prompt(user_input, relevant_logs)
            elif query_intent == "flow_explanation":
                # For flow questions, focus on service flow knowledge
                prompt = self.generate_flow_explanation_prompt(user_input, relevant_logs)
            else:
                # For general network assurance questions, use knowledge base
                prompt = self.generate_general_knowledge_prompt(user_input, relevant_logs)

            # Process through LLM with conversation history
            result = llm.process_query_with_history(prompt, conversation_history)

            return result.get('response', '')

        except Exception as e:
            return f"❌ Error processing query: {str(e)}"

    def get_conversation_history(self):
        """Get conversation history for context"""
        if not st.session_state.chat_history:
            return ""

        # Get last 5 exchanges (10 messages) for context
        recent_messages = st.session_state.chat_history[-10:]

        history = "CONVERSATION HISTORY:\n"
        for i, msg in enumerate(recent_messages):
            role = "USER" if msg["role"] == "user" else "ASSISTANT"
            history += f"{role}: {msg['content']}\n"

        return history

    def get_semantic_search_results(self, query, conn, limit=10):
        """Get relevant logs using semantic search"""
        try:
            from chromadb.utils import embedding_functions
            from chromadb.config import Settings
            import chromadb

            # Initialize ChromaDB client
            chroma_client = chromadb.PersistentClient(path="./chroma_db")
            collection = chroma_client.get_collection("log_embeddings")

            # Get similar logs using semantic search
            results = collection.query(
                query_texts=[query],
                n_results=limit
            )

            # Get log IDs from results
            log_ids = [id_ for id_list in results.get('ids', []) for id_ in id_list]

            # Get full log entries from SQLite
            placeholders = ','.join(['?'] * len(log_ids))
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM logs
                WHERE id IN ({placeholders})
                ORDER BY timestamp DESC
            """, log_ids)

            return cursor.fetchall()

        except Exception as e:
            raise Exception(f"Semantic search failed: {str(e)}")

    def get_text_search_results(self, query, conn, limit=10):
        """Get relevant logs using text search"""
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM logs
            WHERE message LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"%{query}%", limit))
        return cursor.fetchall()

    def generate_log_analysis_prompt(self, query, relevant_logs):
        """Generate prompt for log analysis questions"""
        # Prepare log context
        log_context = ""
        if relevant_logs:
            log_context = "\n".join([
                f"[{row[1]}] [{row[4]}] [{row[8]}] {row[5]}"  # timestamp, severity, module, message
                for row in relevant_logs[:5]  # Limit to top 5 logs
            ])

        return f"""
You are a Network Assurance Expert AI. Use your knowledge of network assurance systems and the provided log data to answer questions.

Network Components:
- 4G Core: {', '.join(NETWORK_ASSURANCE_KNOWLEDGE['network_components']['4g_core']['components'])}
- 5G RAN: {', '.join(NETWORK_ASSURANCE_KNOWLEDGE['network_components']['5g_ran']['components'])}

Service Flow:
1. Boot/Initialization: {', '.join(NETWORK_ASSURANCE_KNOWLEDGE['service_flow']['boot_initialization'])}
2. Active Monitoring: {', '.join(NETWORK_ASSURANCE_KNOWLEDGE['service_flow']['active_monitoring'])}
3. Recovery/Optimization: {', '.join(NETWORK_ASSURANCE_KNOWLEDGE['service_flow']['recovery_optimization'])}

Relevant Logs:
{log_context}

Question: {query}

Please analyze the logs and provide a comprehensive answer considering:
- The network assurance flow and component relationships
- Specific issues mentioned in the logs
- Patterns and trends in the data
- Recommendations for resolution

Structure your response with clear headings and bullet points.
"""

    def generate_flow_explanation_prompt(self, query, relevant_logs):
        """Generate prompt for flow explanation questions"""
        return f"""
You are a Network Assurance Expert AI. Explain the network assurance service flow and processes.

Service Flow:
1. Boot/Initialization: {', '.join(NETWORK_ASSURANCE_KNOWLEDGE['service_flow']['boot_initialization'])}
2. Active Monitoring: {', '.join(NETWORK_ASSURANCE_KNOWLEDGE['service_flow']['active_monitoring'])}
3. Recovery/Optimization: {', '.join(NETWORK_ASSURANCE_KNOWLEDGE['service_flow']['recovery_optimization'])}

Network Components:
- 4G Core: {', '.join(NETWORK_ASSURANCE_KNOWLEDGE['network_components']['4g_core']['components'])}
- 5G RAN: {', '.join(NETWORK_ASSURANCE_KNOWLEDGE['network_components']['5g_ran']['components'])}

Question: {query}

Please explain the requested aspect of the network assurance flow, using specific examples from the logs where relevant.
"""

    def generate_general_knowledge_prompt(self, query, relevant_logs):
        """Generate prompt for general network assurance questions"""
        return f"""
You are a Network Assurance Expert AI. Answer general questions about network assurance systems.

Network Components:
- 4G Core: {', '.join(NETWORK_ASSURANCE_KNOWLEDGE['network_components']['4g_core']['components'])}
- 5G RAN: {', '.join(NETWORK_ASSURANCE_KNOWLEDGE['network_components']['5g_ran']['components'])}

Component Details:
{chr(10).join([f"- {k}: {v}" for k, v in NETWORK_ASSURANCE_KNOWLEDGE['component_details'].items()])}

Question: {query}

Please provide a comprehensive explanation of the requested network assurance concept or component.
"""
    def render_error_correlation_analysis(self):
        """Render error correlation and impact analysis visualization"""
        try:
            conn = self.get_database_connection()
            df = pd.read_sql_query("SELECT * FROM logs WHERE severity IN ('ERROR', 'WARN')", conn)
            conn.close()

            if df.empty:
                st.info("📊 No errors or warnings found for correlation analysis.")
                return

            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d-%H:%M:%S', errors='coerce')
            cutoff = datetime.now() - timedelta(hours=24)
            df_recent = df[df['timestamp'] >= cutoff]

            if df_recent.empty:
                st.info("📊 No recent errors or warnings for correlation analysis.")
                return

            st.markdown("---")
            st.markdown("### 🔍 Error Correlation & Impact Analysis")

            # Error frequency by module
            col1, col2 = st.columns(2)

            with col1:
                module_errors = df_recent.groupby(['module', 'severity']).size().reset_index(name='count')
                if not module_errors.empty:
                    fig = px.bar(
                        module_errors,
                        x='module',
                        y='count',
                        color='severity',
                        title="Error Distribution by Module",
                        color_discrete_map={'ERROR': '#e53e3e', 'WARN': '#dd6b20'}
                    )
                    fig.update_layout(xaxis_title="Module", yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Error timeline with pattern detection
                df_recent['hour'] = df_recent['timestamp'].dt.floor('h')
                timeline_data = df_recent.groupby(['hour', 'severity']).size().reset_index(name='count')

                if not timeline_data.empty:
                    fig = px.line(
                        timeline_data,
                        x='hour',
                        y='count',
                        color='severity',
                        title="Error Timeline (Last 24h)",
                        markers=True,
                        color_discrete_map={'ERROR': '#e53e3e', 'WARN': '#dd6b20'}
                    )
                    fig.update_layout(xaxis_title="Time", yaxis_title="Error Count")
                    st.plotly_chart(fig, use_container_width=True)

            # Error correlation heatmap (component vs module)
            st.markdown("#### 🔗 Component-Module Error Correlation")
            correlation_data = df_recent.groupby(['component', 'module']).size().reset_index(name='error_count')

            if len(correlation_data) > 1:
                # Create a pivot table for heatmap
                pivot = correlation_data.pivot(index='component', columns='module', values='error_count').fillna(0)

                if not pivot.empty:
                    fig = px.imshow(
                        pivot,
                        text_auto=True,
                        aspect="auto",
                        title="Error Correlation Heatmap",
                        color_continuous_scale="Reds"
                    )
                    fig.update_layout(xaxis_title="Module", yaxis_title="Component")
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error rendering correlation analysis: {str(e)}")

    def render_service_flow_visualization(self):
        """Render service flow progress and component interactions"""
        try:
            conn = self.get_database_connection()
            df = pd.read_sql_query("SELECT * FROM logs", conn)
            conn.close()

            if df.empty:
                st.info("📊 No data available for service flow visualization.")
                return

            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d-%H:%M:%S', errors='coerce')
            cutoff = datetime.now() - timedelta(hours=12)
            df_recent = df[df['timestamp'] >= cutoff]

            st.markdown("---")
            st.markdown("### 🔄 Service Flow Progress & Component Interactions")

            # Component activity over time
            col1, col2 = st.columns(2)

            with col1:
                # Module activity timeline
                df_recent['minute'] = df_recent['timestamp'].dt.floor('10min')  # 10-minute intervals
                activity_data = df_recent.groupby(['minute', 'module']).size().reset_index(name='activity_count')

                if not activity_data.empty:
                    # Get top 5 most active modules
                    top_modules = activity_data.groupby('module')['activity_count'].sum().nlargest(5).index
                    filtered_data = activity_data[activity_data['module'].isin(top_modules)]

                    fig = px.line(
                        filtered_data,
                        x='minute',
                        y='activity_count',
                        color='module',
                        title="Module Activity Over Time",
                        markers=True
                    )
                    fig.update_layout(xaxis_title="Time", yaxis_title="Activity Count")
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Component interaction network
                component_activity = df_recent.groupby(['component', 'severity']).size().reset_index(name='count')

                if not component_activity.empty:
                    fig = px.treemap(
                        component_activity,
                        path=['component'],
                        values='count',
                        color='severity',
                        title="Component Activity Distribution",
                        color_discrete_map={'ERROR': '#e53e3e', 'WARN': '#dd6b20', 'INFO': '#38a169', 'DEBUG': '#3182ce'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error rendering service flow visualization: {str(e)}")

    def render_component_interaction_view(self):
        """Render component interaction and network topology view"""
        try:
            conn = self.get_database_connection()
            df = pd.read_sql_query("SELECT * FROM logs WHERE component != 'Unknown'", conn)
            conn.close()

            if df.empty:
                st.info("📊 No component data available for interaction analysis.")
                return

            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d-%H:%M:%S', errors='coerce')
            cutoff = datetime.now() - timedelta(hours=24)
            df_recent = df[df['timestamp'] >= cutoff]

            st.markdown("---")
            st.markdown("### 🌐 Component Interaction & Network Topology")

            # Component activity radar chart
            component_stats = df_recent.groupby(['component', 'severity']).size().reset_index(name='count')

            if not component_stats.empty:
                # Create radar chart data
                components = component_stats['component'].unique()

                fig = go.Figure()

                colors = {'ERROR': '#e53e3e', 'WARN': '#dd6b20', 'INFO': '#38a169', 'DEBUG': '#3182ce'}

                for severity in component_stats['severity'].unique():
                    severity_data = component_stats[component_stats['severity'] == severity]

                    # Create mapping for radar chart
                    values = []
                    for comp in components:
                        count = severity_data[severity_data['component'] == comp]['count'].iloc[0] if comp in severity_data['component'].values else 0
                        values.append(count)

                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=components,
                        fill='toself',
                        name=severity,
                        line=dict(color=colors.get(severity, '#3182ce'))
                    ))

                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    title="Component Activity Radar",
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error rendering component interaction view: {str(e)}")

    def render_advanced_timeline_analysis(self):
        """Render advanced timeline with insights and pattern detection"""
        try:
            conn = self.get_database_connection()
            df = pd.read_sql_query("SELECT * FROM logs", conn)
            conn.close()

            if df.empty:
                st.info("📊 No data available for timeline analysis.")
                return

            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d-%H:%M:%S', errors='coerce')
            cutoff = datetime.now() - timedelta(hours=48)
            df_recent = df[df['timestamp'] >= cutoff]

            st.markdown("---")
            st.markdown("### 📈 Advanced Timeline Analysis with Insights")

            if not df_recent.empty:
                # Multi-level timeline with different granularities
                col1, col2 = st.columns(2)

                with col1:
                    # Hourly aggregation
                    df_recent['hour'] = df_recent['timestamp'].dt.floor('h')
                    hourly_data = df_recent.groupby(['hour', 'severity']).size().reset_index(name='count')

                    fig = px.area(
                        hourly_data,
                        x='hour',
                        y='count',
                        color='severity',
                        title="Event Distribution by Hour",
                        color_discrete_map={'ERROR': '#e53e3e', 'WARN': '#dd6b20', 'INFO': '#38a169', 'DEBUG': '#3182ce'}
                    )
                    fig.update_layout(xaxis_title="Time", yaxis_title="Event Count")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Pattern detection - show potential issues
                    error_spikes = self.detect_error_spikes(df_recent)

                    if error_spikes:
                        st.markdown("#### 🚨 Detected Error Patterns:")
                        for spike in error_spikes:
                            st.markdown(f"""
                            <div style='background-color: #fff5f5; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 4px solid #e53e3e;'>
                                <strong>Spike Detected:</strong> {spike['module']} - {spike['count']} errors in {spike['timeframe']}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("✅ No significant error spikes detected in the analysis period.")

        except Exception as e:
            st.error(f"Error rendering advanced timeline analysis: {str(e)}")

    def render_performance_analysis(self):
        """Render performance trends and anomaly detection"""
        try:
            conn = self.get_database_connection()
            df = pd.read_sql_query("SELECT * FROM logs", conn)
            conn.close()

            if df.empty:
                st.info("📊 No data available for performance analysis.")
                return

            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d-%H:%M:%S', errors='coerce')
            cutoff = datetime.now() - timedelta(hours=24)
            df_recent = df[df['timestamp'] >= cutoff]

            st.markdown("---")
            st.markdown("### ⚡ Performance Trends & Anomaly Detection")

            # Performance metrics extraction and visualization
            col1, col2 = st.columns(2)

            with col1:
                # Module performance over time
                df_recent['minute'] = df_recent['timestamp'].dt.floor('5min')
                performance_data = df_recent.groupby(['minute', 'module']).size().reset_index(name='activity')

                if not performance_data.empty:
                    # Calculate rolling average for trend detection
                    performance_data = performance_data.sort_values('minute')
                    performance_data['rolling_avg'] = performance_data.groupby('module')['activity'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)

                    fig = px.line(
                        performance_data,
                        x='minute',
                        y=['activity', 'rolling_avg'],
                        color='module',
                        title="Module Performance Trends",
                        markers=True
                    )
                    fig.update_layout(xaxis_title="Time", yaxis_title="Activity Level")
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Anomaly detection based on activity patterns
                if not performance_data.empty:
                    # Simple anomaly detection - points above 2 standard deviations
                    stats = performance_data.groupby('module')['activity'].agg(['mean', 'std']).reset_index()

                    anomalies = []
                    for _, row in performance_data.iterrows():
                        module_stats = stats[stats['module'] == row['module']]
                        if not module_stats.empty:
                            mean_val = module_stats.iloc[0]['mean']
                            std_val = module_stats.iloc[0]['std']
                            if row['activity'] > mean_val + (2 * std_val):
                                anomalies.append({
                                    'module': row['module'],
                                    'timestamp': row['minute'],
                                    'activity': row['activity'],
                                    'threshold': mean_val + (2 * std_val)
                                })

                    if anomalies:
                        anomaly_df = pd.DataFrame(anomalies)
                        fig = px.scatter(
                            anomaly_df,
                            x='timestamp',
                            y='activity',
                            color='module',
                            title="Detected Performance Anomalies",
                            size='activity',
                            hover_data=['threshold']
                        )
                        fig.update_layout(xaxis_title="Time", yaxis_title="Activity Level")
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("#### 🚨 Performance Anomalies Detected:")
                        for anomaly in anomalies[:5]:  # Show top 5
                            st.markdown(f"""
                            <div style='background-color: #fff3cd; padding: 8px; border-radius: 4px; margin: 3px 0; border-left: 3px solid #ffc107;'>
                                <strong>{anomaly['module']}</strong> at {anomaly['timestamp'].strftime('%H:%M')} - Activity: {anomaly['activity']} (threshold: {anomaly['threshold']:.1f})
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("✅ No significant performance anomalies detected.")

        except Exception as e:
            st.error(f"Error rendering performance analysis: {str(e)}")

    def detect_error_spikes(self, df, window_minutes=30, threshold_multiplier=3):
        """Detect error spikes in the data"""
        spikes = []

        # Group by module and time windows
        df['time_window'] = df['timestamp'].dt.floor(f'{window_minutes}min')

        for module in df['module'].unique():
            module_data = df[df['module'] == module]
            window_counts = module_data.groupby('time_window').size()

            if not window_counts.empty:
                mean_count = window_counts.mean()
                std_count = window_counts.std()

                for window, count in window_counts.items():
                    if count > mean_count + (threshold_multiplier * std_count):
                        spikes.append({
                            'module': module,
                            'timeframe': f"{window.strftime('%H:%M')}-{window.strftime('%H:%M')}",
                            'count': count
                        })

        return sorted(spikes, key=lambda x: x['count'], reverse=True)

    def render_table_tab(self):
        st.markdown('<div class="tab-header">📋 Error Frequency Table</div>', unsafe_allow_html=True)

        if not st.session_state.db_initialized or not os.path.exists(self.db_path):
            st.info("No data available yet. Please ingest logs from the sidebar to view and filter log entries.")
            return

        try:
            conn = self.get_database_connection()
        except Exception as e:
            st.error(f"Error connecting to database: {str(e)}")
            st.info("Please ingest logs from the sidebar.")
            return

        # Filters
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            severity_filter = st.multiselect(
                "Filter by Severity",
                options=['ERROR', 'WARN', 'INFO', 'DEBUG'],
                default=['ERROR', 'WARN']
            )

        with col2:
            # Get unique filenames
            try:
                filenames_df = pd.read_sql_query("SELECT DISTINCT filename FROM logs", conn)
                filename_options = filenames_df['filename'].tolist()
            except Exception:
                filename_options = []
            filename_filter = st.multiselect(
                "Filter by Filename",
                options=filename_options,
                default=[]
            )

        with col3:
            search_term = st.text_input("Search in Messages", "")

        with col4:
            limit_mode = st.selectbox(
                "Time/Count Filter",
                options=["None", "Recent N", "Last X hours", "Between (absolute)"],
                index=0
            )
            n_limit = None
            hours_back = None
            start_dt = None
            end_dt = None
            if limit_mode == "Recent N":
                n_limit = st.number_input("N", min_value=1, max_value=1000, value=5, step=1)
            elif limit_mode == "Last X hours":
                hours_back = st.number_input("Hours", min_value=1, max_value=168, value=24, step=1)
            elif limit_mode == "Between (absolute)":
                now_dt = datetime.now()
                default_start_date = (now_dt - timedelta(hours=24)).date()
                default_end_date = now_dt.date()
                default_start_time = (now_dt - timedelta(hours=24)).time().replace(microsecond=0)
                default_end_time = now_dt.time().replace(microsecond=0)
                sd = st.date_input("Start date", value=default_start_date, key="abs_start_date")
                stime = st.time_input("Start time", value=default_start_time, key="abs_start_time")
                ed = st.date_input("End date", value=default_end_date, key="abs_end_date")
                etime = st.time_input("End time", value=default_end_time, key="abs_end_time")
                try:
                    start_dt = datetime.combine(sd, stime)
                    end_dt = datetime.combine(ed, etime)
                except Exception:
                    start_dt = None
                    end_dt = None

        # Build query based on filters
        query = "SELECT * FROM logs WHERE 1=1"
        params = []

        if severity_filter:
            placeholders = ','.join(['?' for _ in severity_filter])
            query += f" AND severity IN ({placeholders})"
            params.extend(severity_filter)

        if filename_filter:
            placeholders = ','.join(['?' for _ in filename_filter])
            query += f" AND filename IN ({placeholders})"
            params.extend(filename_filter)

        if search_term:
            query += " AND message LIKE ?"
            params.append(f"%{search_term}%")

        # Time-based filters
        if limit_mode == "Last X hours" and hours_back is not None:
            cutoff = datetime.now() - timedelta(hours=int(hours_back))
            cutoff_str = cutoff.strftime('%Y-%m-%d-%H:%M:%S')
            query += " AND timestamp >= ?"
            params.append(cutoff_str)
        elif limit_mode == "Between (absolute)" and start_dt is not None and end_dt is not None:
            if start_dt > end_dt:
                # Swap if user entered reversed dates
                start_dt, end_dt = end_dt, start_dt
            start_str = start_dt.strftime('%Y-%m-%d-%H:%M:%S')
            end_str = end_dt.strftime('%Y-%m-%d-%H:%M:%S')
            query += " AND timestamp BETWEEN ? AND ?"
            params.extend([start_str, end_str])

        # Always order by most recent first for consistent display
        query += " ORDER BY timestamp DESC"

        # Count-based limit
        if limit_mode == "Recent N" and n_limit is not None:
            query += " LIMIT ?"
            params.append(int(n_limit))

        # Load filtered data
        try:
            df = pd.read_sql_query(query, conn, params=params)
        except Exception as e:
            st.error(f"Error querying logs: {str(e)}")
            st.info("Please ingest logs from the sidebar.")
            conn.close()
            return

        # Display results count
        st.info(f"Found {len(df)} log entries matching your filters")

        if len(df) > 0:
            # Format timestamp for better display
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d-%H:%M:%S')
            df['formatted_timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

            # Display table
            display_df = df[['formatted_timestamp', 'filename', 'line_number', 'severity', 'message', 'log_file_source']]
            display_df.columns = ['Timestamp', 'Filename', 'Line', 'Severity', 'Message', 'Source File']

            # Color code severity
            def color_severity(val):
                if val == 'ERROR':
                    return 'background-color: #fff5f5; color: #e53e3e; font-weight: bold'
                elif val == 'WARN':
                    return 'background-color: #fffbf0; color: #dd6b20; font-weight: bold'
                elif val == 'INFO':
                    return 'background-color: #f0fff4; color: #38a169'
                elif val == 'DEBUG':
                    return 'background-color: #f7fafc; color: #3182ce'
                return ''

            styled_df = display_df.style.map(color_severity, subset=['Severity'])
            st.dataframe(styled_df, use_container_width=True, height=600)

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv,
                file_name=f"filtered_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        conn.close()
    
    def run(self):
        """Main application runner"""
        # Header
        st.markdown('<div class="main-header">💊 Service Cure Insights</div>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs(["🤖 Chat Assistant", "📊 Summary Dashboard", "📋 Error Frequency Table"])
        
        with tab1:
            self.render_chat_tab()
        
        with tab2:
            self.render_dashboard_tab()
        
        with tab3:
            self.render_table_tab()

def main():
    """Main function to run the Streamlit app"""
    app = LogAnalysisApp()
    app.run()

if __name__ == "__main__":
    main()
