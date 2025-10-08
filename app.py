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
        """Render sidebar with system information"""
        st.sidebar.title("💊 Service Cure Insights")
        
        if st.session_state.db_initialized:
            st.sidebar.success("✅ Database Connected")
        else:
            st.sidebar.info("💡 Load logs to get started")
        
        st.sidebar.subheader("📥 Data Ingestion")

        # Generate sample logs and ingest
        if st.sidebar.button("🧪 Generate Sample Logs and Ingest"):
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
        uploaded = st.sidebar.file_uploader("Upload .log files", type=["log", "txt"], accept_multiple_files=True)
        if uploaded:
            st.session_state.uploaded_logs = uploaded
        if st.session_state.uploaded_logs:
            if st.sidebar.button("📦 Ingest Uploaded Logs"):
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
            st.sidebar.info(st.session_state.ingest_message)

        # Ingest only selected existing logs (service-specific ingest)
        if os.path.exists("logs"):
            try:
                existing_logs = [f for f in os.listdir("logs") if f.lower().endswith((".log", ".txt"))]
            except Exception:
                existing_logs = []
            if existing_logs:
                selected_logs = st.sidebar.multiselect("Select logs to ingest", options=existing_logs, default=[])
                if st.sidebar.button("➡️ Ingest Selected Existing Logs") and selected_logs:
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
        st.sidebar.subheader("🧹 Maintenance")
        delete_logs = st.sidebar.checkbox("Also delete log files", value=False)
        if st.sidebar.button("🧽 Soft Reset (Truncate Data)"):
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
        if st.sidebar.button("♻️ Reset Databases"):
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
        
        # Chat input at the top
        user_input = st.chat_input("Ask a question about your logs...")
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Process query with LLM (always structured output)
            try:
                with st.spinner("Analyzing logs and generating response..."):
                    style_guide = (
                        "Please answer in this exact structure using Markdown headings and short bullet lists:\n"
                        "### **Root Cause / Main Issue** (Short, precise explanation of what caused the issue.)\n"
                        "### **Analysis** (A detailed breakdown of patterns, trends, and observations.)\n"
                        "### **Solution** (Specific, actionable steps to resolve or mitigate the problem.)\n"
                        "### **Summary** (Short, precise explanation)\n"
                        "Use **bold** for section headings and keep content concise."
                    )
                    composed_query = f"{style_guide}\n\nUser question: {user_input}"
                    result = llm.process_query(composed_query)
                # Add assistant response to chat history (no raw logs shown)
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": result.get('response', '')
                })
                
                # Contextual visuals inline (based on query intent)
                keywords = [
                    'error','warn','distribution','trend','spike','top','most',
                    'frequency','chart','graph','pie','timeline','count','rate'
                ]
                if any(k in user_input.lower() for k in keywords):
                    try:
                        conn = self.get_database_connection()
                        df_vis = pd.read_sql_query("SELECT * FROM logs", conn)
                        conn.close()
                        if not df_vis.empty:
                            df_vis['timestamp'] = pd.to_datetime(df_vis['timestamp'], format='%Y-%m-%d-%H:%M:%S', errors='coerce')
                            cutoff = datetime.now() - timedelta(hours=24)
                            df_24 = df_vis[df_vis['timestamp'] >= cutoff]
                            st.markdown("---")
                            st.markdown("### Visual context (last 24h)")
                            v1, v2 = st.columns(2)
                            with v1:
                                sev_counts = df_24['severity'].value_counts()
                                if not sev_counts.empty:
                                    fig_pie = px.pie(values=sev_counts.values, names=sev_counts.index, title="Severity Distribution")
                                    st.plotly_chart(fig_pie, width='stretch')
                                else:
                                    st.info("No data in last 24h for severity distribution.")
                            with v2:
                                if not df_24.empty:
                                    df_24['hour'] = df_24['timestamp'].dt.floor('h')
                                    tdata = df_24.groupby(['hour', 'severity']).size().reset_index(name='count')
                                    fig_line = px.line(tdata, x='hour', y='count', color='severity', title="Events Over Time")
                                    st.plotly_chart(fig_line, width='stretch')
                                else:
                                    st.info("No events in last 24h to show timeline.")
                    except Exception:
                        pass
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Encountered an error while generating a response: {e}"
                })
            st.rerun()
        
        # Display chat history in reverse order (newest first)
        for message in reversed(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"]) 
            else:
                with st.chat_message("assistant"):
                    try:
                        content = message.get("content", "")
                        st.markdown(content)
                    except Exception as e:
                        st.exception(e)
        
        # Quick actions row at the bottom
        qa_col1, qa_col2 = st.columns([1,1])
        with qa_col1:
            if st.button("🧾 Summarize Current Logs"):
                try:
                    with st.spinner("Generating summary from current logs..."):
                        summary = llm.get_error_summary()
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
        
        fig_timeline = px.line(
            timeline_data,
            x='hour',
            y='count',
            color='severity',
            title="Log Events Over Time",
            color_discrete_map={
                'ERROR': '#e53e3e',
                'WARN': '#dd6b20',
                'INFO': '#38a169',
                'DEBUG': '#3182ce'
            }
        )
        st.plotly_chart(fig_timeline, width='stretch')
        
        # Recent critical issues
        st.subheader("🚨 Recent Critical Issues")
        recent_errors = df[df['severity'] == 'ERROR'].sort_values('timestamp', ascending=False).head(5)
        
        for _, row in recent_errors.iterrows():
            st.markdown(f"""
            <div class="error-card">
                <strong>{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</strong> | 
                <code>{row['filename']}:{row['line_number']}</code><br>
                {row['message']}
            </div>
            """, unsafe_allow_html=True)
        
        conn.close()
    
    def render_table_tab(self):
        """Render the Error Frequency Table tab"""
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
