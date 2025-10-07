import sqlite3
import chromadb
import os
import re
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pandas as pd

class DatabaseSetup:
    """
    Sets up SQLite database for structured log storage and ChromaDB for vector embeddings.
    Processes log files and stores them in both databases for efficient querying.
    """
    
    def __init__(self, db_path="logs_database.db", chroma_path="./chroma_db"):
        self.db_path = db_path
        self.chroma_path = chroma_path
        self.sqlite_conn = None
        self.chroma_client = None
        self.chroma_collection = None
        
        # Initialize sentence transformer for embeddings
        print("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully!")
    
    def setup_sqlite(self):
        """Initialize SQLite database and create logs table"""
        print("Setting up SQLite database...")
        
        self.sqlite_conn = sqlite3.connect(self.db_path)
        cursor = self.sqlite_conn.cursor()
        
        # Create logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                line_number INTEGER NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                log_file_source TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON logs(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_severity ON logs(severity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename ON logs(filename)')
        
        self.sqlite_conn.commit()
        print("SQLite database setup completed!")
    
    def setup_chromadb(self):
        """Initialize ChromaDB for vector storage"""
        print("Setting up ChromaDB...")
        
        # Create ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        
        # Create or get collection
        try:
            self.chroma_collection = self.chroma_client.get_collection("log_embeddings")
            print("Found existing ChromaDB collection")
        except:
            self.chroma_collection = self.chroma_client.create_collection(
                name="log_embeddings",
                metadata={"description": "Log message embeddings for semantic search"}
            )
            print("Created new ChromaDB collection")
        
        print("ChromaDB setup completed!")
    
    def parse_log_entry(self, log_line, source_file):
        """Parse a single log entry into components"""
        # Expected format: YYYY-MM-DD-HH:MM:SS filename line_number SEVERITY message
        pattern = r'^(\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\d+)\s+(ERROR|WARN|INFO|DEBUG)\s+(.+)$'
        
        match = re.match(pattern, log_line.strip())
        if match:
            timestamp, filename, line_number, severity, message = match.groups()
            return {
                'timestamp': timestamp,
                'filename': filename,
                'line_number': int(line_number),
                'severity': severity,
                'message': message,
                'log_file_source': source_file
            }
        return None
    
    def insert_logs_to_sqlite(self, log_entries):
        """Insert parsed log entries into SQLite database"""
        cursor = self.sqlite_conn.cursor()
        
        for entry in log_entries:
            cursor.execute('''
                INSERT INTO logs (timestamp, filename, line_number, severity, message, log_file_source)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                entry['timestamp'],
                entry['filename'],
                entry['line_number'],
                entry['severity'],
                entry['message'],
                entry['log_file_source']
            ))
        
        self.sqlite_conn.commit()
        print(f"Inserted {len(log_entries)} entries into SQLite database")
    
    def insert_logs_to_chromadb(self, log_entries):
        """Insert log entries into ChromaDB with embeddings"""
        if not log_entries:
            return
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for i, entry in enumerate(log_entries):
            # Create a comprehensive text for embedding
            document_text = f"{entry['severity']} in {entry['filename']}: {entry['message']}"
            documents.append(document_text)
            
            # Metadata for filtering and context
            metadata = {
                'timestamp': entry['timestamp'],
                'filename': entry['filename'],
                'line_number': entry['line_number'],
                'severity': entry['severity'],
                'log_file_source': entry['log_file_source']
            }
            metadatas.append(metadata)
            
            # Unique ID for each entry
            ids.append(f"{entry['log_file_source']}_{i}_{entry['timestamp']}")
        
        # Generate embeddings
        print("Generating embeddings for log entries...")
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Insert into ChromaDB
        self.chroma_collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Inserted {len(log_entries)} entries into ChromaDB")
    
    def process_log_file(self, log_file_path):
        """Process a single log file and extract entries"""
        print(f"Processing log file: {log_file_path}")
        
        if not os.path.exists(log_file_path):
            print(f"Warning: Log file {log_file_path} not found")
            return []
        
        log_entries = []
        source_file = os.path.basename(log_file_path)
        
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                entry = self.parse_log_entry(line, source_file)
                if entry:
                    log_entries.append(entry)
                else:
                    print(f"Warning: Could not parse line {line_num} in {log_file_path}: {line.strip()}")
        
        print(f"Successfully parsed {len(log_entries)} entries from {log_file_path}")
        return log_entries
    
    def load_all_logs(self):
        """Load all log files from the logs directory"""
        logs_dir = Path("logs")
        
        if not logs_dir.exists():
            print("Error: logs directory not found. Please run log_generator.py first.")
            return
        
        all_log_entries = []
        
        # Process all .log files in the logs directory
        for log_file in logs_dir.glob("*.log"):
            entries = self.process_log_file(log_file)
            all_log_entries.extend(entries)
        
        if not all_log_entries:
            print("No log entries found. Please check your log files.")
            return
        
        print(f"Total log entries processed: {len(all_log_entries)}")
        
        # Insert into both databases
        self.insert_logs_to_sqlite(all_log_entries)
        self.insert_logs_to_chromadb(all_log_entries)
        
        # Print summary statistics
        self.print_database_summary()
    
    def print_database_summary(self):
        """Print summary statistics of the loaded data"""
        cursor = self.sqlite_conn.cursor()
        
        # Total count
        cursor.execute("SELECT COUNT(*) FROM logs")
        total_count = cursor.fetchone()[0]
        
        # Severity distribution
        cursor.execute("SELECT severity, COUNT(*) FROM logs GROUP BY severity ORDER BY COUNT(*) DESC")
        severity_stats = cursor.fetchall()
        
        # File distribution
        cursor.execute("SELECT log_file_source, COUNT(*) FROM logs GROUP BY log_file_source")
        file_stats = cursor.fetchall()
        
        print("\n" + "="*50)
        print("DATABASE SUMMARY")
        print("="*50)
        print(f"Total log entries: {total_count}")
        
        print("\nSeverity Distribution:")
        for severity, count in severity_stats:
            percentage = (count / total_count) * 100
            print(f"  {severity}: {count} ({percentage:.1f}%)")
        
        print("\nFile Distribution:")
        for filename, count in file_stats:
            print(f"  {filename}: {count} entries")
        
        # ChromaDB collection info
        collection_count = self.chroma_collection.count()
        print(f"\nChromaDB collection size: {collection_count} embeddings")
        print("="*50)
    
    def test_databases(self):
        """Test both databases with sample queries"""
        print("\nTesting database connections...")
        
        # Test SQLite
        cursor = self.sqlite_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM logs WHERE severity = 'ERROR'")
        error_count = cursor.fetchone()[0]
        print(f"SQLite test - ERROR logs count: {error_count}")
        
        # Test ChromaDB with a sample query
        results = self.chroma_collection.query(
            query_texts=["database connection error"],
            n_results=3
        )
        print(f"ChromaDB test - Found {len(results['documents'][0])} similar logs for 'database connection error'")
        
        print("Database tests completed successfully!")
    
    def close_connections(self):
        """Close database connections"""
        if self.sqlite_conn:
            self.sqlite_conn.close()
        print("Database connections closed")

def main():
    """Main function to set up databases and load logs"""
    print("Starting database setup...")
    
    db_setup = DatabaseSetup()
    
    try:
        # Setup databases
        db_setup.setup_sqlite()
        db_setup.setup_chromadb()
        
        # Load log data
        db_setup.load_all_logs()
        
        # Test databases
        db_setup.test_databases()
        
    except Exception as e:
        print(f"Error during database setup: {e}")
    finally:
        db_setup.close_connections()
    
    print("Database setup completed!")

if __name__ == "__main__":
    main()
