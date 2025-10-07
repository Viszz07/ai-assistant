import random
import datetime
from pathlib import Path

class LogGenerator:
    """
    Generates realistic log files for testing and analysis.
    Creates logs with format: YYYY-MM-DD-HH:MM:SS filename line_number SEVERITY message
    """
    
    def __init__(self):
        # Severity levels with weights to ensure at least 20% errors/warnings
        self.severities = {
            'ERROR': 0.15,
            'WARN': 0.10,
            'INFO': 0.50,
            'DEBUG': 0.25
        }
        
        # Sample filenames for different components
        self.filenames = [
            'PUSHDATA.java', 'DATABASE.py', 'UserService.java', 'AuthController.py',
            'PaymentProcessor.java', 'EmailService.py', 'CacheManager.java',
            'SecurityFilter.py', 'DataValidator.java', 'ConfigLoader.py',
            'LoggingService.java', 'MetricsCollector.py', 'HealthChecker.java',
            'BackupService.py', 'NotificationService.java', 'ApiGateway.py'
        ]
        
        # Error messages by severity
        self.messages = {
            'ERROR': [
                'failure to add data in DB',
                'null pointer exception in user authentication',
                'database connection failed',
                'payment processing failed - invalid card',
                'file not found exception',
                'memory allocation failed',
                'API endpoint returned 500 error',
                'authentication token expired',
                'data validation failed - missing required field',
                'network timeout during external service call',
                'failed to parse JSON response',
                'database transaction rollback',
                'insufficient permissions for operation',
                'cache invalidation failed',
                'email delivery failed - SMTP error'
            ],
            'WARN': [
                'connection timeout',
                'deprecated API method used',
                'high memory usage detected',
                'slow query performance detected',
                'retry attempt 3 of 5',
                'configuration parameter missing, using default',
                'SSL certificate expires in 30 days',
                'disk space usage above 80%',
                'unusual login pattern detected',
                'rate limit approaching threshold',
                'backup process taking longer than expected',
                'external service response time degraded',
                'cache hit ratio below optimal level'
            ],
            'INFO': [
                'user successfully authenticated',
                'database connection established',
                'payment processed successfully',
                'email sent to user',
                'cache refreshed successfully',
                'backup completed successfully',
                'configuration loaded from file',
                'service started on port 8080',
                'user session created',
                'data synchronization completed',
                'health check passed',
                'metrics collected and stored',
                'notification sent successfully',
                'API request processed',
                'file upload completed',
                'user logout successful',
                'scheduled task executed'
            ],
            'DEBUG': [
                'entering method processPayment',
                'variable userId set to 12345',
                'SQL query: SELECT * FROM users WHERE id = ?',
                'HTTP request received: GET /api/users',
                'cache lookup for key user_12345',
                'validation rules applied to input data',
                'response serialized to JSON',
                'thread pool size: 10 active, 5 idle',
                'memory usage: 512MB allocated, 256MB used',
                'database query execution time: 45ms',
                'external API call initiated',
                'session token generated',
                'configuration parameter loaded: timeout=30s',
                'file system operation: read permissions checked'
            ]
        }
    
    def generate_weighted_severity(self):
        """Generate severity level based on weighted distribution"""
        rand = random.random()
        cumulative = 0
        for severity, weight in self.severities.items():
            cumulative += weight
            if rand <= cumulative:
                return severity
        return 'INFO'  # fallback
    
    def generate_log_entry(self, base_time):
        """Generate a single log entry"""
        # Random time offset (0-60 seconds from base_time)
        time_offset = random.randint(0, 60)
        timestamp = base_time + datetime.timedelta(seconds=time_offset)
        
        # Format timestamp
        timestamp_str = timestamp.strftime('%Y-%m-%d-%H:%M:%S')
        
        # Random filename and line number
        filename = random.choice(self.filenames)
        line_number = random.randint(1, 500)
        
        # Generate severity and corresponding message
        severity = self.generate_weighted_severity()
        message = random.choice(self.messages[severity])
        
        return f"{timestamp_str} {filename} {line_number} {severity} {message}"
    
    def generate_log_file(self, filename, num_entries=50):
        """Generate a complete log file with specified number of entries"""
        # Start time (recent past)
        base_time = datetime.datetime.now() - datetime.timedelta(hours=24)
        
        log_entries = []
        current_time = base_time
        
        for i in range(num_entries):
            # Increment time by 1-5 minutes between entries
            time_increment = random.randint(60, 300)  # 1-5 minutes in seconds
            current_time += datetime.timedelta(seconds=time_increment)
            
            entry = self.generate_log_entry(current_time)
            log_entries.append(entry)
        
        # Write to file
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for entry in log_entries:
                f.write(entry + '\n')
        
        print(f"Generated {num_entries} log entries in {filename}")
        
        # Print severity distribution for verification
        severity_counts = {}
        for entry in log_entries:
            parts = entry.split()
            if len(parts) >= 4:
                severity = parts[3]
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        print(f"Severity distribution for {filename}:")
        for severity, count in severity_counts.items():
            percentage = (count / num_entries) * 100
            print(f"  {severity}: {count} ({percentage:.1f}%)")
        print()
    
    def generate_all_logs(self):
        """Generate both log files"""
        print("Starting log generation...")
        
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Generate two log files
        self.generate_log_file("logs/application_service.log", 50)
        self.generate_log_file("logs/database_service.log", 50)
        
        print("Log generation completed!")

def main():
    """Main function to run the log generator"""
    generator = LogGenerator()
    generator.generate_all_logs()

if __name__ == "__main__":
    main()
