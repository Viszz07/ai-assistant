"""
Script to generate sample microservice logs with realistic stack traces.
Based on the existing 4G and 5G multimodule assurance logs.
"""

import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from stack_trace_generator import StackTraceGenerator

class MicroserviceLogGenerator:
    """Generate microservice logs with stack traces for testing"""
    
    def __init__(self):
        self.stack_trace_gen = StackTraceGenerator()
        
        # Sample log messages from actual 4G/5G logs
        self.log_templates = {
            'ERROR': [
                ('ALARM: CPU utilization sustained >90%', 'resource_exhaustion'),
                ('High memory utilization (85%) observed in metrics collector', 'OOM_kill'),
                ('Deployment delay due to image pull rate limit', 'timeout_exceeded'),
                ('Data ingestion lag beyond 30s threshold', 'timeout_exceeded'),
                ('VIM node reports insufficient resources for pod scheduling', 'node_unavailable'),
                ('Activation timeout detected in VNFM client', 'timeout_exceeded'),
                ('ALARM: packet_loss=0.6% detected on fronthaul', 'network_jitter'),
                ('Resource allocation requested from VIM cluster', 'resource_exhaustion'),
                ('NWDAF event trigger delay detected', 'timeout_exceeded'),
                ('Persisting incident data to assurance DB', 'node_unavailable'),
                ('ALARM: latency exceeded 40ms SLA limit', 'network_jitter'),
                ('metrics collected: latency=25ms throughput=210Mbps packet_loss=0.02%', 'network_jitter'),
            ],
            'WARN': [
                ('System stability confirmed, SLA compliance 99.5%', 'timeout_exceeded'),
                ('VNF/CNF registration completed successfully', 'resource_exhaustion'),
                ('Loaded configuration and SLA templates', 'network_jitter'),
                ('Deploying assurance VNFs/CNFs via orchestrator', 'resource_exhaustion'),
                ('Activating assurance components for data collection', 'network_jitter'),
                ('ALARM: CPU utilization sustained >90%', 'node_unavailable'),
                ('Starting network assurance bootstrap sequence', 'node_unavailable'),
                ('Audit event generated for incident closure', 'network_jitter'),
                ('NWDAF event trigger delay detected', 'resource_exhaustion'),
                ('High memory utilization (85%) observed in metrics collector', 'network_jitter'),
                ('ALARM: packet_loss=0.6% detected on fronthaul', 'timeout_exceeded'),
                ('Persisting incident data to assurance DB', 'OOM_kill'),
            ],
            'INFO': [
                'Starting network assurance bootstrap sequence',
                'Restart initiated for failed pod instances',
                'Deployment delay due to image pull rate limit',
                'VNF/CNF registration completed successfully',
                'Activating assurance components for data collection',
                'Policy action executed to adjust resource limits',
                'VNFM connectivity established successfully',
                'Loaded configuration and SLA templates',
                'Deploying assurance VNFs/CNFs via orchestrator',
                'Audit event generated for incident closure',
                'VIM node reports insufficient resources for pod scheduling',
                'Resource allocation requested from VIM cluster',
                'Data ingestion lag beyond 30s threshold',
                'Persisting incident data to assurance DB',
                'System stability confirmed, SLA compliance 99.5%',
                'Anomaly score raised to 0.68 due to jitter increase',
                'metrics collected: latency=25ms throughput=210Mbps packet_loss=0.02%',
                'ALARM: latency exceeded 40ms SLA limit',
                'NWDAF event trigger delay detected',
                'Service discovery via Consul completed',
                'System recovered, performance within SLA',
                'ALARM: resource pool near exhaustion',
            ],
            'DEBUG': [
                'Activation timeout detected in VNFM client',
                'ALARM: CPU utilization sustained >90%',
                'VIM node reports insufficient resources for pod scheduling',
                'Archived log and metric snapshots',
                'Resource allocation requested from VIM cluster',
                'ALARM: packet_loss=0.6% detected on fronthaul',
                'Service discovery via Consul completed',
                'ALARM: latency exceeded 40ms SLA limit',
                'Starting network assurance bootstrap sequence',
                'VNFM connectivity established successfully',
                'VNF/CNF registration completed successfully',
                'Anomaly score raised to 0.68 due to jitter increase',
                'metrics collected: latency=25ms throughput=210Mbps packet_loss=0.02%',
                'Policy action executed to adjust resource limits',
                'Persisting incident data to assurance DB',
                'Loaded configuration and SLA templates',
                'High memory utilization (85%) observed in metrics collector',
                'VIM node reports insufficient resources for pod scheduling',
                'Activating assurance components for data collection',
            ]
        }
        
        # Microservice modules (matching the actual logs)
        self.modules_4g = [
            'main.py', 'config_loader.py', 'report_generator.py', 'audit_logger.py',
            'metrics_collector.py', 'vnfm_client.py', 'alarm_manager.py', 'policy_enforcer.py',
            'resource_manager.py', 'remediation_engine.py'
        ]
        
        self.modules_5g = [
            'main.py', 'data_ingestor.py', 'report_generator.py', 'remediation_engine.py',
            'vnfm_client.py', 'config_loader.py', 'vim_connector.py', 'alarm_manager.py',
            'anomaly_detector.py', 'metrics_collector.py'
        ]
        
        # Some modules are Java-based (30% chance)
        self.java_modules = ['vnfm_client', 'vim_connector', 'alarm_manager', 'metrics_collector']
        
        # Stack trace probability (30% of ERROR logs will have stack traces)
        self.stack_trace_probability = 0.30
    
    def generate_log_entry(self, base_time, network_type='4g'):
        """Generate a single log entry"""
        # Random time offset
        time_offset = random.randint(1, 300)  # 1-5 minutes
        timestamp = base_time + timedelta(seconds=time_offset)
        timestamp_str = timestamp.strftime('%Y-%m-%d-%H:%M:%S')
        
        # Select module
        modules = self.modules_4g if network_type == '4g' else self.modules_5g
        module = random.choice(modules)
        
        # Randomly convert some modules to Java
        if module.replace('.py', '') in self.java_modules and random.random() < 0.3:
            module = module.replace('.py', '.java')
        
        line_number = random.randint(1, 500)
        
        # Select severity with weighted distribution
        severity_weights = {'ERROR': 0.20, 'WARN': 0.15, 'INFO': 0.45, 'DEBUG': 0.20}
        severity = random.choices(list(severity_weights.keys()), 
                                 weights=list(severity_weights.values()))[0]
        
        # Select message
        if severity in ['ERROR', 'WARN'] and isinstance(self.log_templates[severity][0], tuple):
            message, root_cause = random.choice(self.log_templates[severity])
            message_with_cause = f"{message} | root_cause={root_cause}; suggested_action={self._get_suggested_action(root_cause)}"
        else:
            if isinstance(self.log_templates[severity][0], tuple):
                message, _ = random.choice(self.log_templates[severity])
            else:
                message = random.choice(self.log_templates[severity])
            message_with_cause = message
        
        # Build log entry
        log_entry = f"{timestamp_str} {module} {line_number} {severity} {message_with_cause}"
        
        # Add stack trace for some ERROR logs
        if severity == 'ERROR' and random.random() < self.stack_trace_probability:
            stack_trace = self.stack_trace_gen.generate_stack_trace(module, message_with_cause, severity)
            if stack_trace:
                log_entry += "\n" + stack_trace
        
        return log_entry, timestamp
    
    def _get_suggested_action(self, root_cause):
        """Get suggested action based on root cause"""
        actions = {
            'OOM_kill': 'increase CPU/memory limits',
            'timeout_exceeded': 'inspect network path',
            'network_jitter': 'analyze VIM node utilization',
            'resource_exhaustion': 'restart failed component',
            'node_unavailable': 'validate VNFM health'
        }
        return actions.get(root_cause, 'investigate issue')
    
    def generate_log_file(self, filename, num_entries=100, network_type='4g'):
        """Generate a complete log file with stack traces"""
        print(f"Generating {filename} with {num_entries} entries...")
        
        # Start time (recent past)
        base_time = datetime.now() - timedelta(hours=24)
        
        log_entries = []
        current_time = base_time
        
        for i in range(num_entries):
            entry, current_time = self.generate_log_entry(current_time, network_type)
            log_entries.append(entry)
        
        # Write to file
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for entry in log_entries:
                f.write(entry + '\n')
        
        print(f"Generated {filename} with {num_entries} entries")
        
        # Count severity distribution
        severity_counts = {'ERROR': 0, 'WARN': 0, 'INFO': 0, 'DEBUG': 0}
        stack_trace_count = 0
        
        for entry in log_entries:
            for severity in severity_counts.keys():
                if f' {severity} ' in entry:
                    severity_counts[severity] += 1
                    break
            if 'Exception' in entry or 'Error:' in entry or 'Traceback' in entry:
                stack_trace_count += 1
        
        print(f"Severity distribution:")
        for severity, count in severity_counts.items():
            percentage = (count / num_entries) * 100
            print(f"  {severity}: {count} ({percentage:.1f}%)")
        print(f"  Entries with stack traces: {stack_trace_count}")
        print()
    
    def generate_all_logs(self):
        """Generate all log files with stack traces"""
        print("Starting log generation with stack traces...")
        print("="*60)
        
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Generate 4G and 5G logs with stack traces
        self.generate_log_file("logs/4g_multimodule_assurance_v2.log", 120, '4g')
        self.generate_log_file("logs/5g_multimodule_assurance_v2.log", 120, '5g')
        
        print("="*60)
        print("Log generation completed!")
        print("\nYou can now run db_setup.py to process these logs.")


def main():
    """Main function"""
    generator = MicroserviceLogGenerator()
    generator.generate_all_logs()


if __name__ == "__main__":
    main()
