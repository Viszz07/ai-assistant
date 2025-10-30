import random
from typing import Dict, List

class StackTraceGenerator:
    """
    Generates realistic stack traces for microservice logs based on error types.
    Supports Java, Python, and JavaScript/Node.js microservices.
    """
    
    def __init__(self):
        # Microservice-specific modules and components
        self.microservice_modules = {
            'java': [
                'vnfm_client', 'vim_connector', 'alarm_manager', 'metrics_collector',
                'config_loader', 'policy_enforcer', 'remediation_engine', 'report_generator',
                'audit_logger', 'resource_manager', 'data_ingestor', 'anomaly_detector'
            ],
            'python': [
                'vnfm_client', 'vim_connector', 'alarm_manager', 'metrics_collector',
                'config_loader', 'policy_enforcer', 'remediation_engine', 'report_generator',
                'audit_logger', 'resource_manager', 'data_ingestor', 'anomaly_detector', 'main'
            ]
        }
        
        # Exception mappings based on root causes
        self.root_cause_exceptions = {
            'OOM_kill': {
                'java': ('OutOfMemoryError', 'Java heap space'),
                'python': ('MemoryError', 'Unable to allocate memory for operation')
            },
            'timeout_exceeded': {
                'java': ('TimeoutException', 'Request timed out after 30000ms'),
                'python': ('TimeoutError', 'Operation timed out after 30 seconds')
            },
            'network_jitter': {
                'java': ('IOException', 'Network connection unstable: packet loss detected'),
                'python': ('ConnectionError', 'Network jitter caused connection instability')
            },
            'resource_exhaustion': {
                'java': ('ResourceExhaustedException', 'CPU/Memory resources exhausted'),
                'python': ('RuntimeError', 'System resources exhausted')
            },
            'node_unavailable': {
                'java': ('NodeUnavailableException', 'VIM node not responding'),
                'python': ('ConnectionRefusedError', 'Unable to connect to VIM node')
            }
        }
        
        # Method/function names for different operations
        self.operation_methods = {
            'java': {
                'vnfm_client': ['connectToVNFM', 'registerVNF', 'activateVNF', 'monitorHealth'],
                'vim_connector': ['allocateResources', 'connectToVIM', 'queryNodeStatus'],
                'alarm_manager': ['processAlarm', 'triggerAlert', 'evaluateThreshold'],
                'metrics_collector': ['collectMetrics', 'aggregateData', 'storeMetrics'],
                'config_loader': ['loadConfiguration', 'validateConfig', 'applySettings'],
                'policy_enforcer': ['enforcePolicy', 'evaluateRules', 'executeAction'],
                'remediation_engine': ['analyzeIncident', 'executeRemediation', 'validateRecovery'],
                'data_ingestor': ['ingestData', 'processStream', 'validateInput']
            },
            'python': {
                'vnfm_client': ['connect_to_vnfm', 'register_vnf', 'activate_vnf', 'monitor_health'],
                'vim_connector': ['allocate_resources', 'connect_to_vim', 'query_node_status'],
                'alarm_manager': ['process_alarm', 'trigger_alert', 'evaluate_threshold'],
                'metrics_collector': ['collect_metrics', 'aggregate_data', 'store_metrics'],
                'config_loader': ['load_configuration', 'validate_config', 'apply_settings'],
                'policy_enforcer': ['enforce_policy', 'evaluate_rules', 'execute_action'],
                'remediation_engine': ['analyze_incident', 'execute_remediation', 'validate_recovery'],
                'data_ingestor': ['ingest_data', 'process_stream', 'validate_input']
            }
        }
    
    def should_generate_stack_trace(self, severity: str, message: str) -> bool:
        """Determine if a log entry should have a stack trace"""
        # Only ERROR logs get stack traces
        if severity != 'ERROR':
            return False
        
        # Check if message indicates an exception scenario
        exception_keywords = [
            'failed', 'exception', 'error', 'timeout', 'unavailable',
            'exhausted', 'exceeded', 'OOM', 'kill', 'crash'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in exception_keywords)
    
    def extract_root_cause(self, message: str) -> str:
        """Extract root cause from log message"""
        # Parse root_cause from message format: "... | root_cause=XXX; ..."
        match = re.search(r'root_cause=([^;]+)', message)
        if match:
            return match.group(1)
        return 'unknown'
    
    def generate_stack_trace(self, filename: str, message: str, severity: str) -> str:
        """Generate appropriate stack trace based on file and message"""
        if not self.should_generate_stack_trace(severity, message):
            return ""
        
        # Determine language from filename
        ext = filename.split('.')[-1].lower()
        if ext == 'java':
            return self.generate_java_stack_trace(filename, message)
        elif ext == 'py':
            return self.generate_python_stack_trace(filename, message)
        
        return ""
    
    def generate_java_stack_trace(self, filename: str, message: str) -> str:
        """Generate realistic Java stack trace for microservice"""
        import re
        
        # Extract root cause if present
        root_cause = 'unknown'
        match = re.search(r'root_cause=([^;]+)', message)
        if match:
            root_cause = match.group(1)
        
        # Get exception type and message based on root cause
        exception_info = self.root_cause_exceptions.get(root_cause, {}).get('java')
        if not exception_info:
            exception_info = ('RuntimeException', 'An error occurred during operation')
        
        exception_type, exception_msg = exception_info
        
        # Determine module from filename
        module = filename.split('.')[0].lower()
        
        # Build stack trace
        stack_lines = [f"java.lang.{exception_type}: {exception_msg}"]
        
        # Add relevant method calls (3-5 frames)
        methods = self.operation_methods['java'].get(module, ['processRequest', 'handleOperation'])
        num_frames = random.randint(3, 5)
        
        for i in range(num_frames):
            method = random.choice(methods)
            line_num = random.randint(45, 350)
            class_name = ''.join(word.capitalize() for word in module.split('_'))
            stack_lines.append(f"    at com.assurance.{module}.{class_name}.{method}({filename}:{line_num})")
        
        # Add common framework calls
        stack_lines.append(f"    at com.assurance.core.ServiceExecutor.execute(ServiceExecutor.java:{random.randint(100, 200)})")
        stack_lines.append(f"    at com.assurance.core.RequestHandler.handle(RequestHandler.java:{random.randint(50, 150)})")
        
        # Sometimes add a "Caused by" for certain root causes
        if root_cause in ['timeout_exceeded', 'node_unavailable', 'network_jitter']:
            caused_by_exceptions = {
                'timeout_exceeded': ('SocketTimeoutException', 'Read timed out'),
                'node_unavailable': ('ConnectException', 'Connection refused'),
                'network_jitter': ('IOException', 'Connection reset by peer')
            }
            caused_type, caused_msg = caused_by_exceptions.get(root_cause, ('IOException', 'I/O error'))
            stack_lines.append(f"Caused by: java.io.{caused_type}: {caused_msg}")
            stack_lines.append(f"    at com.assurance.network.NetworkClient.connect(NetworkClient.java:{random.randint(80, 150)})")
        
        return '\n'.join(stack_lines)
    
    def generate_python_stack_trace(self, filename: str, message: str) -> str:
        """Generate realistic Python stack trace for microservice"""
        import re
        
        # Extract root cause if present
        root_cause = 'unknown'
        match = re.search(r'root_cause=([^;]+)', message)
        if match:
            root_cause = match.group(1)
        
        # Get exception type and message based on root cause
        exception_info = self.root_cause_exceptions.get(root_cause, {}).get('python')
        if not exception_info:
            exception_info = ('RuntimeError', 'An error occurred during operation')
        
        exception_type, exception_msg = exception_info
        
        # Determine module from filename
        module = filename.split('.')[0]
        
        # Build stack trace
        stack_lines = ["Traceback (most recent call last):"]
        
        # Add relevant function calls (2-4 frames)
        functions = self.operation_methods['python'].get(module, ['process_request', 'handle_operation'])
        num_frames = random.randint(2, 4)
        
        for i in range(num_frames):
            function = random.choice(functions)
            line_num = random.randint(45, 350)
            stack_lines.append(f'  File "/app/{module}.py", line {line_num}, in {function}')
            
            # Add code snippet
            code_snippets = {
                'OOM_kill': '    data = process_large_dataset(metrics)',
                'timeout_exceeded': '    response = requests.get(url, timeout=30)',
                'network_jitter': '    connection.send(data)',
                'resource_exhaustion': '    result = allocate_resources(requirements)',
                'node_unavailable': '    node_status = vim_client.get_node_status()'
            }
            code = code_snippets.get(root_cause, '    result = execute_operation()')
            stack_lines.append(code)
        
        # Add framework/library calls
        stack_lines.append(f'  File "/app/core/service_executor.py", line {random.randint(80, 150)}, in execute')
        stack_lines.append('    return handler.process(request)')
        
        # Add the exception line
        stack_lines.append(f"{exception_type}: {exception_msg}")
        
        return '\n'.join(stack_lines)


if __name__ == "__main__":
    # Test the stack trace generator
    import re
    generator = StackTraceGenerator()
    
    # Test with sample log messages
    test_cases = [
        ("vnfm_client.java", "VNFM connectivity failed | root_cause=timeout_exceeded; suggested_action=restart", "ERROR"),
        ("metrics_collector.py", "High memory utilization (85%) | root_cause=OOM_kill; suggested_action=increase limits", "ERROR"),
        ("vim_connector.py", "VIM node unavailable | root_cause=node_unavailable; suggested_action=check node", "ERROR"),
    ]
    
    for filename, message, severity in test_cases:
        print(f"\n{'='*60}")
        print(f"File: {filename}")
        print(f"Message: {message}")
        print(f"{'='*60}")
        trace = generator.generate_stack_trace(filename, message, severity)
        if trace:
            print(trace)
        else:
            print("No stack trace generated")
