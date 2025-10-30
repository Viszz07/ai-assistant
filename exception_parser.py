import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ExceptionInfo:
    """Structured information about an exception"""
    exception_type: str
    exception_message: str
    root_cause: str
    language: str
    affected_file: str
    affected_line: int
    full_stack_trace: str
    summary: str
    severity: str

class ExceptionParser:
    """
    Parses stack traces and exceptions from various programming languages
    and extracts meaningful information for microservice analysis.
    """
    
    def __init__(self):
        # Exception patterns for different languages
        self.language_patterns = {
            'java': {
                'exception_line': r'^([a-zA-Z0-9_.]+Exception|Error):\s*(.+)$',
                'stack_line': r'^\s*at\s+([a-zA-Z0-9_.$<>]+)\(([^:]+):(\d+)\)',
                'caused_by': r'^Caused by:\s+([a-zA-Z0-9_.]+Exception|Error):\s*(.+)$'
            },
            'python': {
                'exception_line': r'^([a-zA-Z0-9_]+(?:Error|Exception|Warning)):\s*(.+)$',
                'stack_line': r'^\s*File\s+"([^"]+)",\s+line\s+(\d+),\s+in\s+([a-zA-Z0-9_<>]+)',
                'traceback_start': r'^Traceback \(most recent call last\):'
            },
            'javascript': {
                'exception_line': r'^([a-zA-Z0-9_]+Error):\s*(.+)$',
                'stack_line': r'^\s*at\s+(?:([a-zA-Z0-9_.$<>]+)\s+)?\(?([^:]+):(\d+):(\d+)\)?',
            },
            'csharp': {
                'exception_line': r'^([a-zA-Z0-9_.]+Exception):\s*(.+)$',
                'stack_line': r'^\s*at\s+([a-zA-Z0-9_.<>]+)\s+in\s+([^:]+):line\s+(\d+)',
            },
            'go': {
                'exception_line': r'^panic:\s*(.+)$',
                'stack_line': r'^\s*([a-zA-Z0-9_./]+)\(.*\)\s+([^:]+):(\d+)',
            }
        }
        
        # Critical exception categories for severity assessment
        self.critical_exceptions = [
            'NullPointerException', 'NullReferenceException', 'SegmentationFault',
            'OutOfMemoryError', 'StackOverflowError', 'DatabaseConnectionError',
            'SecurityException', 'AuthenticationException', 'AuthorizationException',
            'ConnectionError', 'TimeoutError', 'MemoryError'
        ]
        
        self.warning_exceptions = [
            'ValidationException', 'DeprecationWarning', 'ResourceWarning'
        ]
    
    def detect_language(self, stack_trace: str) -> str:
        """Detect programming language from stack trace format"""
        if re.search(r'^\s*at\s+.+\(.+\.java:\d+\)', stack_trace, re.MULTILINE):
            return 'java'
        elif re.search(r'^Traceback \(most recent call last\):', stack_trace, re.MULTILINE):
            return 'python'
        elif re.search(r'^\s*at\s+.+\(.+\.js:\d+:\d+\)', stack_trace, re.MULTILINE):
            return 'javascript'
        elif re.search(r'^\s*at\s+.+in\s+.+:line\s+\d+', stack_trace, re.MULTILINE):
            return 'csharp'
        elif re.search(r'^panic:', stack_trace, re.MULTILINE):
            return 'go'
        return 'unknown'
    
    def parse_java_exception(self, stack_trace: str) -> Optional[ExceptionInfo]:
        """Parse Java exception stack trace"""
        lines = stack_trace.strip().split('\n')
        
        exception_type = None
        exception_message = None
        root_cause_type = None
        root_cause_message = None
        affected_file = None
        affected_line = 0
        
        # Parse exception header
        for line in lines:
            # Check for main exception
            match = re.match(self.language_patterns['java']['exception_line'], line)
            if match and not exception_type:
                exception_type = match.group(1)
                exception_message = match.group(2)
                continue
            
            # Check for caused by (root cause)
            match = re.match(self.language_patterns['java']['caused_by'], line)
            if match:
                root_cause_type = match.group(1)
                root_cause_message = match.group(2)
                continue
            
            # Parse stack trace line to find affected file
            match = re.match(self.language_patterns['java']['stack_line'], line)
            if match and not affected_file:
                method = match.group(1)
                affected_file = match.group(2)
                affected_line = int(match.group(3))
        
        if not exception_type:
            return None
        
        # Determine root cause
        root_cause = root_cause_message if root_cause_message else exception_message
        
        # Generate summary
        summary = self._generate_exception_summary(
            exception_type, exception_message, root_cause_type or exception_type, affected_file
        )
        
        # Determine severity
        severity = self._assess_severity(exception_type)
        
        return ExceptionInfo(
            exception_type=exception_type,
            exception_message=exception_message or "",
            root_cause=root_cause or "",
            language='java',
            affected_file=affected_file or "Unknown",
            affected_line=affected_line,
            full_stack_trace=stack_trace,
            summary=summary,
            severity=severity
        )
    
    def parse_python_exception(self, stack_trace: str) -> Optional[ExceptionInfo]:
        """Parse Python exception stack trace"""
        lines = stack_trace.strip().split('\n')
        
        exception_type = None
        exception_message = None
        affected_file = None
        affected_line = 0
        
        # Python exceptions have the exception at the end
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            match = re.match(self.language_patterns['python']['exception_line'], line)
            if match:
                exception_type = match.group(1)
                exception_message = match.group(2)
                break
        
        # Find the first file in the stack trace (where error originated)
        for line in lines:
            match = re.match(self.language_patterns['python']['stack_line'], line)
            if match:
                affected_file = match.group(1)
                affected_line = int(match.group(2))
                break
        
        if not exception_type:
            return None
        
        # Generate summary
        summary = self._generate_exception_summary(
            exception_type, exception_message, exception_type, affected_file
        )
        
        # Determine severity
        severity = self._assess_severity(exception_type)
        
        return ExceptionInfo(
            exception_type=exception_type,
            exception_message=exception_message or "",
            root_cause=exception_message or "",
            language='python',
            affected_file=affected_file or "Unknown",
            affected_line=affected_line,
            full_stack_trace=stack_trace,
            summary=summary,
            severity=severity
        )
    
    def parse_javascript_exception(self, stack_trace: str) -> Optional[ExceptionInfo]:
        """Parse JavaScript exception stack trace"""
        lines = stack_trace.strip().split('\n')
        
        exception_type = None
        exception_message = None
        affected_file = None
        affected_line = 0
        
        # First line usually contains the exception
        if lines:
            match = re.match(self.language_patterns['javascript']['exception_line'], lines[0])
            if match:
                exception_type = match.group(1)
                exception_message = match.group(2)
        
        # Parse stack trace for file location
        for line in lines[1:]:
            match = re.match(self.language_patterns['javascript']['stack_line'], line)
            if match:
                affected_file = match.group(2) if match.group(2) else match.group(1)
                affected_line = int(match.group(3))
                break
        
        if not exception_type:
            return None
        
        # Generate summary
        summary = self._generate_exception_summary(
            exception_type, exception_message, exception_type, affected_file
        )
        
        # Determine severity
        severity = self._assess_severity(exception_type)
        
        return ExceptionInfo(
            exception_type=exception_type,
            exception_message=exception_message or "",
            root_cause=exception_message or "",
            language='javascript',
            affected_file=affected_file or "Unknown",
            affected_line=affected_line,
            full_stack_trace=stack_trace,
            summary=summary,
            severity=severity
        )
    
    def parse_exception(self, stack_trace: str) -> Optional[ExceptionInfo]:
        """
        Main method to parse exception from any supported language.
        Automatically detects language and uses appropriate parser.
        """
        if not stack_trace or not stack_trace.strip():
            return None
        
        language = self.detect_language(stack_trace)
        
        if language == 'java':
            return self.parse_java_exception(stack_trace)
        elif language == 'python':
            return self.parse_python_exception(stack_trace)
        elif language == 'javascript':
            return self.parse_javascript_exception(stack_trace)
        
        # For unknown languages, try to extract basic info
        return self._parse_generic_exception(stack_trace)
    
    def _parse_generic_exception(self, stack_trace: str) -> Optional[ExceptionInfo]:
        """Fallback parser for unknown exception formats"""
        lines = stack_trace.strip().split('\n')
        if not lines:
            return None
        
        # Try to extract exception type and message from first line
        first_line = lines[0]
        exception_type = "UnknownException"
        exception_message = first_line
        
        # Look for common patterns like "ExceptionType: message"
        match = re.match(r'^([a-zA-Z0-9_]+(?:Exception|Error)):\s*(.+)$', first_line)
        if match:
            exception_type = match.group(1)
            exception_message = match.group(2)
        
        summary = f"{exception_type}: {exception_message[:100]}..."
        
        return ExceptionInfo(
            exception_type=exception_type,
            exception_message=exception_message,
            root_cause=exception_message,
            language='unknown',
            affected_file="Unknown",
            affected_line=0,
            full_stack_trace=stack_trace,
            summary=summary,
            severity='ERROR'
        )
    
    def _generate_exception_summary(self, exception_type: str, exception_message: str, 
                                   root_cause_type: str, affected_file: str) -> str:
        """Generate a concise summary of the exception for display"""
        # Truncate long messages
        message_preview = exception_message[:80] + "..." if len(exception_message) > 80 else exception_message
        
        # Create human-readable summary
        if affected_file and affected_file != "Unknown":
            file_name = affected_file.split('/')[-1].split('\\')[-1]
            summary = f"{exception_type} in {file_name}: {message_preview}"
        else:
            summary = f"{exception_type}: {message_preview}"
        
        return summary
    
    def _assess_severity(self, exception_type: str) -> str:
        """Assess severity based on exception type"""
        if any(critical in exception_type for critical in self.critical_exceptions):
            return 'ERROR'
        elif any(warning in exception_type for warning in self.warning_exceptions):
            return 'WARN'
        else:
            return 'ERROR'  # Default to ERROR for exceptions
    
    def is_stack_trace(self, text: str) -> bool:
        """
        Determine if a text block contains a stack trace.
        Returns True if stack trace patterns are detected.
        """
        if not text or len(text.strip()) < 20:
            return False
        
        # Check for common stack trace indicators
        stack_trace_indicators = [
            r'^\s*at\s+.+\(.+:\d+',  # Java/JavaScript style
            r'^Traceback \(most recent call last\):',  # Python
            r'^\s*File\s+".+",\s+line\s+\d+',  # Python
            r'^panic:',  # Go
            r'Exception|Error',  # General
            r'^\s*at\s+.+in\s+.+:line\s+\d+',  # C#
        ]
        
        for pattern in stack_trace_indicators:
            if re.search(pattern, text, re.MULTILINE):
                return True
        
        return False
    
    def extract_stack_traces_from_log(self, log_content: str) -> List[Tuple[int, str]]:
        """
        Extract all stack traces from a log file.
        Returns list of tuples: (start_line_number, stack_trace_text)
        """
        lines = log_content.split('\n')
        stack_traces = []
        current_trace = []
        trace_start_line = -1
        in_trace = False
        
        for i, line in enumerate(lines, 1):
            # Check if this line starts a stack trace
            if self._is_trace_start(line):
                if current_trace:
                    # Save previous trace
                    stack_traces.append((trace_start_line, '\n'.join(current_trace)))
                current_trace = [line]
                trace_start_line = i
                in_trace = True
            elif in_trace:
                # Check if this line is part of the stack trace
                if self._is_trace_continuation(line):
                    current_trace.append(line)
                else:
                    # End of stack trace
                    stack_traces.append((trace_start_line, '\n'.join(current_trace)))
                    current_trace = []
                    in_trace = False
        
        # Don't forget the last trace
        if current_trace:
            stack_traces.append((trace_start_line, '\n'.join(current_trace)))
        
        return stack_traces
    
    def _is_trace_start(self, line: str) -> bool:
        """Check if a line starts a stack trace"""
        trace_start_patterns = [
            r'Exception|Error',
            r'^Traceback \(most recent call last\):',
            r'^panic:',
        ]
        
        for pattern in trace_start_patterns:
            if re.search(pattern, line):
                return True
        return False
    
    def _is_trace_continuation(self, line: str) -> bool:
        """Check if a line is part of a stack trace"""
        if not line.strip():
            return False
        
        continuation_patterns = [
            r'^\s*at\s+',  # Java/JavaScript/C#
            r'^\s*File\s+"',  # Python
            r'^Caused by:',  # Java
            r'^\s+[a-zA-Z0-9_./]+\(',  # Go
            r'^\s+\^',  # Python error pointer
            r'^\s*\.\.\.',  # Continuation indicator
        ]
        
        for pattern in continuation_patterns:
            if re.search(pattern, line):
                return True
        
        return False


if __name__ == "__main__":
    # Test the exception parser
    parser = ExceptionParser()
    
    # Test Java exception
    java_trace = """java.lang.NullPointerException: Cannot invoke method on null object
    at com.example.UserService.processUser(UserService.java:45)
    at com.example.Controller.handleRequest(Controller.java:123)
Caused by: java.sql.SQLException: Database connection failed
    at com.example.DatabaseManager.connect(DatabaseManager.java:78)"""
    
    print("Testing Java Exception:")
    result = parser.parse_exception(java_trace)
    if result:
        print(f"  Type: {result.exception_type}")
        print(f"  Summary: {result.summary}")
        print(f"  Language: {result.language}")
        print(f"  Severity: {result.severity}")
