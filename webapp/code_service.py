import sys
import time
import subprocess
import ast
from typing import Dict, Any, List

class SecurityException(Exception):
    """Raised when code violates sandbox security policies."""
    pass

class CodeService:
    """
    Subprocess-based sandbox for running Python code safely with timeout and AST security inspection.
    Similar to ChatGPT Advanced Data Analysis / Code Interpreter.
    """

    BLOCKED_MODULES = {"shutil", "socket", "ctypes", "winreg"}
    BLOCKED_CALLS = {"system", "rmtree", "kill", "remove", "unlink"}

    def __init__(self, python_executable: str = sys.executable, timeout_seconds: float = 10.0):
        self.python_executable = python_executable
        self.timeout_seconds = timeout_seconds

    def check_security(self, code: str) -> None:
        """Inspects code AST to detect dangerous host operating system calls."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return  # Allow syntax error to be caught naturally by Python execution

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.BLOCKED_MODULES:
                        raise SecurityException(f"Importing '{alias.name}' is prohibited in the local sandbox.")
            elif isinstance(node, ast.ImportFrom):
                if node.module in self.BLOCKED_MODULES:
                    raise SecurityException(f"Importing from '{node.module}' is prohibited in the local sandbox.")

            # Check function calls
            elif isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in self.BLOCKED_CALLS:
                    raise SecurityException(f"Executing system call '{func_name}()' is prohibited in the local sandbox.")

    def execute_code(self, code: str) -> Dict[str, Any]:
        t0 = time.time()
        # 1. Static AST Security Audit
        try:
            self.check_security(code)
        except SecurityException as se:
            return {
                "success": False,
                "output": "",
                "error": f"🔒 Sandbox Security Rejection: {str(se)}",
                "exit_code": -2,
                "execution_time_ms": 0.0
            }

        # 2. Subprocess Execution with Timeout
        try:
            process = subprocess.run(
                [self.python_executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            elapsed_ms = (time.time() - t0) * 1000
            
            output = process.stdout
            error = process.stderr if process.returncode != 0 else None
            
            return {
                "success": process.returncode == 0,
                "output": output if output else ("(No output returned)" if not error else ""),
                "error": error,
                "exit_code": process.returncode,
                "execution_time_ms": round(elapsed_ms, 2)
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Execution timed out after {self.timeout_seconds} seconds.",
                "exit_code": -1,
                "execution_time_ms": self.timeout_seconds * 1000
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "exit_code": -1,
                "execution_time_ms": 0.0
            }
