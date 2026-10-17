import sys
import time
import subprocess
from typing import Dict, Any

class CodeService:
    """
    Subprocess-based sandbox for running Python code safely with timeout.
    Similar to ChatGPT Advanced Data Analysis / Code Interpreter.
    """

    def __init__(self, python_executable: str = sys.executable, timeout_seconds: float = 6.0):
        self.python_executable = python_executable
        self.timeout_seconds = timeout_seconds

    def execute_code(self, code: str) -> Dict[str, Any]:
        t0 = time.time()
        try:
            # Run code in an isolated subprocess with strict timeout
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
                "execution_time_ms": 0
            }
