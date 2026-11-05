"""
webapp/code_service.py
Subprocess-based sandbox for running Python code safely with timeout and AST security inspection.
Delegates to the sovereign sandbox engine in sandbox/code_sandbox.py.
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional

from sandbox.code_sandbox import SovereignCodeSandbox, SandboxSecurityError

class SecurityException(Exception):
    """Raised when code violates sandbox security policies."""
    pass

class CodeService:
    """
    Subprocess-based sandbox for running Python code safely with timeout and AST security inspection.
    """

    def __init__(self, python_executable: str = sys.executable, timeout_seconds: float = 15.0):
        self.python_executable = python_executable
        self.timeout_seconds = timeout_seconds
        self.sandbox = SovereignCodeSandbox(
            python_executable=python_executable,
            timeout_seconds=timeout_seconds
        )

    def check_security(self, code: str) -> None:
        """Inspects code AST to detect dangerous host operating system calls."""
        if os.environ.get("MOON_ENABLE_SANDBOX", "1").lower() in ("0", "false", "no"):
            raise SecurityException("Code execution is disabled on this server instance (MOON_ENABLE_SANDBOX=0).")

        try:
            self.sandbox.analyze_ast(code)
        except SandboxSecurityError as se:
            raise SecurityException(str(se))

    def execute_code(self, code: str) -> Dict[str, Any]:
        """Executes Python code safely in the sandbox."""
        if os.environ.get("MOON_ENABLE_SANDBOX", "1").lower() in ("0", "false", "no"):
            return {
                "success": False,
                "output": "",
                "error": "🔒 Sandbox Security Rejection: Code execution is disabled on this server instance (MOON_ENABLE_SANDBOX=0).",
                "exit_code": -2,
                "execution_time_ms": 0.0,
                "artifacts": []
            }

        res = self.sandbox.execute(code, timeout_override=self.timeout_seconds)
        output = res.get("output", "")
        error = res.get("error")

        return {
            "success": res.get("success", False),
            "output": output if output else ("(No output returned)" if not error else ""),
            "error": error,
            "exit_code": res.get("exit_code", 0),
            "execution_time_ms": res.get("duration_ms", 0.0),
            "artifacts": res.get("artifacts", [])
        }
