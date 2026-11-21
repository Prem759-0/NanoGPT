"""
Moon AI — Objective Verification Engine
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Mandatory Objective Verification:
1. Coding Verification: Isolated AST parsing + SovereignCodeSandbox execution in separate process.
2. Mathematical Verification: Deterministic AST-based calculation & numerical tolerance checks.
3. Content Classification:
   - VERIFIED_CODE
   - VERIFIED_MATH
   - VERIFIED_FACTUAL
   - HUMAN_REVIEW_REQUIRED
   - REJECTED
4. Separation of Feedback: Subjective user ratings are NEVER treated as correctness proofs.
   A 5-star rating NEVER overrides a failed verifier.
"""

import ast
import operator
import re
import time
from typing import Dict, Any, List, Optional, Tuple, Union

from sandbox.code_sandbox import SovereignCodeSandbox, SandboxSecurityError

# Verification Classifications
VERIFIED_CODE = "VERIFIED_CODE"
VERIFIED_MATH = "VERIFIED_MATH"
VERIFIED_FACTUAL = "VERIFIED_FACTUAL"
HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
REJECTED = "REJECTED"


class SafeMathEvaluator:
    """
    Evaluates basic arithmetic expressions using Python AST without using eval().
    Supports +, -, *, /, //, %, **, unary +, unary -.
    """

    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    @classmethod
    def evaluate(cls, expr_str: str) -> Tuple[bool, Optional[float], Optional[str]]:
        """
        Safely evaluates an arithmetic expression string.
        Returns (success, result_float, error_message).
        """
        try:
            # Clean string
            cleaned = expr_str.strip().replace("=", "").replace(",", "")
            tree = ast.parse(cleaned, mode="eval")
            val = cls._eval_node(tree.body)
            return True, float(val), None
        except ZeroDivisionError:
            return False, None, "Division by zero"
        except Exception as e:
            return False, None, f"Arithmetic evaluation error: {str(e)}"

    @classmethod
    def _eval_node(cls, node: ast.AST) -> Union[int, float]:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in cls.OPERATORS:
                raise ValueError(f"Unsupported binary operator: {op_type}")
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            if op_type == ast.Pow and (right > 1000 or right < -1000):
                raise ValueError("Exponent too large")
            return cls.OPERATORS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in cls.OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type}")
            operand = cls._eval_node(node.operand)
            return cls.OPERATORS[op_type](operand)

        raise ValueError(f"Unsupported AST node in math expression: {type(node)}")


class ObjectiveVerifier:
    """
    Independently verifies candidate examples for correctness before admitting to candidate storage.
    """

    def __init__(self, sandbox_timeout: float = 5.0):
        self.sandbox_timeout = sandbox_timeout
        self._sandbox: Optional[SovereignCodeSandbox] = None

    @property
    def sandbox(self) -> SovereignCodeSandbox:
        if self._sandbox is None:
            self._sandbox = SovereignCodeSandbox(timeout_seconds=self.sandbox_timeout)
        return self._sandbox

    def verify_code(
        self,
        code: str,
        test_assertions: Optional[str] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates Python code example syntax:
        1. Validates AST structure and syntax without dynamic execution.
        2. Dynamic execution of untrusted donated code is disabled pending
           genuine OS-level isolation (e.g. Windows AppContainer / OCI container).
        3. Never executes untrusted code in an unconfined subprocess.
        """
        details: Dict[str, Any] = {
            "type": "code",
            "syntax_valid": False,
            "sandbox_executed": False,
            "exit_code": None,
            "duration_ms": 0.0,
            "security_status": "UNTRUSTED_CODE_EXECUTION_DISABLED"
        }

        # Step 1: AST Parsing Check
        try:
            ast.parse(code)
            details["syntax_valid"] = True
        except SyntaxError as syn_err:
            details["error"] = f"SyntaxError: {syn_err.msg} at line {syn_err.lineno}"
            details["security_status"] = "SYNTAX_ERROR"
            return False, "SYNTAX_ERROR", details

        # Step 2: Static AST Security Screening
        try:
            self.sandbox.analyze_ast(code)
            details["ast_security"] = "PASSED"
        except SandboxSecurityError as sse:
            details["security_status"] = "REJECTED"
            details["error"] = str(sse)
            return False, f"SECURITY_BLOCK: {str(sse)}", details

        # Step 3: Dynamic Execution Disabled (SEC-01)
        # Windows subprocess + AST filtering does not constitute an OS-level isolation boundary.
        # Dynamic execution of untrusted donated code is blocked pending AppContainer/OCI isolation.
        details["sandbox_executed"] = False
        details["security_status"] = "UNTRUSTED_CODE_EXECUTION_DISABLED"
        details["notice"] = (
            "Dynamic execution of untrusted code is disabled pending OS-level isolation "
            "(Windows AppContainer / Job Object / OCI container). Example queued for human review."
        )
        return False, "UNTRUSTED_CODE_EXECUTION_DISABLED", details

    def verify_math(
        self,
        expression_or_problem: str,
        expected_solution: str,
        tolerance: float = 1e-6
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Independently computes arithmetic / mathematical expressions and asserts numerical equivalence.
        """
        details: Dict[str, Any] = {
            "type": "math",
            "expression": expression_or_problem,
            "expected": expected_solution,
            "calculated": None,
            "tolerance": tolerance,
            "verification_method": "ast_deterministic_eval"
        }

        # Extract numerical expected answer
        num_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", expected_solution.strip())
        if not num_match:
            return False, "UNPARSEABLE_EXPECTED_SOLUTION", details

        expected_val = float(num_match.group(0))
        details["expected_float"] = expected_val

        # Evaluate candidate expression
        ok, calc_val, err = SafeMathEvaluator.evaluate(expression_or_problem)
        if not ok or calc_val is None:
            details["error"] = err
            return False, f"CALCULATION_FAILED: {err}", details

        details["calculated"] = calc_val
        diff = abs(calc_val - expected_val)
        details["absolute_difference"] = diff

        if diff <= tolerance:
            return True, VERIFIED_MATH, details
        else:
            return False, f"NUMERICAL_MISMATCH: expected {expected_val}, got {calc_val}", details

    def classify_and_verify(
        self,
        prompt: str,
        response: str,
        subjective_rating: Optional[float] = None
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Classifies interaction and runs relevant objective verifiers.
        Subjective rating is captured purely as metadata; it NEVER overrides verification.
        """
        verification_metadata: Dict[str, Any] = {
            "timestamp": time.time(),
            "subjective_user_rating": subjective_rating,
            "overridden_by_user_rating": False  # Invariant: Never allow rating to override verifier
        }

        # Check for Python code blocks in response
        code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)\n```", response, re.DOTALL)
        if code_blocks:
            combined_code = "\n\n".join(code_blocks)
            is_valid, reason, code_details = self.verify_code(combined_code)
            verification_metadata.update(code_details)
            if not code_details.get("syntax_valid", False) or reason.startswith("SECURITY_BLOCK"):
                return REJECTED, f"Code verification failed: {reason}", verification_metadata
            # Valid syntax, but dynamic execution is disabled pending OS-level isolation (SEC-01)
            # Never return VERIFIED_CODE; route to human review
            return HUMAN_REVIEW_REQUIRED, (
                "Code syntax verified via AST. Dynamic execution disabled pending OS-level "
                "AppContainer/container isolation. Example queued for human review."
            ), verification_metadata

        # Check for simple arithmetic queries (e.g. "what is 2 + 2", "calculate 15 * 8")
        math_match = re.search(r"(?:what\s+is|calculate|compute|solve|eval(?:uate)?)\s+([\d\s\+\-\*\/\(\)\.\^\%]+)", prompt, re.I)
        if math_match:
            expr = math_match.group(1).strip()
            is_valid, reason, math_details = self.verify_math(expr, response)
            verification_metadata.update(math_details)
            if is_valid:
                return VERIFIED_MATH, "Mathematical calculation verified deterministically.", verification_metadata
            else:
                return REJECTED, f"Math verification failed: {reason}", verification_metadata

        # For general natural-language conversation: cannot deterministically prove ground truth
        verification_metadata["notice"] = "Natural language text requires human triage; cannot be deterministically verified."
        return HUMAN_REVIEW_REQUIRED, "Content queued for human review.", verification_metadata

