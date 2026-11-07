"""
Moon AI — Deterministic Mock Teacher Provider
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT

Provides a fully offline, deterministic mock provider for CI, unit testing, and pipeline verification.
Can simulate:
- Normal completions across all 5 standard methods.
- Rate limiting errors.
- Timeout errors.
- Malformed response errors.
- Authentication errors.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import json
from teacher.providers.base import (
    BaseTeacherProvider,
    TeacherRateLimitError,
    TeacherTimeoutError,
    TeacherResponseError,
    TeacherAuthenticationError
)

class MockTeacherProvider(BaseTeacherProvider):
    """Deterministic offline teacher provider for unit tests and local pipeline verification."""

    def __init__(
        self,
        model_id: str = "mock-teacher-v1",
        simulate_error: Optional[str] = None,
        custom_response: Optional[str] = None
    ):
        super().__init__(
            provider_name="mock",
            model_id=model_id,
            timeout_seconds=5.0,
            allow_external_calls=True  # Mock doesn't call external networks
        )
        self.simulate_error = simulate_error
        self.custom_response = custom_response
        self.is_mock = True
        self.is_dev_fixture = True
        self.is_synthetic = True

    def _maybe_trigger_error(self):
        if self.simulate_error == "rate_limit":
            self.failed_calls += 1
            raise TeacherRateLimitError("Mock: HTTP 429 Too Many Requests (Rate limit exceeded)")
        elif self.simulate_error == "timeout":
            self.failed_calls += 1
            raise TeacherTimeoutError("Mock: Request timed out after 5.0 seconds")
        elif self.simulate_error == "auth":
            self.failed_calls += 1
            raise TeacherAuthenticationError("Mock: HTTP 401 Unauthorized (Invalid API key)")
        elif self.simulate_error == "malformed":
            self.failed_calls += 1
            raise TeacherResponseError("Mock: Received corrupted non-JSON response payload")

    def generate_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        self._maybe_trigger_error()
        resp_text = self.custom_response or (
            f"Mock teacher demonstration for prompt: '{prompt[:40]}...'. "
            "Follows first-principles reasoning and structured ChatML formatting."
        )
        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, len(resp_text.split()))
        self.record_usage(prompt_tokens, completion_tokens)

        return {
            "text": resp_text,
            "provider": self.provider_name,
            "model_id": self.model_id,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "deterministic": True,
                "mock": True,
                "is_dev_fixture": True,
                "is_synthetic": True,
                "eligible_for_production": False
            }
        }

    def generate_structured_example(
        self,
        prompt: str,
        schema: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        self._maybe_trigger_error()
        data = {
            "category": "education",
            "instruction": prompt,
            "explanation": "Structured explanation following verified schema.",
            "status": "APPROVED",
            "score": 95.0
        }
        raw_text = json.dumps(data)
        self.record_usage(len(prompt.split()), len(raw_text.split()))

        return {
            "data": data,
            "raw_text": raw_text,
            "provider": self.provider_name,
            "model_id": self.model_id,
            "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(raw_text.split())},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {"schema_verified": True}
        }

    def generate_reasoning_demonstration(
        self,
        problem: str,
        domain: str = "mathematics",
        **kwargs
    ) -> Dict[str, Any]:
        self._maybe_trigger_error()
        thought_steps = [
            f"1. Understand the problem statement: '{problem[:50]}...'",
            "2. Identify constraints and mathematical invariants.",
            "3. Execute logical derivation step-by-step.",
            "4. Verify edge cases and consistency."
        ]
        final_answer = "Derived solution satisfies all formal constraints."
        raw_resp = (
            "<thought>\n" + "\n".join(thought_steps) + "\n</thought>\n\n"
            f"Final Answer: {final_answer}"
        )
        self.record_usage(len(problem.split()), len(raw_resp.split()))

        return {
            "thought_steps": thought_steps,
            "final_answer": final_answer,
            "raw_response": raw_resp,
            "domain": domain,
            "provider": self.provider_name,
            "model_id": self.model_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def generate_code_demonstration(
        self,
        task: str,
        language: str = "python",
        **kwargs
    ) -> Dict[str, Any]:
        self._maybe_trigger_error()
        code = (
            "def solution():\n"
            "    \"\"\"Solves the requested task with O(n) complexity.\"\"\"\n"
            "    return True\n\n"
            "def test_solution():\n"
            "    assert solution() is True\n"
        )
        explanation = "Clean idiomatic implementation with comprehensive docstring and assertion test."
        tests = "assert solution() is True"
        raw_text = f"```python\n{code}```\n\n{explanation}"
        self.record_usage(len(task.split()), len(raw_text.split()))

        return {
            "code": code,
            "explanation": explanation,
            "tests": tests,
            "language": language,
            "raw_response": raw_text,
            "provider": self.provider_name,
            "model_id": self.model_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def generate_tool_use_demonstration(
        self,
        user_query: str,
        available_tools: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        self._maybe_trigger_error()
        tool_name = available_tools[0].get("name", "calculator") if available_tools else "calculator"
        tool_calls = [
            {
                "tool": tool_name,
                "arguments": {"query": user_query[:30]}
            }
        ]
        thought = f"User asks '{user_query[:30]}'. Calling {tool_name} to fetch precise data."
        self.record_usage(len(user_query.split()), 50)

        return {
            "thought": thought,
            "tool_calls": tool_calls,
            "provider": self.provider_name,
            "model_id": self.model_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def health_check(self) -> Dict[str, Any]:
        self._maybe_trigger_error()
        return {
            "status": "HEALTHY",
            "provider": self.provider_name,
            "model_id": self.model_id,
            "latency_ms": 0.5,
            "details": "Mock provider operational (zero external network dependency)."
        }

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "provider": "mock",
            "model_id": self.model_id,
            "context_window": 8192,
            "supports_reasoning": True,
            "supports_code": True,
            "supports_tools": True,
            "offline_only": True
        }

