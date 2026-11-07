"""
Moon AI — Base Teacher Provider Abstraction
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT

Abstract base class establishing the standard interface for external foundation model teachers
(e.g., Kimi K3.1 Pro, Mock Provider).
Enforces:
- Credential isolation (never hardcoded, never logged).
- Request timeouts and bounded exponential retries.
- Rate limiting and outage handling.
- Token and cost tracking.
- Mockability for unit testing without network dependencies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time
import os
import re
from datetime import datetime, timezone

def redact_api_key(key: Optional[str]) -> str:
    """Safely redacts an API key for logs and diagnostics, exposing only trailing chars."""
    if not key or not isinstance(key, str):
        return "[NOT_SET]"
    clean = key.strip()
    if len(clean) <= 8:
        return "[REDACTED]"
    return f"{clean[:3]}...{clean[-4:]}"


class TeacherProviderError(Exception):
    """Base exception for teacher provider failures."""
    pass

class TeacherRateLimitError(TeacherProviderError):
    """Raised when provider rate limits are exceeded."""
    pass

class TeacherAuthenticationError(TeacherProviderError):
    """Raised when provider authentication fails or credentials are missing."""
    pass

class TeacherTimeoutError(TeacherProviderError):
    """Raised when provider request times out."""
    pass

class TeacherResponseError(TeacherProviderError):
    """Raised when provider returns a malformed response."""
    pass


class BaseTeacherProvider(ABC):
    """
    Standard interface for all Moon AI teacher models.
    Provides uniform method signatures for text completion, structured JSON,
    step-by-step reasoning demonstrations, code synthesis, and tool-use trajectories.
    """

    def __init__(
        self,
        provider_name: str,
        model_id: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        allow_external_calls: bool = False
    ):
        self.provider_name = provider_name
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.allow_external_calls = allow_external_calls
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.failed_calls = 0

    @abstractmethod
    def generate_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generates standard text completion or conversation turn.
        Returns:
            Dict containing 'text', 'model_id', 'usage', 'timestamp', 'metadata'
        """
        pass

    @abstractmethod
    def generate_structured_example(
        self,
        prompt: str,
        schema: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates a validated structured JSON example adhering to schema.
        Returns:
            Dict containing 'data', 'raw_text', 'model_id', 'usage', 'timestamp'
        """
        pass

    @abstractmethod
    def generate_reasoning_demonstration(
        self,
        problem: str,
        domain: str = "mathematics",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates a step-by-step chain-of-thought reasoning trajectory.
        Returns:
            Dict containing 'thought_steps', 'final_answer', 'raw_response', 'model_id'
        """
        pass

    @abstractmethod
    def generate_code_demonstration(
        self,
        task: str,
        language: str = "python",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates production-grade code with explanation, docstrings, and tests.
        Returns:
            Dict containing 'code', 'explanation', 'tests', 'language', 'model_id'
        """
        pass

    @abstractmethod
    def generate_tool_use_demonstration(
        self,
        user_query: str,
        available_tools: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates a structured tool call or multi-step tool execution trajectory.
        Returns:
            Dict containing 'tool_calls', 'thought', 'tool_results', 'model_id'
        """
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Checks connectivity, authentication, and operational status.
        Returns:
            Dict containing 'status', 'provider', 'model_id', 'latency_ms', 'details'
        """
        pass

    @abstractmethod
    def get_model_metadata(self) -> Dict[str, Any]:
        """
        Returns architectural parameters, context window, capabilities, and pricing.
        """
        pass

    def record_usage(self, prompt_tokens: int, completion_tokens: int):
        """Updates internal usage statistics."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_calls += 1

    def get_usage_summary(self) -> Dict[str, Any]:
        """Returns cumulative token metrics."""
        return {
            "provider": self.provider_name,
            "model_id": self.model_id,
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens
        }

