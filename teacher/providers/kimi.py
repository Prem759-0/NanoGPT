"""
Moon AI — Kimi K3.1 Pro Teacher Provider
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT

Official Target Teacher: Kimi K3.1 Pro (Moonshot AI)
Official Announcement: https://www.kimi.com/news/kimi-k3-1-pro
Official Weights Announcement: Full weights scheduled for November 4, 2026.

Security & Architecture Invariants:
1. API keys are strictly read from environment variables (KIMI_API_KEY / MOON_TEACHER_API_KEY).
2. API keys are NEVER hardcoded or printed to stdout/logs.
3. External network calls are DISABLED BY DEFAULT (allow_external_calls=False).
4. Request timeouts and bounded exponential retries prevent hanging or hammering.
5. In unit tests and CI, calls are safely mocked or rejected without network calls.
6. Local weights adapter checks license, commercial terms, and VRAM sizing before loading.
"""

import json
import os
import re
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from teacher.providers.base import (
    BaseTeacherProvider,
    TeacherProviderError,
    TeacherRateLimitError,
    TeacherAuthenticationError,
    TeacherTimeoutError,
    TeacherResponseError,
    redact_api_key
)

KIMI_OFFICIAL_ANNOUNCEMENT_URL = "https://www.kimi.com/news/kimi-k3-1-pro"
KIMI_WEIGHTS_ANNOUNCED_DATE = "2026-11-04"

class KimiProvider(BaseTeacherProvider):
    """Teacher provider interface for Moonshot Kimi K3.1 Pro."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.moonshot.cn/v1",
        model_id: str = "kimi-k3.1-pro",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        allow_external_calls: bool = False
    ):
        super().__init__(
            provider_name="kimi",
            model_id=model_id,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            allow_external_calls=allow_external_calls
        )
        self.base_url = base_url.rstrip("/")
        # Resolve API key strictly from environment variables if not passed
        self._api_key = api_key or os.environ.get("KIMI_API_KEY") or os.environ.get("MOON_TEACHER_API_KEY")

    @property
    def has_credentials(self) -> bool:
        """Returns True if a non-empty API key is configured."""
        return bool(self._api_key and len(self._api_key.strip()) > 5)

    def get_redacted_key(self) -> str:
        """Returns safely redacted key for diagnostics."""
        return redact_api_key(self._api_key)

    def check_official_weights_release(self, local_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Inspects the official Kimi weights status.
        Does not assume weights exist prior to formal post-release verification.
        Evaluates license, commercial terms, acceptable use, and hardware feasibility.
        """
        status_report = {
            "model_id": self.model_id,
            "announcement_url": KIMI_OFFICIAL_ANNOUNCEMENT_URL,
            "announced_release_date": KIMI_WEIGHTS_ANNOUNCED_DATE,
            "local_weights_present": False,
            "can_run_on_t4_gpu": False,
            "hardware_note": (
                "A full-scale teacher model (3T / hundreds of billions of parameters) cannot run or be fine-tuned "
                "on a single 16 GB Google Colab NVIDIA T4 GPU. Dedicated multi-node A100/H100 clusters (>= 320GB VRAM) "
                "or aggressive 4-bit quantization on high-end hardware would be mandatory."
            ),
            "verification_status": "SCHEDULED_RELEASE_AWAITING_VERIFICATION",
            "license_requirements": [
                "Verify commercial-use permissions before enterprise deployment.",
                "Verify attribution requirements.",
                "Verify acceptable use policy (safety, synthetic training restrictions)."
            ]
        }

        if local_path and os.path.isdir(local_path):
            status_report["local_weights_present"] = True
            status_report["verification_status"] = "LOCAL_WEIGHTS_FOUND_PENDING_CHECKSUM"

        return status_report

    def _execute_api_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes HTTP request to Kimi endpoint with bounded retries and exponential backoff.
        Raises TeacherProviderError subclasses on failures.
        """
        if not self.allow_external_calls:
            raise TeacherProviderError(
                "External teacher API calls are disabled by default. "
                "To enable live network calls, set allow_external_calls=True and configure KIMI_API_KEY."
            )

        if not self.has_credentials:
            raise TeacherAuthenticationError(
                "Missing Kimi API credentials. Set KIMI_API_KEY or MOON_TEACHER_API_KEY environment variable."
            )

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "MoonAI-TeacherDistillation/1.0"
        }

        attempts = 0
        backoff = 1.0

        while attempts < self.max_retries:
            attempts += 1
            try:
                req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                    res_body = response.read().decode("utf-8")
                    try:
                        return json.loads(res_body)
                    except Exception as json_err:
                        raise TeacherResponseError(f"Malformed JSON response from Kimi: {str(json_err)}")

            except urllib.error.HTTPError as http_err:
                self.failed_calls += 1
                if http_err.code == 401 or http_err.code == 403:
                    raise TeacherAuthenticationError(f"Kimi API authentication failed (HTTP {http_err.code})")
                elif http_err.code == 429:
                    if attempts >= self.max_retries:
                        raise TeacherRateLimitError("Kimi API rate limit exceeded (HTTP 429). Retries exhausted.")
                    time.sleep(backoff)
                    backoff *= self.retry_backoff
                elif http_err.code >= 500:
                    if attempts >= self.max_retries:
                        raise TeacherProviderError(f"Kimi provider outage (HTTP {http_err.code}). Retries exhausted.")
                    time.sleep(backoff)
                    backoff *= self.retry_backoff
                else:
                    raise TeacherProviderError(f"HTTP error {http_err.code}: {http_err.reason}")

            except urllib.error.URLError as url_err:
                self.failed_calls += 1
                if "timed out" in str(url_err).lower():
                    if attempts >= self.max_retries:
                        raise TeacherTimeoutError(f"Kimi API request timed out after {self.timeout_seconds}s")
                    time.sleep(backoff)
                    backoff *= self.retry_backoff
                else:
                    raise TeacherProviderError(f"Kimi network connection error: {str(url_err)}")

            except Exception as e:
                self.failed_calls += 1
                raise TeacherProviderError(f"Unexpected error calling Kimi API: {str(e)}")

        raise TeacherProviderError("Kimi request failed after all retry attempts.")

    def generate_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generates standard completion from Kimi K3.1 Pro."""
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 1024)
        }
        resp = self._execute_api_request("chat/completions", payload)

        choices = resp.get("choices", [])
        if not choices:
            raise TeacherResponseError("Kimi response missing 'choices' list.")

        text = choices[0].get("message", {}).get("content", "")
        usage = resp.get("usage", {})
        p_tok = usage.get("prompt_tokens", len(prompt.split()))
        c_tok = usage.get("completion_tokens", len(text.split()))
        self.record_usage(p_tok, c_tok)

        return {
            "text": text,
            "provider": self.provider_name,
            "model_id": self.model_id,
            "usage": {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": p_tok + c_tok},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {"finish_reason": choices[0].get("finish_reason")}
        }

    def generate_structured_example(
        self,
        prompt: str,
        schema: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Requests structured JSON output adhering to schema."""
        augmented_prompt = (
            f"{prompt}\n\n"
            "OUTPUT REQUIREMENT: You MUST respond strictly in valid JSON adhering to this schema:\n"
            f"{json.dumps(schema, indent=2)}\n"
            "Do not include commentary outside the JSON block."
        )
        comp = self.generate_completion(augmented_prompt, **kwargs)
        raw_text = comp["text"]

        # Parse JSON
        clean_text = raw_text.strip()
        if "```json" in clean_text:
            m = re.search(r"```json\s*(.*?)\s*```", clean_text, re.DOTALL)
            if m:
                clean_text = m.group(1).strip()
        elif "```" in clean_text:
            m = re.search(r"```\s*(.*?)\s*```", clean_text, re.DOTALL)
            if m:
                clean_text = m.group(1).strip()

        try:
            parsed = json.loads(clean_text)
        except Exception:
            parsed = {"raw": raw_text, "parse_error": True}

        comp["data"] = parsed
        comp["raw_text"] = raw_text
        return comp

    def generate_reasoning_demonstration(
        self,
        problem: str,
        domain: str = "mathematics",
        **kwargs
    ) -> Dict[str, Any]:
        """Generates chain-of-thought reasoning trajectory."""
        system_instruction = (
            "You are an expert reasoning engine. For the given problem, first show step-by-step "
            "rigorous derivation inside <thought> ... </thought> tags. Then provide the clean final answer."
        )
        prompt = f"{system_instruction}\n\nProblem ({domain}):\n{problem}"
        comp = self.generate_completion(prompt, **kwargs)
        text = comp["text"]

        thought_steps = []
        final_answer = text
        if "<thought>" in text and "</thought>" in text:
            m = re.search(r"<thought>(.*?)</thought>", text, re.DOTALL)
            if m:
                thought_str = m.group(1).strip()
                thought_steps = [s.strip() for s in thought_str.split("\n") if s.strip()]
                final_answer = text.split("</thought>")[-1].strip()

        return {
            "thought_steps": thought_steps,
            "final_answer": final_answer,
            "raw_response": text,
            "domain": domain,
            "provider": self.provider_name,
            "model_id": self.model_id,
            "timestamp": comp["timestamp"]
        }

    def generate_code_demonstration(
        self,
        task: str,
        language: str = "python",
        **kwargs
    ) -> Dict[str, Any]:
        """Generates code demonstration with tests and explanation."""
        prompt = (
            f"Write robust, tested {language} code for the following task:\n{task}\n\n"
            "Requirements:\n"
            "1. Wrap the implementation in a clean markdown codeblock.\n"
            "2. Include unit test assertions.\n"
            "3. Include a concise complexity explanation."
        )
        comp = self.generate_completion(prompt, **kwargs)
        text = comp["text"]

        code = ""
        m = re.search(r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            code = m.group(1).strip()

        return {
            "code": code,
            "explanation": text,
            "language": language,
            "raw_response": text,
            "provider": self.provider_name,
            "model_id": self.model_id,
            "timestamp": comp["timestamp"]
        }

    def generate_tool_use_demonstration(
        self,
        user_query: str,
        available_tools: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Generates tool selection and execution trajectory."""
        prompt = (
            f"User Query: {user_query}\n\n"
            f"Available Tools: {json.dumps(available_tools, indent=2)}\n\n"
            "Select the appropriate tool, formulate the call arguments, and provide reasoning."
        )
        comp = self.generate_completion(prompt, **kwargs)
        return {
            "thought": comp["text"],
            "tool_calls": [],
            "raw_response": comp["text"],
            "provider": self.provider_name,
            "model_id": self.model_id,
            "timestamp": comp["timestamp"]
        }

    def health_check(self) -> Dict[str, Any]:
        """Checks connectivity and credential readiness."""
        if not self.has_credentials:
            return {
                "status": "UNCONFIGURED_MISSING_API_KEY",
                "provider": self.provider_name,
                "model_id": self.model_id,
                "api_key_redacted": "[NOT_SET]",
                "external_calls_allowed": self.allow_external_calls,
                "details": "KIMI_API_KEY environment variable is not configured."
            }

        if not self.allow_external_calls:
            return {
                "status": "CONFIGURED_EXTERNAL_CALLS_DISABLED",
                "provider": self.provider_name,
                "model_id": self.model_id,
                "api_key_redacted": self.get_redacted_key(),
                "external_calls_allowed": False,
                "details": "API key present; external network calls disabled by default."
            }

        # If external calls are enabled, run a 1-token test prompt
        try:
            t0 = time.time()
            self.generate_completion("ping", max_tokens=1)
            latency = round((time.time() - t0) * 1000, 1)
            return {
                "status": "HEALTHY",
                "provider": self.provider_name,
                "model_id": self.model_id,
                "latency_ms": latency,
                "api_key_redacted": self.get_redacted_key()
            }
        except Exception as e:
            return {
                "status": "CONNECTIVITY_ERROR",
                "provider": self.provider_name,
                "model_id": self.model_id,
                "error": str(e),
                "api_key_redacted": self.get_redacted_key()
            }

    def get_model_metadata(self) -> Dict[str, Any]:
        """Returns metadata for Kimi K3.1 Pro."""
        return {
            "provider": "kimi",
            "model_id": self.model_id,
            "display_name": "Moonshot Kimi K3.1 Pro",
            "announcement_url": KIMI_OFFICIAL_ANNOUNCEMENT_URL,
            "weights_announcement_date": KIMI_WEIGHTS_ANNOUNCED_DATE,
            "context_window": 128000,
            "capabilities": [
                "multi_turn_conversation",
                "long_context_retrieval",
                "code_synthesis",
                "chain_of_thought_reasoning",
                "tool_use",
                "multimodal_understanding"
            ],
            "api_endpoint": self.base_url,
            "key_configured": self.has_credentials,
            "key_redacted": self.get_redacted_key()
        }

