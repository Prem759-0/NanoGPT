"""
Moon AI — Teacher Token Usage & Cost Tracker
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT

Tracks API token usage, latencies, failure rates, and estimated USD compute expenditures.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

USAGE_LOG_PATH = os.path.join(os.path.dirname(__file__), "usage_log.json")

# Default pricing rates in USD per 1,000 tokens
DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    "kimi-k3.1-pro": {"prompt_per_1k": 0.002, "completion_per_1k": 0.008},
    "mock-teacher-v1": {"prompt_per_1k": 0.0, "completion_per_1k": 0.0},
}

class TokenCostTracker:
    """Tracks token consumption, estimated financial cost, and error metrics across teacher runs."""

    def __init__(self, log_path: str = USAGE_LOG_PATH):
        self.log_path = log_path
        self.session_data: Dict[str, Any] = {
            "session_start": datetime.now(timezone.utc).isoformat(),
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "records": []
        }

    def track_call(
        self,
        provider: str,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        is_success: bool = True,
        latency_ms: float = 0.0,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Records a single teacher model interaction."""
        self.session_data["total_calls"] += 1
        if is_success:
            self.session_data["successful_calls"] += 1
        else:
            self.session_data["failed_calls"] += 1

        self.session_data["total_prompt_tokens"] += prompt_tokens
        self.session_data["total_completion_tokens"] += completion_tokens
        self.session_data["total_tokens"] += (prompt_tokens + completion_tokens)

        # Estimate cost
        pricing = DEFAULT_PRICING.get(model_id, {"prompt_per_1k": 0.002, "completion_per_1k": 0.008})
        cost = (prompt_tokens / 1000.0) * pricing["prompt_per_1k"] + (completion_tokens / 1000.0) * pricing["completion_per_1k"]
        self.session_data["estimated_cost_usd"] += cost

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model_id": model_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": round(cost, 6),
            "is_success": is_success,
            "latency_ms": latency_ms,
            "error": error
        }
        self.session_data["records"].append(record)
        return record

    def save_log(self):
        """Persists session usage metrics to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.session_data, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Returns consolidated session summary."""
        return {
            "total_calls": self.session_data["total_calls"],
            "successful_calls": self.session_data["successful_calls"],
            "failed_calls": self.session_data["failed_calls"],
            "total_prompt_tokens": self.session_data["total_prompt_tokens"],
            "total_completion_tokens": self.session_data["total_completion_tokens"],
            "total_tokens": self.session_data["total_tokens"],
            "estimated_cost_usd": round(self.session_data["estimated_cost_usd"], 4)
        }

