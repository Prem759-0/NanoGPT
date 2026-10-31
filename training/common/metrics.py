"""
training/common/metrics.py
Training Metrics, Loss Logging, and Resource Tracking for Moon AI.
"""
import time
from typing import Dict, Any, List

class MetricTracker:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.history: Dict[str, List[float]] = {}
        self.start_time = time.time()

    def update(self, key: str, value: float):
        if key not in self.history:
            self.history[key] = []
        self.history[key].append(value)

    def get_recent_avg(self, key: str) -> float:
        if key not in self.history or not self.history[key]:
            return 0.0
        recent = self.history[key][-self.window_size:]
        return round(sum(recent) / len(recent), 5)

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "elapsed_seconds": round(time.time() - self.start_time, 2)
        }
        for k in self.history:
            summary[f"{k}_avg"] = self.get_recent_avg(k)
            summary[f"{k}_latest"] = round(self.history[k][-1], 5)
        return summary
