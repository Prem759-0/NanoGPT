"""
training/common/experiment.py
Unified Experiment Tracking and Metrics Logging for Moon AI.
Records run parameters, manifest hashes, step metrics, hardware snapshots, and provenance.
"""
import os
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from .distributed import get_hardware_info, get_vram_usage_mb

class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        model_name: str,
        manifest_path: str,
        manifest_hash: str,
        config: Dict[str, Any],
        output_dir: str,
        run_id: Optional[str] = None
    ):
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.manifest_path = manifest_path
        self.manifest_hash = manifest_hash
        self.config = config
        self.output_dir = output_dir

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id or f"{experiment_name}_{timestamp}"
        self.log_dir = os.path.join(output_dir, "runs", self.run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "experiment_log.json")

        self.hardware_info = get_hardware_info()
        self.start_time = time.time()
        self.step_logs: List[Dict[str, Any]] = []
        self.best_metrics: Dict[str, Any] = {}
        self.provenance = "moon_ai_random_init"
        self.architect = "Prem Gaikwad (India 🇮🇳)"
        self.status = "RUNNING"

        self._flush_metadata()

    def log_step(
        self,
        step: int,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        lr: Optional[float] = None,
        throughput_samples_sec: Optional[float] = None,
        extra_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Appends metrics record for a training step."""
        elapsed = round(time.time() - self.start_time, 2)
        vram_mb = get_vram_usage_mb()

        entry = {
            "step": step,
            "epoch": epoch,
            "train_loss": round(float(train_loss), 6),
            "val_loss": round(float(val_loss), 6) if val_loss is not None else None,
            "learning_rate": lr,
            "throughput_samples_sec": round(throughput_samples_sec, 2) if throughput_samples_sec else None,
            "vram_mb": vram_mb,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        if extra_metrics:
            entry.update(extra_metrics)

        self.step_logs.append(entry)
        if len(self.step_logs) % 10 == 0 or val_loss is not None:
            self._flush_metadata()

    def update_best(self, metric_name: str, value: float, step: int) -> None:
        """Tracks the best metric observed during the run."""
        self.best_metrics[metric_name] = {
            "value": round(float(value), 6),
            "step": step,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        self._flush_metadata()

    def finish(self, status: str = "COMPLETED", checkpoint_path: Optional[str] = None) -> None:
        """Marks experiment as complete and records final artifacts."""
        self.status = status
        self.end_time = time.time()
        self.total_duration_sec = round(self.end_time - self.start_time, 2)
        self.final_checkpoint = checkpoint_path
        self._flush_metadata()

    def _flush_metadata(self) -> None:
        """Writes current experiment state to disk as JSON."""
        data = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "status": self.status,
            "provenance": self.provenance,
            "architect": self.architect,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "manifest_path": self.manifest_path,
            "manifest_hash": self.manifest_hash,
            "hardware": self.hardware_info,
            "config": self.config,
            "best_metrics": self.best_metrics,
            "total_steps_recorded": len(self.step_logs),
            "recent_logs": self.step_logs[-25:],
            "log_file": self.log_file
        }
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Warning: Could not write experiment log: {e}")
