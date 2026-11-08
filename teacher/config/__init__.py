"""
Teacher Configuration Loader
"""
import os
import yaml
from typing import Dict, Any

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "teacher_config.yaml")

def load_teacher_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"allow_external_calls": False, "default_provider": "mock"}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

