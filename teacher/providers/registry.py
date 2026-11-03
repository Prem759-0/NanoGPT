"""
Moon AI — Teacher Provider Registry & Factory
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT
"""

from typing import Dict, Type, Optional, Any
from teacher.providers.base import BaseTeacherProvider
from teacher.providers.kimi import KimiProvider
from teacher.providers.mock import MockTeacherProvider
from teacher.config import load_teacher_config

_PROVIDER_REGISTRY: Dict[str, Type[BaseTeacherProvider]] = {
    "kimi": KimiProvider,
    "mock": MockTeacherProvider,
}

def register_provider(name: str, provider_cls: Type[BaseTeacherProvider]):
    """Registers a new custom teacher provider."""
    _PROVIDER_REGISTRY[name.lower()] = provider_cls

def get_teacher_provider(
    provider_name: Optional[str] = None,
    allow_external_calls: Optional[bool] = None,
    **kwargs
) -> BaseTeacherProvider:
    """
    Factory creating a teacher provider instance.
    Defaults to configuration if provider_name is omitted.
    """
    cfg = load_teacher_config()
    p_name = (provider_name or cfg.get("default_provider", "kimi")).lower()

    if p_name not in _PROVIDER_REGISTRY:
        raise ValueError(f"Unknown teacher provider '{p_name}'. Registered: {list(_PROVIDER_REGISTRY.keys())}")

    provider_cls = _PROVIDER_REGISTRY[p_name]

    # Resolve allow_external_calls
    ext_allowed = allow_external_calls
    if ext_allowed is None:
        ext_allowed = cfg.get("allow_external_calls", False)

    if p_name == "kimi":
        k_cfg = cfg.get("providers", {}).get("kimi", {})
        return KimiProvider(
            model_id=kwargs.get("model_id", k_cfg.get("model_id", "kimi-k3.1-pro")),
            timeout_seconds=kwargs.get("timeout_seconds", k_cfg.get("timeout_seconds", 30.0)),
            allow_external_calls=ext_allowed,
            api_key=kwargs.get("api_key")
        )
    elif p_name == "mock":
        return MockTeacherProvider(
            model_id=kwargs.get("model_id", "mock-teacher-v1"),
            simulate_error=kwargs.get("simulate_error"),
            custom_response=kwargs.get("custom_response")
        )
    else:
        return provider_cls(allow_external_calls=ext_allowed, **kwargs)

