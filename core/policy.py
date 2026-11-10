"""
Moon AI — Central Ownership & Training Policy Engine
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Active Policy: STRICT SOVEREIGN MODEL OWNERSHIP (STRICT_SOVEREIGN_RANDOM_INIT_ONLY)

Maintains strict sovereign model independence:
- Moon student models MUST begin exclusively with random Gaussian scratch initialization.
- Student weights are NEVER initialized, transferred, or imported from teacher checkpoints or pretrained models.
- Zero teacher-generated training examples or teacher distillation: teacher demonstrations are strictly excluded from all training pipelines.
- Zero external generative AI inference APIs in runtime or training data generation.
- Gemini and Kimi may assist with engineering and tooling only, never as model teachers or data generators.
- Backward compatibility identifiers for historical evaluation are strictly preserved.
"""

from typing import Dict, Any, List, Optional, Tuple, Set

# Policy Identifiers
POLICY_STRICT_SOVEREIGN_RANDOM_INIT = "STRICT_SOVEREIGN_RANDOM_INIT_ONLY"  # Active Sovereign Policy
POLICY_TEACHER_STUDENT_RANDOM_INIT = "TEACHER_STUDENT_RANDOM_INIT"        # Historical / Comparative Reference

ACTIVE_OWNERSHIP_POLICY = POLICY_STRICT_SOVEREIGN_RANDOM_INIT

VALID_OWNERSHIP_POLICIES: Set[str] = {
    POLICY_STRICT_SOVEREIGN_RANDOM_INIT,
    POLICY_TEACHER_STUDENT_RANDOM_INIT,
}

# Permitted Student Initialization Methods
STUDENT_INIT_RANDOM_GAUSSIAN = "random_gaussian_scratch"
STUDENT_INIT_XAVIER_UNIFORM = "xavier_uniform_scratch"

VALID_STUDENT_INIT_METHODS: Set[str] = {
    STUDENT_INIT_RANDOM_GAUSSIAN,
    STUDENT_INIT_XAVIER_UNIFORM,
}

# Explicit Architectural Definitions
POLICY_DEFINITIONS: Dict[str, str] = {
    "teacher_model": (
        "An external, larger foundation model (e.g. Kimi K3.1 Pro) used exclusively for generating "
        "high-quality demonstrations, reasoning trajectories, synthetic code, and benchmark evaluations. "
        "Its weights are never part of the Moon runtime."
    ),
    "student_model": (
        "The proprietary Moon AI neural network (e.g. Moon Chat, Moon Code) whose weights are initialized "
        "from random Gaussian distributions and trained on vetted data."
    ),
    "teacher_inference": (
        "Execution of the teacher model via authenticated external API or local weights for the purpose of "
        "generating demonstrations or evaluating student outputs."
    ),
    "teacher_generated_data": (
        "Synthetic or annotated conversational, reasoning, or code examples produced by teacher inference. "
        "Must be verified, filtered for sensitive data, and licensed transparently."
    ),
    "student_initialization": (
        "The algorithmic initialization of the student model's parameters. Must be 100% random scratch initialization. "
        "Copying, slicing, or distilling weights from third-party checkpoints into student weights is strictly prohibited."
    ),
    "student_training": (
        "The optimization process (e.g. AdamW, cross-entropy with assistant-only loss masking) adjusting student "
        "parameters on curated teacher demonstrations, original self-authored data, and verified public corpora."
    ),
    "external_provider_dependency": (
        "Any third-party hosted API (e.g. Moonshot AI / Kimi) required during data generation or evaluation. "
        "Must be metered, credential-isolated, mockable in CI, and disabled by default in production runtime."
    )
}

# Permitted and Prohibited Rules for Active Policy
POLICY_RULES: Dict[str, Dict[str, Any]] = {
    POLICY_STRICT_SOVEREIGN_RANDOM_INIT: {
        "allow_teacher_demonstrations": False,
        "allow_licensed_public_corpora": True,
        "allow_original_self_authored": True,
        "allow_third_party_weights": False,
        "allow_external_apis_in_runtime": False,
        "require_random_initialization": True,
        "description": "Active strict sovereign policy: 100% random scratch initialization and zero teacher influence or distillation."
    },
    POLICY_TEACHER_STUDENT_RANDOM_INIT: {
        "allow_teacher_demonstrations": True,
        "allow_licensed_public_corpora": True,
        "allow_original_self_authored": True,
        "allow_user_feedback_with_consent": True,
        "allow_teacher_assisted_evaluation": True,
        "allow_third_party_weights": False,  # Student weights must NEVER be initialized from teacher
        "allow_external_apis_in_runtime": False,  # Runtime inference must be independent
        "require_random_initialization": True,  # Student models MUST begin with random scratch weights
        "description": "Stage 2.1.3+ strategy: Moon student trains from scratch on teacher demonstrations and independent data."
    }
}


def get_active_policy() -> str:
    """Returns the current active ownership policy identifier."""
    return ACTIVE_OWNERSHIP_POLICY


def is_valid_policy(policy: str) -> bool:
    """Checks whether a given policy string is recognized."""
    return policy in VALID_OWNERSHIP_POLICIES


def validate_student_initialization(
    init_method: str,
    checkpoint_metadata: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str]:
    """
    Validates that a student model was initialized legitimately from random weights.
    Strictly forbids initializing student weights from external foundation models.
    """
    if init_method not in VALID_STUDENT_INIT_METHODS:
        return False, (
            f"Invalid student initialization method '{init_method}'. "
            f"Must be one of {VALID_STUDENT_INIT_METHODS}. Initializing from pretrained checkpoints is strictly forbidden."
        )

    if checkpoint_metadata:
        base_weights = checkpoint_metadata.get("base_weights") or checkpoint_metadata.get("pretrained_source")
        if base_weights and base_weights not in ("none", "scratch", "random_gaussian"):
            return False, (
                f"Ownership violation: Checkpoint declares external base weights '{base_weights}'. "
                "Moon student models must be initialized from random Gaussian scratch weights."
            )

    return True, "Student initialization verified: purely random scratch weights."


def get_policy_summary(policy: Optional[str] = None) -> Dict[str, Any]:
    """Returns a comprehensive summary of policy rules and architectural definitions."""
    pol = policy or ACTIVE_OWNERSHIP_POLICY
    rules = POLICY_RULES.get(pol, {})
    return {
        "policy": pol,
        "is_active": pol == ACTIVE_OWNERSHIP_POLICY,
        "rules": rules,
        "definitions": POLICY_DEFINITIONS
    }

