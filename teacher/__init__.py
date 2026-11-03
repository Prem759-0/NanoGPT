"""
Moon AI — Teacher Distillation & Multimodal Foundation Engine
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT
"""

from teacher.providers import (
    BaseTeacherProvider,
    KimiProvider,
    MockTeacherProvider,
    get_teacher_provider,
    register_provider,
    redact_api_key
)
from teacher.schemas import (
    TeacherExample,
    validate_teacher_example,
    compute_example_sha256,
    VALID_DOMAINS,
    STATUS_APPROVED,
    STATUS_QUARANTINED,
    STATUS_REJECTED
)
from teacher.validation import (
    SensitiveDataFilter,
    PromptInjectionFilter,
    DuplicateDetector,
    QualityValidator,
    LeakageDetector,
    TeacherDataPipeline
)
from teacher.cost_tracking import TokenCostTracker
from teacher.generation import DemonstrationGenerator

