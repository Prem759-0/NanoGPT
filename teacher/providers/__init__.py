"""
Moon AI Teacher Providers Package
"""
from teacher.providers.base import (
    BaseTeacherProvider,
    TeacherProviderError,
    TeacherRateLimitError,
    TeacherAuthenticationError,
    TeacherTimeoutError,
    TeacherResponseError,
    redact_api_key
)
from teacher.providers.kimi import KimiProvider
from teacher.providers.mock import MockTeacherProvider
from teacher.providers.local import (
    LocalTeacherProvider,
    LocalTeacherNotInstalledError,
    TeacherStudentIsolationViolation,
    STATUS_NOT_INSTALLED,
    STATUS_INSTALLED
)
from teacher.providers.registry import get_teacher_provider, register_provider

