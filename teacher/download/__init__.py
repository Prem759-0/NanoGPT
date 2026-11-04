"""
Moon AI — Teacher Model Download Package
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT (Policy B)
"""

from teacher.download.downloader import (
    PublicModelDownloader,
    DownloadVerificationError,
    InsufficientStorageError,
    QUALIFIED_TEACHER_MODELS,
    PRIMARY_TEACHER_KEY
)

__all__ = [
    "PublicModelDownloader",
    "DownloadVerificationError",
    "InsufficientStorageError",
    "QUALIFIED_TEACHER_MODELS",
    "PRIMARY_TEACHER_KEY"
]

