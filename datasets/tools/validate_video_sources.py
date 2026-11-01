"""
datasets/tools/validate_video_sources.py
Licensing, Provenance, and Motion Metadata Validation for Moon Video.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Enforces fail-closed validation:
1. Permissible open video licenses with verified evidence or explicit owner declaration.
2. Verified creator and verifiable asset-level source URL.
3. Explicit is_synthetic=false requirement (never defaulted).
4. Informative scene caption + motion description + camera motion.
5. Safe path resolution and actual file byte SHA-256 verification.
6. Rejection of synthetic still-image animations in genuine manifests.
"""
import os
import sys
from typing import Dict, Any, List, Tuple, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.tools.provenance_validator import (
    validate_provenance_record,
    PERMISSIBLE_LICENSES as VALID_VIDEO_LICENSES,
    VALID_CAMERA_MOTIONS,
    VALID_IMAGE_CATEGORIES as VALID_VIDEO_CATEGORIES
)

def validate_video_source_record(
    record: Dict[str, Any],
    allow_synthetic: bool = False,
    base_dir: str = PROJECT_ROOT,
    verify_file: bool = False,
    verifiers_path: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validates a single video metadata record.
    Returns (is_valid, list_of_errors).
    """
    audit = validate_provenance_record(
        record=record,
        modality="video",
        base_dir=base_dir,
        allow_synthetic=allow_synthetic,
        verify_file=verify_file,
        verifiers_path=verifiers_path
    )
    return audit["is_valid"], audit["errors"]
