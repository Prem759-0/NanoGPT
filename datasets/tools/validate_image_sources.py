"""
datasets/tools/validate_image_sources.py
Licensing and Provenance Validation for Moon AI Image Datasets.
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)

Enforces fail-closed validation:
1. Permissible open licenses only with verified evidence or explicit owner declaration.
2. Verified creator and verifiable asset-level source URL attribution.
3. Explicit is_synthetic=false requirement (never defaulted).
4. Safe path resolution and actual file byte SHA-256 verification.
5. Strict rejection of synthetic fixtures from production manifests.
"""
import os
import sys
from typing import Dict, Any, List, Tuple, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.tools.provenance_validator import (
    validate_provenance_record,
    PERMISSIBLE_LICENSES as VALID_IMAGE_LICENSES,
    VALID_IMAGE_CATEGORIES
)

def validate_image_source_record(
    record: Dict[str, Any],
    allow_synthetic: bool = False,
    base_dir: str = PROJECT_ROOT,
    verify_file: bool = False,
    verifiers_path: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validates a single image metadata record against licensing, provenance, and hygiene rules.
    Returns (is_valid, list_of_errors).
    """
    audit = validate_provenance_record(
        record=record,
        modality="image",
        base_dir=base_dir,
        allow_synthetic=allow_synthetic,
        verify_file=verify_file,
        verifiers_path=verifiers_path
    )
    return audit["is_valid"], audit["errors"]
