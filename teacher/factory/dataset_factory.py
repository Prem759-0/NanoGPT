"""
Moon AI — Offline Teacher Dataset Factory
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT (Policy B)

Orchestrates offline dataset curation and generation across all 10 curriculum domains:
1. General conversation.
2. Instruction following.
3. Mathematics and logical reasoning.
4. Programming fundamentals.
5. Advanced programming and debugging.
6. Software architecture.
7. Computer science.
8. Research and evidence evaluation.
9. Tool selection and usage.
10. Safety and uncertainty handling.

Invariants:
- 100% offline-capable; zero external API keys or network endpoints.
- If LocalTeacherProvider is uninstalled, reports LOCAL_TEACHER_NOT_INSTALLED and halts (zero silent mock fallback).
- Mock provider records are strictly stamped as dev/test fixtures and rejected by production readiness gates.
- Requires explicit review approval before admitting generated records into eligible production manifests.
"""

import os
import sys
import json
import uuid
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from teacher.providers.base import BaseTeacherProvider
from teacher.providers.local import LocalTeacherProvider, LocalTeacherNotInstalledError, STATUS_NOT_INSTALLED
from teacher.providers.mock import MockTeacherProvider
from teacher.schemas.example import (
    TeacherExample,
    CURRICULUM_DOMAINS_2_1_4,
    STATUS_APPROVED,
    STATUS_QUARANTINED,
    STATUS_REJECTED,
    REVIEW_STATUS_APPROVED,
    REVIEW_STATUS_PENDING,
    REVIEW_STATUS_QUARANTINED,
    REVIEW_STATUS_REJECTED,
    SPLIT_TRAIN,
    SPLIT_VAL,
    SPLIT_TEST,
    SPLIT_QUARANTINE,
    validate_teacher_example
)
from teacher.validation.filter import (
    TeacherDataPipeline,
    SensitiveDataFilter,
    PromptInjectionFilter,
    DuplicateDetector,
    QualityValidator,
    CodeSyntaxValidator,
    LicenseSourceValidator,
    LeakageDetector,
    QuarantineManager,
    CurriculumDistributionTracker
)


@dataclass
class DatasetBatchResult:
    """Summary of a dataset generation or audit batch."""
    total_generated: int
    approved_count: int
    quarantined_count: int
    rejected_count: int
    is_mock_batch: bool
    domain_distribution: Dict[str, Any]
    records: List[Dict[str, Any]]


class ApprovalGate:
    """Manages explicit human or audited approval for generated training records."""

    @classmethod
    def review_and_approve(
        cls,
        example: TeacherExample,
        reviewer_id: str = "curator_audit_v1",
        notes: str = "Verified educational accuracy and provenance compliance."
    ) -> Tuple[bool, str]:
        """Approves a validated example for production manifest entry."""
        if example.is_dev_fixture or example.teacher_provider == "mock":
            return False, "Approval rejected: Mock provider records are development fixtures only."

        if example.validation_status != STATUS_APPROVED:
            return False, f"Approval rejected: Record has validation status '{example.validation_status}'."

        example.reviewer_status = REVIEW_STATUS_APPROVED
        example.generation_metadata["approved_by"] = reviewer_id
        example.generation_metadata["approval_notes"] = notes
        example.generation_metadata["approved_at"] = datetime.now(timezone.utc).isoformat()
        return True, "Record approved for production manifests."


class OfflineDatasetFactory:
    """
    Offline dataset factory producing and validating structured demonstrations
    across the 10 core curriculum domains.
    """

    def __init__(
        self,
        provider: BaseTeacherProvider,
        pipeline: Optional[TeacherDataPipeline] = None,
        quarantine_file: Optional[str] = None
    ):
        self.provider = provider
        self.pipeline = pipeline or TeacherDataPipeline()
        if quarantine_file:
            self.pipeline.quarantine_mgr = QuarantineManager(quarantine_file=quarantine_file)
        self.curated_pool: Dict[str, TeacherExample] = {}

    def check_provider_status(self) -> Dict[str, Any]:
        """Checks provider operational status and local checkpoint availability."""
        if isinstance(self.provider, LocalTeacherProvider):
            return self.provider.check_installation()
        elif isinstance(self.provider, MockTeacherProvider):
            return {
                "status": "MOCK_PROVIDER_ACTIVE",
                "installed": True,
                "is_dev_fixture": True,
                "message": "Mock provider active for testing only. Not eligible for production dataset curation."
            }
        else:
            return {
                "status": "PROVIDER_CONNECTED",
                "installed": True,
                "provider_name": self.provider.provider_name
            }

    def generate_domain_record(
        self,
        domain: str,
        prompt: str,
        task_type: str = "instruction_following",
        license_str: str = "CC-BY-4.0",
        source_url: str = "local://curated_offline_prompts",
        record_id: Optional[str] = None
    ) -> Tuple[bool, TeacherExample]:
        """
        Generates and validates a single demonstration record for a specified curriculum domain.
        Fails closed with LocalTeacherNotInstalledError if local teacher is missing.
        Never silently falls back to mock.
        """
        if domain not in CURRICULUM_DOMAINS_2_1_4:
            raise ValueError(f"Domain '{domain}' is not in the 10 curriculum domains: {CURRICULUM_DOMAINS_2_1_4}")

        # Check local teacher installation
        if isinstance(self.provider, LocalTeacherProvider) and not self.provider.is_installed():
            raise LocalTeacherNotInstalledError(
                f"LOCAL_TEACHER_NOT_INSTALLED: Local teacher model checkpoint not installed at '{self.provider.checkpoint_path}'. "
                "Production dataset curation halted. Silent fallback to mock is strictly forbidden."
            )

        rec_id = record_id or f"moon_{domain[:4]}_{uuid.uuid4().hex[:8]}"

        # Invoke provider based on domain
        if "math" in domain or "reason" in domain:
            gen_res = self.provider.generate_reasoning_demonstration(prompt, domain=domain)
            response_text = gen_res["raw_response"]
        elif "program" in domain or "code" in domain:
            gen_res = self.provider.generate_code_demonstration(prompt)
            response_text = gen_res["raw_response"]
        elif "tool" in domain:
            gen_res = self.provider.generate_tool_use_demonstration(
                prompt,
                available_tools=[{"name": "local_calc", "description": "Local deterministic calculator"}]
            )
            response_text = gen_res.get("thought", "Executed local tool.")
        else:
            gen_res = self.provider.generate_completion(prompt)
            response_text = gen_res["text"]

        is_mock = isinstance(self.provider, MockTeacherProvider) or getattr(self.provider, "is_mock", False)
        source_type = "DEV_FIXTURE_MOCK" if is_mock else "LOCAL_TEACHER_OFFLINE"

        # Build complete schema-compliant record
        example = TeacherExample.create(
            example_id=rec_id,
            teacher_provider=self.provider.provider_name,
            teacher_model_id=self.provider.model_id,
            prompt=prompt,
            response=response_text,
            domain=domain,
            task_type=task_type,
            license_str=license_str,
            source_type=source_type,
            source_url=source_url,
            is_dev_fixture=is_mock,
            is_synthetic=is_mock,
            split=SPLIT_TRAIN
        )

        # Run quality control pipeline
        passed, validated = self.pipeline.process_candidate(example)
        self.curated_pool[rec_id] = validated
        return passed, validated

    def approve_record(self, record_id: str, reviewer_id: str = "curator_audit") -> Tuple[bool, str]:
        """Explicitly reviews and approves a candidate record for production admission."""
        if record_id not in self.curated_pool:
            return False, f"Record '{record_id}' not found in curated pool."
        ex = self.curated_pool[record_id]
        return ApprovalGate.review_and_approve(ex, reviewer_id=reviewer_id)

    def get_curriculum_distribution(self) -> Dict[str, Any]:
        """Returns distribution metrics across all 10 curriculum domains."""
        return self.pipeline.distribution_tracker.get_distribution_report()

    def export_production_manifest(self, file_path: str, split: str = SPLIT_TRAIN) -> Tuple[int, List[str]]:
        """
        Exports only APPROVED, NON-DEV-FIXTURE records into a target production manifest.
        Returns (count_exported, list_of_errors).
        """
        eligible_records = []
        errors = []

        for r_id, ex in self.curated_pool.items():
            if ex.split != split:
                continue

            rec = ex.to_dict()
            is_valid, err_msg = validate_teacher_example(rec, is_production=True)
            if is_valid:
                eligible_records.append(rec)
            else:
                errors.append(f"Record '{r_id}' excluded from production export: {err_msg}")

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            for r in eligible_records:
                f.write(json.dumps(r) + "\n")

        return len(eligible_records), errors

    @staticmethod
    def compute_manifest_sha256(file_path: str) -> str:
        """Computes deterministic SHA-256 for a manifest file on disk."""
        if not os.path.isfile(file_path):
            return "none"
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

