"""
Moon AI — Controlled Learning Lifecycle Engine
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: STRICT_SOVEREIGN_RANDOM_INIT_ONLY

Implements the strict, staged improvement lifecycle:
Interaction
  -> Explicit Donation Consent & Eligibility Verification
  -> Sovereign Privacy & Secret Filtering
  -> Prompt Injection Screening
  -> Objective Deterministic Verification (AST / Sandbox / Math)
  -> Deduplication & Durable WAL Staging (Never Live Updates)
  -> Scheduled Offline Batch Training (Only with AUTHORIZED_BY_PREM = True)
  -> Candidate Checkpoint Formulation
  -> Independent Evaluation Gate
  -> Production Promotion or Rollback

Critical Architectural Invariants:
1. Zero automatic weight updates during live user conversations.
2. Zero silent ingestion of private conversations without explicit user training consent.
3. User Memory, Feedback Data, Candidate Data, and Model Weights are four strictly separate entities.
4. Subjective user ratings are NEVER treated as correctness proofs.
5. Zero teacher models, teacher distillation, or external third-party inference.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import hashlib
import json

from core.security.filtering import SensitiveDataFilter, PromptInjectionFilter, InputIntegrityValidator
from core.learning_lifecycle import donation as donation_module
from core.learning_lifecycle.donation import (
    DonationConsentManager,
    DONATION_POLICY_VERSION,
    is_authenticated_user,
)
from core.learning_lifecycle.verifier import ObjectiveVerifier, REJECTED
from core.learning_lifecycle.candidate_store import CandidateWALStore, STATE_APPROVED_CANDIDATE, STATE_REJECTED


class ControlledLearningLifecycle:
    """Orchestrates safe, non-live model learning and data eligibility."""

    def __init__(
        self,
        allow_live_weight_updates: bool = False,
        wal_path: Optional[str] = None,
        consent_db_path: Optional[str] = None
    ):
        # Strict Invariant: Live weight updates are permanently disabled
        if allow_live_weight_updates:
            raise ValueError("Safety violation: Live model weight updates during conversations are forbidden.")
        self.allow_live_weight_updates = False

        # Durable WAL candidate store
        self.candidate_store = CandidateWALStore(wal_path=wal_path)
        self.consent_manager = DonationConsentManager(storage_path=consent_db_path)
        self.verifier = ObjectiveVerifier()

        # In-memory collections maintained for backwards compatibility
        self.staged_training_candidates: List[Dict[str, Any]] = self.candidate_store.list_candidates()
        self.rejected_candidates: List[Dict[str, Any]] = []

    def check_interaction_training_eligibility(
        self,
        user_id: str,
        interaction: Dict[str, Any],
        user_training_consent: bool,
        is_private_chat: bool = True
    ) -> Tuple[bool, str]:
        """
        Validates whether a conversational turn is legally and ethically eligible
        to be processed as candidate training data.
        """
        # 0. Global ingestion gate & server-side authentication containment (Phase 2.6.2.2 Task 1)
        if not donation_module.DONATION_INGESTION_ENABLED or not donation_module.SERVER_AUTHENTICATION_VERIFIED:
            return False, (
                "Ineligible: Server-side cryptographic authentication is not verified. "
                "Donation ingestion is permanently disabled."
            )

        # 1. User authentication check (SEC-08)
        if not is_authenticated_user(user_id):
            return False, "Ineligible: Unauthenticated user."

        # 2. Input integrity check
        prompt = interaction.get("prompt", "")
        response = interaction.get("response", "")
        valid_p, p_err = InputIntegrityValidator.validate_input(prompt)
        if not valid_p:
            return False, f"Ineligible: {p_err}"
        valid_r, r_err = InputIntegrityValidator.validate_input(response)
        if not valid_r:
            return False, f"Ineligible: {r_err}"

        # 3. Explicit double opt-in consent requirement
        if not user_training_consent:
            return False, f"Ineligible: User '{user_id}' has not opted in to training data contribution."

        # 4. Memory isolation invariant: memory items can NEVER be converted into training candidates
        if interaction.get("eligible_for_training") is False and interaction.get("source") == "memory":
            return False, "Ineligible: User personal memories cannot be converted to training candidates."

        # 5. Private conversation protection: must have explicit affirmative donation
        if is_private_chat and not interaction.get("explicitly_donated_for_training", False):
            return False, "Ineligible: Private conversations cannot be silently ingested without explicit donation."

        # 6. Minimum length & substantive turn check
        if len(prompt.strip()) < 10 or len(response.strip()) < 15:
            return False, "Ineligible: Interaction is too brief or trivial."

        return True, "Eligible for privacy filtering and quality staging."

    def ingest_candidate_example(
        self,
        user_id: str,
        interaction: Dict[str, Any],
        user_training_consent: bool
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Runs candidate through consent, privacy filtering, objective verification, and WAL staging.
        Never updates model weights directly.
        """
        prompt = interaction.get("prompt", "")
        response = interaction.get("response", "")
        cid = interaction.get("consent_id")

        # Step 0a: Fail closed if server-side authentication is unverified or ingestion disabled (Phase 2.6.2.2 Task 1)
        if not donation_module.DONATION_INGESTION_ENABLED or not donation_module.SERVER_AUTHENTICATION_VERIFIED:
            rej_reason = (
                "Ineligible: Server-side cryptographic authentication is not verified. "
                "Donation ingestion is permanently disabled."
            )
            self.rejected_candidates.append({"user_id": user_id, "reason": rej_reason})
            return False, rej_reason, None

        # Step 0: Require explicit valid consent_id; reject synthetic IDs (SEC-03)
        if not cid:
            rej_reason = "Ineligible: Missing explicit consent_id. Synthetic consent IDs are strictly prohibited."
            self.rejected_candidates.append({"user_id": user_id, "reason": rej_reason})
            return False, rej_reason, None

        if cid.startswith("auto_consent_"):
            rej_reason = "Ineligible: Synthetic auto_consent IDs are strictly prohibited."
            self.rejected_candidates.append({"user_id": user_id, "reason": rej_reason})
            return False, rej_reason, None

        # Step 0b: Check durable consent record in DonationConsentManager (SEC-02, SEC-03)
        consent_ok, consent_reason, consent_rec = self.consent_manager.verify_donation_eligibility(
            user_id=user_id,
            interaction=interaction,
            consent_id=cid
        )
        if not consent_ok:
            self.rejected_candidates.append({"user_id": user_id, "reason": consent_reason})
            return False, consent_reason, None

        # Step 1: Interaction format & policy eligibility check
        is_eligible, reason = self.check_interaction_training_eligibility(
            user_id, interaction, user_training_consent,
            is_private_chat=interaction.get("is_private_chat", True)
        )
        if not is_eligible:
            self.rejected_candidates.append({
                "user_id": user_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": reason
            })
            return False, reason, None

        # Step 2: Privacy and Sensitive Data Filtering
        combined = f"{prompt} {response}"
        has_pii, pii_errs = SensitiveDataFilter.check_sensitive_data(combined)
        if has_pii:
            rej_reason = f"Privacy violation: {'; '.join(pii_errs)}"
            self.rejected_candidates.append({"user_id": user_id, "reason": rej_reason})
            return False, rej_reason, None

        # Step 3: Prompt Injection Check
        is_inj, inj_err = PromptInjectionFilter.check_injection(prompt)
        if is_inj:
            self.rejected_candidates.append({"user_id": user_id, "reason": inj_err})
            return False, inj_err, None

        # Step 4: Objective Verification & Classification
        subjective_rating = interaction.get("user_rating")
        v_class, v_reason, v_meta = self.verifier.classify_and_verify(
            prompt, response, subjective_rating=subjective_rating
        )
        if v_class == REJECTED:
            rej_reason = f"Objective verification failed: {v_reason}"
            self.rejected_candidates.append({"user_id": user_id, "reason": rej_reason})
            return False, rej_reason, None

        # Step 5: Durable WAL Staging & Deduplication
        cand_id = f"user_contrib_{hashlib.sha256(combined.encode('utf-8')).hexdigest()[:12]}"
        ok_stage, stage_msg, candidate_record = self.candidate_store.stage_candidate(
            candidate_id=cand_id,
            user_id=user_id,
            consent_ref=cid,
            prompt=prompt,
            response=response,
            policy_version=DONATION_POLICY_VERSION,
            verification_status=v_class,
            verification_method=v_reason,
            sanitization_status="PASSED",
            subjective_feedback={"user_rating": subjective_rating}
        )

        if not ok_stage or not candidate_record:
            self.rejected_candidates.append({"user_id": user_id, "reason": stage_msg})
            return False, stage_msg, None

        # Backwards compatibility compatibility mirror
        self.staged_training_candidates = self.candidate_store.list_candidates()
        return True, "Candidate accepted and staged for scheduled offline batch training.", candidate_record

    def get_lifecycle_status(self) -> Dict[str, Any]:
        """Returns diagnostic status of the learning pipeline."""
        wal_metrics = self.candidate_store.get_wal_metrics()
        return {
            "live_weight_updates_enabled": self.allow_live_weight_updates,
            "staged_candidates_count": len(self.candidate_store.list_candidates()),
            "rejected_candidates_count": len(self.rejected_candidates),
            "wal_metrics": wal_metrics,
            "pipeline_stages": [
                "Interaction",
                "Consent_Check",
                "Privacy_Filtering",
                "Prompt_Injection_Screening",
                "Objective_Verification",
                "Deduplication",
                "Durable_WAL_Staging",
                "Scheduled_Offline_Batch_Training",
                "Candidate_Checkpoint",
                "Independent_Evaluation",
                "Promotion_Gate"
            ]
        }
