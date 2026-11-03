"""
Moon AI — Controlled Learning Lifecycle Engine
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT

Implements the strict, staged improvement lifecycle:
Interaction
  -> Consent & Eligibility Verification
  -> Privacy & Secret Filtering
  -> Candidate Example Formulation
  -> Quality & Safety Evaluation
  -> Dataset Staging Approval
  -> Scheduled Batch Training (Never Live Updates)
  -> Candidate Checkpoint Formulation
  -> Independent Evaluation Gate
  -> Production Promotion or Rollback

Critical Architectural Invariants:
1. Zero automatic weight updates during live user conversations.
2. Zero silent ingestion of private conversations without explicit user training consent.
3. User Memory, Feedback Data, Training Data, and Model Weights are four strictly separate entities.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import hashlib
import json

from teacher.validation.filter import SensitiveDataFilter, PromptInjectionFilter

class ControlledLearningLifecycle:
    """Orchestrates safe, non-live model learning and data eligibility."""

    def __init__(self, allow_live_weight_updates: bool = False):
        # Strict Invariant: Live weight updates are permanently disabled
        if allow_live_weight_updates:
            raise ValueError("Safety violation: Live model weight updates during conversations are forbidden.")
        self.allow_live_weight_updates = False
        self.staged_training_candidates: List[Dict[str, Any]] = []
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
        # 1. Explicit double opt-in consent requirement
        if not user_training_consent:
            return False, f"Ineligible: User '{user_id}' has not opted in to training data contribution."

        # 2. Private conversation protection
        if is_private_chat and not interaction.get("explicitly_donated_for_training", False):
            return False, "Ineligible: Private conversations cannot be silently ingested without explicit donation."

        # 3. Minimum length & substantive turn check
        prompt = interaction.get("prompt", "")
        response = interaction.get("response", "")
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
        Runs candidate through consent, privacy filtering, and formatting.
        Never updates model weights directly.
        """
        # Step 1: Consent check
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

        prompt = interaction.get("prompt", "")
        response = interaction.get("response", "")

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

        # Step 4: Formulation into Staged Candidate
        candidate_record = {
            "id": f"user_contrib_{hashlib.sha256(combined.encode('utf-8')).hexdigest()[:12]}",
            "prompt": prompt,
            "response": response,
            "staged_at": datetime.now(timezone.utc).isoformat(),
            "donor_user_id": user_id,
            "status": "STAGED_FOR_BATCH_TRAINING",
            "quality_score": interaction.get("user_rating", 5) * 20.0  # Scale 1-5 to 20-100
        }
        self.staged_training_candidates.append(candidate_record)
        return True, "Candidate accepted and staged for scheduled offline batch training.", candidate_record

    def get_lifecycle_status(self) -> Dict[str, Any]:
        """Returns diagnostic status of the learning pipeline."""
        return {
            "live_weight_updates_enabled": self.allow_live_weight_updates,
            "staged_candidates_count": len(self.staged_training_candidates),
            "rejected_candidates_count": len(self.rejected_candidates),
            "pipeline_stages": [
                "Interaction",
                "Consent_Check",
                "Privacy_Filtering",
                "Candidate_Formulation",
                "Staging_Approval",
                "Scheduled_Batch_Training",
                "Candidate_Checkpoint",
                "Independent_Evaluation",
                "Promotion_Gate"
            ]
        }

