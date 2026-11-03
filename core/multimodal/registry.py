"""
Moon AI — Multimodal Platform Capability Registry
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT

Central registry defining the capability status and contracts of all Moon AI specialist engines:
1. Moon Chat
2. Moon Code
3. Moon Image
4. Moon Video (ambition comparable to Higgsfield AI)
5. Moon Documents
6. Moon Research
7. Moon Core (Unified Orchestrator)

Architectural Invariant:
Teacher text demonstrations can teach Moon Core how to plan, prompt, and coordinate
visual workflows, but do NOT train latent image diffusion or spatiotemporal video models.
Moon Image and Moon Video are accurately recorded as ARCHITECTURE_DEFINED and
MODEL_TRAINING_PENDING until genuine visual training is conducted.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# Standard Capability Statuses
STATUS_NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
STATUS_ARCHITECTURE_DEFINED = "ARCHITECTURE_DEFINED"
STATUS_PROVIDER_INTEGRATION_PENDING = "PROVIDER_INTEGRATION_PENDING"
STATUS_PROVIDER_CONNECTED = "PROVIDER_CONNECTED"
STATUS_MODEL_TRAINING_PENDING = "MODEL_TRAINING_PENDING"
STATUS_MODEL_TRAINED = "MODEL_TRAINED"
STATUS_EVALUATION_PASSED = "EVALUATION_PASSED"

VALID_CAPABILITY_STATUSES = {
    STATUS_NOT_IMPLEMENTED,
    STATUS_ARCHITECTURE_DEFINED,
    STATUS_PROVIDER_INTEGRATION_PENDING,
    STATUS_PROVIDER_CONNECTED,
    STATUS_MODEL_TRAINING_PENDING,
    STATUS_MODEL_TRAINED,
    STATUS_EVALUATION_PASSED,
}

class MultimodalCapabilityRegistry:
    """Tracks the operational and training readiness of all Moon AI engines."""

    def __init__(self):
        self._engines: Dict[str, Dict[str, Any]] = {
            "moon-chat": {
                "name": "Moon Chat",
                "track": "moon-chat-v0",
                "status": STATUS_MODEL_TRAINING_PENDING,
                "readiness_gate_status": "ENGINEERING_SMOKE_TEST_ONLY",
                "primary_model_type": "causal_autoregressive_transformer",
                "capabilities": [
                    "natural_conversation",
                    "contextual_understanding",
                    "step_by_step_reasoning",
                    "instruction_following",
                    "multimodal_planning"
                ],
                "description": "General-purpose conversation, instruction-following, and reasoning engine."
            },
            "moon-code": {
                "name": "Moon Code",
                "track": "moon-code-v0",
                "status": STATUS_ARCHITECTURE_DEFINED,
                "primary_model_type": "fill_in_the_middle_transformer",
                "capabilities": [
                    "code_generation",
                    "repository_analysis",
                    "debugging",
                    "unit_test_generation",
                    "software_architecture"
                ],
                "description": "Software engineering, code synthesis, and repository refactoring engine."
            },
            "moon-image": {
                "name": "Moon Image",
                "track": "moon-image-v0",
                "status": STATUS_ARCHITECTURE_DEFINED,
                "primary_model_type": "latent_diffusion",
                "capabilities": [
                    "text_to_image",
                    "image_to_image",
                    "image_editing",
                    "inpainting",
                    "outpainting",
                    "resolution_enhancement"
                ],
                "description": "Visual creation engine initialized from random weights. Training pending genuine image collection."
            },
            "moon-video": {
                "name": "Moon Video",
                "track": "moon-video-v0",
                "status": STATUS_ARCHITECTURE_DEFINED,
                "primary_model_type": "spatiotemporal_diffusion",
                "capabilities": [
                    "text_to_video",
                    "image_to_video",
                    "video_extension",
                    "camera_movement_control",
                    "scene_shot_planning",
                    "character_consistency"
                ],
                "description": (
                    "Video generation and editing engine comparable in product ambition to Higgsfield AI. "
                    "Architecture defined with 100% random Gaussian weights. Training pending genuine video dataset collection."
                )
            },
            "moon-documents": {
                "name": "Moon Documents",
                "track": "moon-documents-v0",
                "status": STATUS_PROVIDER_CONNECTED,
                "primary_model_type": "structured_document_engine",
                "capabilities": [
                    "pdf_creation_and_reading",
                    "docx_creation_and_reading",
                    "pptx_creation_and_editing",
                    "xlsx_creation_and_analysis",
                    "csv_markdown_processing",
                    "executive_report_generation"
                ],
                "description": "Comprehensive document creation, parsing, analysis, and report generation engine."
            },
            "moon-research": {
                "name": "Moon Research",
                "track": "moon-research-v0",
                "status": STATUS_PROVIDER_CONNECTED,
                "primary_model_type": "evidence_retrieval_engine",
                "capabilities": [
                    "web_research",
                    "source_citations",
                    "multi_document_summarization",
                    "fact_checking",
                    "structured_data_extraction"
                ],
                "description": "Agentic web search, citation verification, and factual summarization engine."
            },
            "moon-core": {
                "name": "Moon Core",
                "track": "moon-core-v0",
                "status": STATUS_PROVIDER_CONNECTED,
                "primary_model_type": "unified_orchestrator",
                "capabilities": [
                    "intent_understanding",
                    "specialist_engine_routing",
                    "multi_engine_coordination",
                    "tool_permission_management",
                    "failure_handling_and_fallback",
                    "unified_response_aggregation"
                ],
                "description": "Central coordinator routing complex multi-modal requests across specialist engines."
            }
        }

    def get_engine(self, engine_key: str) -> Optional[Dict[str, Any]]:
        """Returns engine metadata by key."""
        return self._engines.get(engine_key.lower())

    def get_all_engines(self) -> Dict[str, Dict[str, Any]]:
        """Returns all registered engines and statuses."""
        return self._engines

    def get_engine_status(self, engine_key: str) -> str:
        """Returns current status of an engine."""
        engine = self.get_engine(engine_key)
        if not engine:
            return STATUS_NOT_IMPLEMENTED
        return engine.get("status", STATUS_NOT_IMPLEMENTED)

    def set_engine_status(self, engine_key: str, status: str):
        """Updates status of an engine ensuring status validity."""
        if status not in VALID_CAPABILITY_STATUSES:
            raise ValueError(f"Invalid capability status '{status}'. Allowed: {VALID_CAPABILITY_STATUSES}")
        if engine_key in self._engines:
            self._engines[engine_key]["status"] = status
            self._engines[engine_key]["last_updated"] = datetime.now(timezone.utc).isoformat()

    def get_status_matrix(self) -> Dict[str, str]:
        """Returns concise mapping of engine key to status."""
        return {k: v["status"] for k, v in self._engines.items()}

