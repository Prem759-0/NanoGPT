"""
Moon AI — Moon Core Unified Orchestrator
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT

Coordinates specialist engines across conversational, code, visual, document, and research tasks:
- Analyzes user intent.
- Deconstructs complex requests into coordinated execution stages.
- Routes tasks to appropriate specialist engines.
- Manages fallback mechanisms when engines are training-pending or unconfigured.
- Aggregates final unified responses.
"""

from typing import Dict, Any, List, Optional
import re
from datetime import datetime, timezone
from core.multimodal.registry import (
    MultimodalCapabilityRegistry,
    STATUS_MODEL_TRAINED,
    STATUS_PROVIDER_CONNECTED
)

class MoonCoreOrchestrator:
    """Central intelligent dispatch and coordination engine."""

    def __init__(self, registry: Optional[MultimodalCapabilityRegistry] = None):
        self.registry = registry or MultimodalCapabilityRegistry()

    def analyze_intent(self, user_prompt: str) -> Dict[str, Any]:
        """
        Classifies user prompt and identifies required specialist engines.
        Supports single-engine and multi-engine composite workflows.
        """
        prompt_lower = user_prompt.lower()
        selected_engines = []
        is_multi_engine = False

        # Visual triggers
        wants_image = any(w in prompt_lower for w in ["image", "picture", "photo", "render", "illustration", "generate an image"])
        wants_video = any(w in prompt_lower for w in ["video", "animation", "motion", "clip", "generate a video", "higgsfield"])

        # Code triggers
        wants_code = any(w in prompt_lower for w in ["code", "function", "script", "debug", "refactor", "unit test", "python", "javascript", "bug"])

        # Document triggers
        wants_doc = any(w in prompt_lower for w in ["pdf", "docx", "pptx", "xlsx", "spreadsheet", "presentation", "report", "document"])

        # Research triggers
        wants_research = any(w in prompt_lower for w in ["research", "search the web", "latest news", "find citations", "sources", "fact-check"])

        if wants_video:
            selected_engines.append("moon-video")
        elif wants_image:
            selected_engines.append("moon-image")

        if wants_doc:
            selected_engines.append("moon-documents")

        if wants_code:
            selected_engines.append("moon-code")

        if wants_research:
            selected_engines.append("moon-research")

        if not selected_engines:
            selected_engines.append("moon-chat")

        if len(selected_engines) > 1:
            is_multi_engine = True

        return {
            "prompt": user_prompt,
            "primary_engine": selected_engines[0],
            "all_required_engines": selected_engines,
            "is_multi_engine": is_multi_engine,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def plan_execution(self, intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Formulates ordered execution steps and verifies engine operational readiness.
        """
        engines = intent.get("all_required_engines", ["moon-chat"])
        plan = []

        for step_idx, eng_key in enumerate(engines, 1):
            eng_meta = self.registry.get_engine(eng_key)
            status = eng_meta.get("status", "NOT_IMPLEMENTED") if eng_meta else "NOT_IMPLEMENTED"

            # Check if engine is directly executable
            can_execute = status in (STATUS_MODEL_TRAINED, STATUS_PROVIDER_CONNECTED)

            action = {
                "step": step_idx,
                "engine": eng_key,
                "engine_name": eng_meta.get("name", eng_key) if eng_meta else eng_key,
                "status": status,
                "can_execute": can_execute,
                "fallback_plan": "Execute architectural planning via Moon Core." if not can_execute else None
            }
            plan.append(action)

        return plan

    def execute_workflow(self, user_prompt: str) -> Dict[str, Any]:
        """
        Coordinates full dispatch workflow from intent to execution plan.
        """
        intent = self.analyze_intent(user_prompt)
        plan = self.plan_execution(intent)

        return {
            "orchestrator": "Moon Core Unified Dispatcher",
            "prompt": user_prompt,
            "intent": intent,
            "execution_plan": plan,
            "total_steps": len(plan),
            "status": "WORKFLOW_PLANNED"
        }

