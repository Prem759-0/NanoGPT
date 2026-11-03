"""
Moon AI — Teacher Demonstration Generator
Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT

Coordinates teacher provider generation, schema formatting, and validation pipeline.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import uuid

from teacher.providers.base import BaseTeacherProvider
from teacher.schemas.example import (
    TeacherExample,
    DOMAIN_GENERAL_CONVERSATION,
    DOMAIN_MATH_LOGIC_REASONING,
    DOMAIN_PROGRAMMING_DEBUGGING,
    DOMAIN_TOOL_SELECTION_EXECUTION,
    SPLIT_TRAIN
)
from teacher.validation.filter import TeacherDataPipeline
from teacher.cost_tracking.tracker import TokenCostTracker

class DemonstrationGenerator:
    """End-to-end generator creating validated teacher demonstrations."""

    def __init__(
        self,
        provider: BaseTeacherProvider,
        pipeline: Optional[TeacherDataPipeline] = None,
        tracker: Optional[TokenCostTracker] = None
    ):
        self.provider = provider
        self.pipeline = pipeline or TeacherDataPipeline()
        self.tracker = tracker or TokenCostTracker()

    def generate_and_validate(
        self,
        prompt: str,
        domain: str = DOMAIN_GENERAL_CONVERSATION,
        task_type: str = "general_qa",
        license_str: str = "Permissive-Teacher-Terms",
        example_id: Optional[str] = None
    ) -> Tuple[bool, TeacherExample]:
        """
        Invokes provider based on task_type, packages response into TeacherExample,
        runs validation pipeline, records token cost, and returns (passed, example).
        """
        rec_id = example_id or f"teacher_{self.provider.provider_name}_{uuid.uuid4().hex[:8]}"

        # Route to specialized provider method based on domain / task_type
        if domain == DOMAIN_MATH_LOGIC_REASONING:
            gen_res = self.provider.generate_reasoning_demonstration(prompt, domain=domain)
            response_text = gen_res["raw_response"]
        elif domain == DOMAIN_PROGRAMMING_DEBUGGING:
            gen_res = self.provider.generate_code_demonstration(prompt)
            response_text = gen_res["raw_response"]
        elif domain == DOMAIN_TOOL_SELECTION_EXECUTION:
            gen_res = self.provider.generate_tool_use_demonstration(prompt, available_tools=[{"name": "search", "description": "Web search tool"}])
            response_text = gen_res.get("thought", "Executed tool selection.")
        else:
            gen_res = self.provider.generate_completion(prompt)
            response_text = gen_res["text"]

        # Track token usage if reported
        usage = gen_res.get("usage", {})
        p_tok = usage.get("prompt_tokens", len(prompt.split()))
        c_tok = usage.get("completion_tokens", len(response_text.split()))
        self.tracker.track_call(
            provider=self.provider.provider_name,
            model_id=self.provider.model_id,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            is_success=True
        )

        # Build candidate TeacherExample
        candidate = TeacherExample.create(
            example_id=rec_id,
            teacher_provider=self.provider.provider_name,
            teacher_model_id=self.provider.model_id,
            prompt=prompt,
            response=response_text,
            domain=domain,
            task_type=task_type,
            license_str=license_str,
            split=SPLIT_TRAIN
        )

        # Run validation pipeline
        passed, validated_example = self.pipeline.process_candidate(candidate)
        return passed, validated_example

