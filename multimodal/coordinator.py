"""
multimodal/coordinator.py
Unified Moon Core Multimodal Coordinator.

Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT (Policy B)

Orchestrates and routes between specialized sovereign engines:
1. Moon Core LLM Engine (ModelService: reasoning, chat, software engineering).
2. Moon Image Studio (ImageService: procedural & latent diffusion visual synthesis).
3. Moon Video Studio (VideoService: spatio-temporal video sequences).
4. Sovereign Document Engine (DocumentEngine: PDF, DOCX, XLSX, PPTX generation & parsing).
5. Secure Code Interpreter (CodeService: sandboxed execution).
"""

import os
import sys
import re
from typing import Dict, Any, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from webapp.model_service import ModelService
from webapp.image_service import ImageService
from webapp.video_service import VideoService
from webapp.code_service import CodeService
from multimodal.document_engine import SovereignDocumentEngine


class MoonMultimodalCoordinator:
    """Unified Moon Core coordinator for multi-engine orchestration."""

    def __init__(
        self,
        model_service: Optional[ModelService] = None,
        image_service: Optional[ImageService] = None,
        video_service: Optional[VideoService] = None,
        code_service: Optional[CodeService] = None,
        document_engine: Optional[SovereignDocumentEngine] = None
    ):
        self.model_service = model_service or ModelService(base_dir=PROJECT_ROOT)
        static_gen_dir = os.path.join(PROJECT_ROOT, "webapp", "static", "generated")
        os.makedirs(static_gen_dir, exist_ok=True)
        self.image_service = image_service or ImageService(output_dir=static_gen_dir)
        self.video_service = video_service or VideoService(output_dir=static_gen_dir)
        self.code_service = code_service or CodeService()
        self.document_engine = document_engine or SovereignDocumentEngine()

    def detect_intent(self, prompt: str) -> str:
        """Determines the primary modality and intent of the user prompt."""
        p = prompt.strip().lower()

        # 1. Video generation
        if p.startswith("/video") or any(k in p for k in ["generate video", "create video", "make video", "render video"]):
            return "video_generation"

        # 2. Image generation
        if (
            p.startswith("/imagine")
            or p.startswith("/image")
            or any(k in p for k in ["generate image", "create image", "paint", "draw a", "render image", "photo of", "picture of"])
        ):
            return "image_generation"

        # 3. Document generation
        if any(k in p for k in ["generate pdf", "create pdf", "export pdf", "make pdf"]):
            return "document_pdf"
        if any(k in p for k in ["generate docx", "create docx", "export docx", "word document"]):
            return "document_docx"
        if any(k in p for k in ["generate xlsx", "create xlsx", "export xlsx", "excel sheet", "spreadsheet"]):
            return "document_xlsx"
        if any(k in p for k in ["generate pptx", "create pptx", "export pptx", "presentation slides", "slide deck"]):
            return "document_pptx"

        # 4. Code execution
        if p.startswith("/run") or p.startswith("/exec") or "execute python" in p or "run this code" in p:
            return "code_execution"

        # 5. Default: Sovereign reasoning & chat
        return "text_chat"

    def dispatch(
        self,
        prompt: str,
        user_id: str = "guest",
        history: Optional[List[Dict[str, str]]] = None,
        extra_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Dispatches request to appropriate specialist engine and returns unified payload.
        """
        intent = self.detect_intent(prompt)
        extra = extra_args or {}

        # 1. Video Studio Dispatch
        if intent == "video_generation":
            clean_p = re.sub(r"^/video\s*", "", prompt, flags=re.I).strip()
            if not clean_p:
                clean_p = "a celestial nebula with cosmic star clusters and orbiting spacecraft"
            v_res = self.video_service.generate_video(
                prompt=clean_p,
                duration_sec=extra.get("duration_sec", 3.0),
                fps=extra.get("fps", 15)
            )
            return {
                "modality": "video",
                "engine": "Moon Video Studio (Spatio-Temporal UNet3D)",
                "result": v_res,
                "reply": f"Generated spatio-temporal video sequence for '{clean_p}'.",
                "artifacts": [v_res.get("video_url")] if v_res.get("video_url") else []
            }

        # 2. Image Studio Dispatch
        if intent == "image_generation":
            clean_p = re.sub(r"^/(?:imagine|image)\s*", "", prompt, flags=re.I).strip()
            if not clean_p:
                clean_p = "cybernetic sovereign laboratory overlooking a futuristic moon base"
            i_res = self.image_service.generate_image(clean_p)
            return {
                "modality": "image",
                "engine": "Moon Image Studio (Latent Diffusion & Procedural)",
                "result": i_res,
                "reply": f"Synthesized procedural artwork for '{clean_p}'.",
                "artifacts": [i_res.get("image_url")] if i_res.get("image_url") else []
            }

        # 3. Document PDF Dispatch
        if intent == "document_pdf":
            title = extra.get("title", "Moon AI Sovereign Report")
            sections = extra.get("sections", [
                {"heading": "Executive Summary", "body": prompt},
                {"heading": "Technical Architecture", "body": "Sovereign multimodal intelligence powered by Moon Core."}
            ])
            pdf_path = self.document_engine.generate_pdf(title=title, sections=sections)
            return {
                "modality": "document",
                "document_type": "pdf",
                "engine": "Sovereign Document Engine",
                "path": pdf_path,
                "reply": f"Generated sovereign PDF document: {os.path.basename(pdf_path)}",
                "artifacts": [pdf_path]
            }

        # 4. Document DOCX Dispatch
        if intent == "document_docx":
            title = extra.get("title", "Moon AI Sovereign Document")
            sections = extra.get("sections", [
                {"heading": "Overview", "body": prompt}
            ])
            docx_path = self.document_engine.generate_docx(title=title, sections=sections)
            return {
                "modality": "document",
                "document_type": "docx",
                "engine": "Sovereign Document Engine",
                "path": docx_path,
                "reply": f"Generated sovereign Word DOCX document: {os.path.basename(docx_path)}",
                "artifacts": [docx_path]
            }

        # 5. Document XLSX Dispatch
        if intent == "document_xlsx":
            sheet_name = extra.get("sheet_name", "Data")
            headers = extra.get("headers", ["Metric", "Value", "Status"])
            rows = extra.get("rows", [
                ["Parameters", "25,698,816", "VERIFIED"],
                ["Policy", "Policy B (Random Scratch)", "PASSED"],
                ["Total Tokens", "445,034", "CERTIFIED"]
            ])
            xlsx_path = self.document_engine.generate_xlsx(sheet_name=sheet_name, headers=headers, rows=rows)
            return {
                "modality": "document",
                "document_type": "xlsx",
                "engine": "Sovereign Document Engine",
                "path": xlsx_path,
                "reply": f"Generated sovereign Excel XLSX workbook: {os.path.basename(xlsx_path)}",
                "artifacts": [xlsx_path]
            }

        # 6. Document PPTX Dispatch
        if intent == "document_pptx":
            title = extra.get("title", "Moon AI Platform Presentation")
            subtitle = extra.get("subtitle", "Autonomous Sovereign Multimodal Intelligence")
            slides = extra.get("slides", [
                {"title": "Core Architecture", "bullets": ["25.7M Sovereign Pilot", "Policy B Gaussian Scratch", "100% Offline Zero-Key"]},
                {"title": "Multimodal Capabilities", "bullets": ["Procedural & Diffusion Images", "Spatio-Temporal Video", "PDF, DOCX, XLSX, PPTX"]}
            ])
            pptx_path = self.document_engine.generate_pptx(title=title, subtitle=subtitle, slides=slides)
            return {
                "modality": "document",
                "document_type": "pptx",
                "engine": "Sovereign Document Engine",
                "path": pptx_path,
                "reply": f"Generated sovereign PowerPoint PPTX presentation: {os.path.basename(pptx_path)}",
                "artifacts": [pptx_path]
            }

        # 7. Code Execution Dispatch
        if intent == "code_execution":
            code_body = re.sub(r"^/(?:run|exec)\s*", "", prompt).strip()
            exec_res = self.code_service.execute_code(code_body)
            return {
                "modality": "code",
                "engine": "Secure Code Interpreter Sandbox",
                "result": exec_res,
                "reply": exec_res.get("output", ""),
                "artifacts": []
            }

        # 8. Default: Text Chat via Moon Core LLM
        chat_res = self.model_service.generate(
            prompt=prompt,
            max_new_tokens=extra.get("max_new_tokens", 150),
            temperature=extra.get("temperature", 0.7),
            history=history
        )
        return {
            "modality": "text",
            "engine": f"Moon Core ({self.model_service.active_model_name})",
            "reply": chat_res.get("reply", ""),
            "model_name": chat_res.get("model_name", ""),
            "tokens_generated": chat_res.get("tokens_generated", 0),
            "device": chat_res.get("device", ""),
            "artifacts": []
        }

