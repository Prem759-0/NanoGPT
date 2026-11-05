"""
tests/test_stage_2_2_scaled_training_and_multimodal.py
Unit & Integration Verification for Moon AI Stage 2.2:
1. Sovereign Scaled Model Checkpoint (moon_chat_scaled_v1.pt) & Policy B Attestation.
2. Sovereign Document Intelligence Engine (PDF, DOCX, XLSX, PPTX generation & parsing).
3. Unified Moon Core Multimodal Coordinator & Dispatch.
4. FastAPI Multimodal & Document REST API endpoints.

Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT (Policy B)
"""

import os
import sys
import json
import unittest
from typing import Dict, Any

import torch
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from webapp.main import app
from multimodal.document_engine import SovereignDocumentEngine
from multimodal.coordinator import MoonMultimodalCoordinator

SCALED_CKPT_PATH = os.path.join(PROJECT_ROOT, "models", "checkpoints", "chat", "moon_chat_scaled_v1.pt")


class TestStage22ScaledTrainingAndMultimodal(unittest.TestCase):
    """Test suite certifying Stage 2.2 platform deliverables."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.doc_engine = SovereignDocumentEngine()
        cls.coordinator = MoonMultimodalCoordinator(document_engine=cls.doc_engine)

    def test_01_scaled_training_checkpoint_policy_b_attestation(self):
        """Certifies that moon_chat_scaled_v1.pt exists with strict Policy B attestation."""
        self.assertTrue(os.path.isfile(SCALED_CKPT_PATH), f"Scaled checkpoint not found at: {SCALED_CKPT_PATH}")
        ckpt = torch.load(SCALED_CKPT_PATH, map_location="cpu", weights_only=False)
        meta = ckpt.get("config", {}) or ckpt.get("metadata", {})

        self.assertFalse(meta.get("teacher_checkpoint_weights_transferred", True),
                         "Policy B violation: teacher_checkpoint_weights_transferred must be False.")
        self.assertEqual(meta.get("initialization"), "random_gaussian_scratch")
        self.assertEqual(meta.get("ownership_policy"), "TEACHER_STUDENT_RANDOM_INIT")
        self.assertIn("final_val_loss", meta)
        self.assertIn("perplexity", meta)
        self.assertLess(meta["final_val_loss"], 6.0)
        self.assertLess(meta["perplexity"], 350.0)

    def test_02_document_engine_pdf_generation_and_parsing(self):
        """Certifies generating and parsing native PDF-1.4 documents."""
        sections = [
            {"heading": "Executive Summary", "body": "Moon AI sovereign document synthesis."},
            {"heading": "Telemetry", "table": {"headers": ["Metric", "Value"], "rows": [["Loss", "5.66"], ["PPL", "289.2"]]}}
        ]
        pdf_path = self.doc_engine.generate_pdf(title="Quantum Report", sections=sections)
        self.assertTrue(os.path.isfile(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 500)

        # Parse back
        parsed = self.doc_engine.parse_document(pdf_path)
        self.assertEqual(parsed["extension"], ".pdf")
        self.assertIn("Quantum Report", parsed["text"])
        self.assertIn("Executive Summary", parsed["text"])

    def test_03_document_engine_docx_generation_and_parsing(self):
        """Certifies generating and parsing Microsoft Word (.docx) OpenXML archives."""
        sections = [
            {"heading": "Abstract", "body": "Pure python OpenXML DOCX container."},
            {"heading": "Data", "table": {"headers": ["ID", "Score"], "rows": [["1", "100"], ["2", "95"]]}}
        ]
        docx_path = self.doc_engine.generate_docx(title="Sovereign Architecture", sections=sections)
        self.assertTrue(os.path.isfile(docx_path))
        self.assertGreater(os.path.getsize(docx_path), 500)

        # Parse back
        parsed = self.doc_engine.parse_document(docx_path)
        self.assertEqual(parsed["extension"], ".docx")
        self.assertIn("Sovereign Architecture", parsed["text"])
        self.assertIn("Abstract", parsed["text"])

    def test_04_document_engine_xlsx_generation_and_parsing(self):
        """Certifies generating and parsing Microsoft Excel (.xlsx) OpenXML spreadsheets."""
        headers = ["Domain", "Records", "Tokens"]
        rows = [
            ["coding", 350, 140000],
            ["mathematics", 200, 80000],
            ["science", 200, 85000]
        ]
        xlsx_path = self.doc_engine.generate_xlsx(sheet_name="CurriculumStats", headers=headers, rows=rows)
        self.assertTrue(os.path.isfile(xlsx_path))
        self.assertGreater(os.path.getsize(xlsx_path), 500)

        # Parse back
        parsed = self.doc_engine.parse_document(xlsx_path)
        self.assertEqual(parsed["extension"], ".xlsx")
        self.assertGreaterEqual(len(parsed["tables"]), 1)
        self.assertEqual(parsed["tables"][0]["headers"], headers)

    def test_05_document_engine_pptx_generation(self):
        """Certifies generating Microsoft PowerPoint (.pptx) OpenXML presentation slides."""
        slides = [
            {"title": "Overview", "bullets": ["First Principles AI", "Zero Third Party APIs"]},
            {"title": "Multimodal Specs", "bullets": ["Document Engine", "Latent Diffusion", "Spatio-Temporal Video"]}
        ]
        pptx_path = self.doc_engine.generate_pptx(title="Moon AI Sovereign", subtitle="Stage 2.2", slides=slides)
        self.assertTrue(os.path.isfile(pptx_path))
        self.assertGreater(os.path.getsize(pptx_path), 1000)

    def test_06_multimodal_coordinator_intent_detection(self):
        """Certifies accurate intent classification across multimodal prompts."""
        self.assertEqual(self.coordinator.detect_intent("generate pdf report on system design"), "document_pdf")
        self.assertEqual(self.coordinator.detect_intent("export docx document for project plan"), "document_docx")
        self.assertEqual(self.coordinator.detect_intent("generate xlsx spreadsheet with quarterly results"), "document_xlsx")
        self.assertEqual(self.coordinator.detect_intent("create pptx presentation slides"), "document_pptx")
        self.assertEqual(self.coordinator.detect_intent("/imagine a futuristic space station"), "image_generation")
        self.assertEqual(self.coordinator.detect_intent("/video soaring over cybernetic canyon"), "video_generation")
        self.assertEqual(self.coordinator.detect_intent("/run print(42 * 2)"), "code_execution")
        self.assertEqual(self.coordinator.detect_intent("Explain how transformers calculate self-attention"), "text_chat")

    def test_07_multimodal_coordinator_document_dispatch(self):
        """Certifies multimodal coordinator end-to-end dispatch for documents."""
        res_pdf = self.coordinator.dispatch(prompt="generate pdf summary of distributed consensus")
        self.assertEqual(res_pdf["modality"], "document")
        self.assertEqual(res_pdf["document_type"], "pdf")
        self.assertTrue(os.path.isfile(res_pdf["path"]))

        res_xlsx = self.coordinator.dispatch(prompt="create xlsx spreadsheet with benchmarks")
        self.assertEqual(res_xlsx["modality"], "document")
        self.assertEqual(res_xlsx["document_type"], "xlsx")
        self.assertTrue(os.path.isfile(res_xlsx["path"]))

    def test_08_webapp_document_api_endpoints(self):
        """Certifies FastAPI document generation and parsing HTTP endpoints."""
        r_gen = self.client.post("/api/document/generate", json={
            "doc_type": "pdf",
            "title": "API Test PDF",
            "sections": [{"heading": "Header 1", "body": "Body content via REST API."}]
        })
        self.assertEqual(r_gen.status_code, 200)
        data = r_gen.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["doc_type"], "pdf")
        file_path = data["file_path"]

        # Parse via endpoint
        r_parse = self.client.post("/api/document/parse", json={"filepath": file_path})
        self.assertEqual(r_parse.status_code, 200)
        parse_data = r_parse.json()
        self.assertEqual(parse_data["extension"], ".pdf")
        self.assertIn("API Test PDF", parse_data["text"])

    def test_09_webapp_multimodal_dispatch_endpoint(self):
        """Certifies FastAPI unified multimodal dispatch HTTP endpoint."""
        r_disp = self.client.post("/api/multimodal/dispatch", json={
            "prompt": "generate pdf report on machine learning"
        })
        self.assertEqual(r_disp.status_code, 200)
        disp_data = r_disp.json()
        self.assertEqual(disp_data["modality"], "document")
        self.assertEqual(disp_data["engine"], "Sovereign Document Engine")
        self.assertGreater(len(disp_data["artifacts"]), 0)


if __name__ == "__main__":
    unittest.main()

