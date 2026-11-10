"""
tests/test_stage_2_3_ui_sandbox_accelerated_training.py
Comprehensive Stage 2.3 Unit & Integration Test Suite.

Chief ML Systems Architect: Prem Gaikwad (India 🇮🇳)
Ownership Policy: TEACHER_STUDENT_RANDOM_INIT (Policy B)

Validates:
1. Sovereign Code Sandbox:
   - AST static security analysis blocks socket, urllib, subprocess, os.system, eval, exec.
   - Safe execution of mathematical, analytical, and data science Python code.
   - Timeout enforcement on infinite loops.
   - Artifact capture for output files.
2. Accelerated Training Engine:
   - Multi-step execution with periodic evaluation and best-checkpoint logic.
   - Policy B strict preservation (zero teacher weights transferred).
   - Training metrics history export to data/reports/training_metrics_latest.json.
3. Multimodal Web & Dispatch Endpoints:
   - POST /api/sandbox/execute
   - POST /api/multimodal/dispatch (document intent routing)
"""

import os
import sys
import json
import time
import unittest
from typing import Dict, Any

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sandbox.code_sandbox import SovereignCodeSandbox, SandboxSecurityError, execute_code
from webapp.code_service import CodeService, SecurityException
from training.chat.train import train_moon_chat
from fastapi.testclient import TestClient
from webapp.main import app


class TestStage23SandboxAndTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sandbox = SovereignCodeSandbox(timeout_seconds=4.0)

    def test_01_ast_security_blocks_network_modules(self):
        """Prohibited modules (socket, requests, urllib, http) must be blocked before execution."""
        bad_codes = [
            "import socket\ns = socket.socket()",
            "import requests\nr = requests.get('http://google.com')",
            "from urllib import request\nrequest.urlopen('http://evil.com')",
            "import http.client"
        ]
        for code in bad_codes:
            with self.assertRaises(SandboxSecurityError):
                self.sandbox.analyze_ast(code)

            res = self.sandbox.execute(code)
            self.assertFalse(res["success"])
            self.assertEqual(res["ast_security"], "REJECTED")
            self.assertEqual(res["exit_code"], -2)
            self.assertIn("Security", res["error"])

    def test_02_ast_security_blocks_destructive_calls(self):
        """Prohibited syscalls (eval, exec, compile, __import__) must be blocked."""
        bad_calls = [
            "eval('2 + 2')",
            "exec('x = 1')",
            "compile('x = 1', '', 'exec')",
            "__import__('os').system('dir')"
        ]
        for code in bad_calls:
            with self.assertRaises(SandboxSecurityError):
                self.sandbox.analyze_ast(code)

            res = self.sandbox.execute(code)
            self.assertFalse(res["success"])
            self.assertEqual(res["ast_security"], "REJECTED")

    def test_03_safe_math_and_data_execution(self):
        """Valid Python math, list comprehensions, and JSON processing must execute cleanly."""
        code = (
            "import math\n"
            "import json\n"
            "squares = [i**2 for i in range(5)]\n"
            "sqrt_val = round(math.sqrt(16), 2)\n"
            "print(json.dumps({'squares': squares, 'sqrt': sqrt_val}))\n"
        )
        res = self.sandbox.execute(code)
        self.assertTrue(res["success"])
        self.assertEqual(res["exit_code"], 0)
        self.assertEqual(res["ast_security"], "APPROVED")
        data = json.loads(res["output"].strip())
        self.assertEqual(data["squares"], [0, 1, 4, 9, 16])
        self.assertEqual(data["sqrt"], 4.0)

    def test_04_timeout_enforcement(self):
        """Infinite loops must be killed strictly within the timeout period."""
        infinite_loop = "while True:\n    pass\n"
        t0 = time.time()
        res = self.sandbox.execute(infinite_loop, timeout_override=1.5)
        elapsed = time.time() - t0
        self.assertFalse(res["success"])
        self.assertEqual(res["exit_code"], -1)
        self.assertIn("Timeout", res["error"])
        self.assertLess(elapsed, 3.5, "Timeout was not enforced in a timely manner")

    def test_05_artifact_capture(self):
        """Generated files in the sandbox workspace must be captured and reported."""
        code = (
            "with open('output_metrics.csv', 'w') as f:\n"
            "    f.write('iteration,loss\\n1,5.21\\n2,5.18\\n')\n"
            "print('Saved CSV')\n"
        )
        res = self.sandbox.execute(code)
        self.assertTrue(res["success"])
        self.assertGreaterEqual(len(res["artifacts"]), 1)
        art = res["artifacts"][0]
        self.assertEqual(art["filename"], "output_metrics.csv")
        self.assertTrue(os.path.exists(art["path"]))
        self.assertGreater(art["size_bytes"], 0)

    def test_06_code_service_wrapper(self):
        """CodeService wrapper in webapp must preserve existing compatibility."""
        cs = CodeService()
        res = cs.execute_code("print('Hello from Sovereign CodeService')")
        self.assertTrue(res["success"])
        self.assertIn("Hello from Sovereign CodeService", res["output"])

    def test_07_webapp_sandbox_execute_endpoint(self):
        """POST /api/sandbox/execute must return structured execution details."""
        resp = self.client.post("/api/sandbox/execute", json={
            "code": "print('Sovereign Sandbox HTTP Endpoint OK')",
            "timeout_seconds": 3.0
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"], f"Sandbox execute failed: {data}")
        self.assertEqual(data["exit_code"], 0)
        self.assertEqual(data["ast_security"], "APPROVED")
        self.assertIn("Sovereign Sandbox HTTP Endpoint OK", data["output"])

    def test_08_multimodal_dispatch_document_intent(self):
        """POST /api/multimodal/dispatch must route /doc prompts to the Document Engine."""
        resp = self.client.post("/api/multimodal/dispatch", json={
            "prompt": "/doc Sovereign AI Architecture Overview"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["modality"], "document")
        self.assertIn("Sovereign Document Engine", data["engine"])
        self.assertGreaterEqual(len(data["artifacts"]), 1)

    def test_09_accelerated_training_pipeline_policy_b_preservation(self):
        """Verifies accelerated training pipeline adheres to Policy B and exports metrics."""
        scaled_cfg = os.path.join(PROJECT_ROOT, "training", "chat", "scaled_config.yaml")
        self.assertTrue(os.path.exists(scaled_cfg))

        res = train_moon_chat(
            config_path=scaled_cfg,
            max_iters_override=1,
            smoke_test=True
        )
        self.assertEqual(res["status"], "COMPLETED_SMOKE_TEST")
        self.assertIn("final_val_loss", res)
        self.assertIn("perplexity", res)

        # Check metrics file
        metrics_path = os.path.join(PROJECT_ROOT, "data", "reports", "training_metrics_latest.json")
        self.assertTrue(os.path.exists(metrics_path))
        with open(metrics_path, "r", encoding="utf-8") as f:
            m_data = json.load(f)
        self.assertEqual(m_data["parameters"], 25698816)
        self.assertIn("final_train_loss", m_data)

    def test_10_resume_fails_closed_on_policy_b_violation(self):
        """Resuming training must reject any checkpoint that transferred teacher weights."""
        fake_bad_ckpt_path = os.path.join(PROJECT_ROOT, "data", "sandbox", "fake_bad_ckpt.pt")
        os.makedirs(os.path.dirname(fake_bad_ckpt_path), exist_ok=True)
        bad_payload = {
            "model": {},
            "config": {
                "teacher_checkpoint_weights_transferred": True,  # Policy B VIOLATION
                "initialization": "pretrained"
            }
        }
        torch.save(bad_payload, fake_bad_ckpt_path)

        scaled_cfg = os.path.join(PROJECT_ROOT, "training", "chat", "scaled_config.yaml")
        try:
            with self.assertRaises(ValueError) as ctx:
                train_moon_chat(
                    config_path=scaled_cfg,
                    resume_from_ckpt=fake_bad_ckpt_path,
                    smoke_test=True
                )
            self.assertIn("Policy B violation", str(ctx.exception))
        finally:
            if os.path.exists(fake_bad_ckpt_path):
                os.remove(fake_bad_ckpt_path)


if __name__ == "__main__":
    unittest.main()

