import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from webapp.main import app

class TestSuperAIWebApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_endpoint(self):
        """Verify GET /api/health reports active server and PyTorch readiness."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["checkpoint_exists"])
        self.assertTrue(data["torch_available"])

    def test_02_model_info_endpoint(self):
        """Verify GET /api/model-info returns valid metadata."""
        response = self.client.get("/api/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(
            any(k in data["model_type"].lower() for k in ["nanogpt", "moon ai", "frontier"]),
            f"Unexpected model_type: {data['model_type']}"
        )
        self.assertIsNotNone(data["checkpoint_path"])

    def test_03_empty_message_rejected(self):
        """Verify POST /api/chat rejects empty messages with 422."""
        res = self.client.post("/api/chat", json={"message": "   "})
        self.assertEqual(res.status_code, 422)

    def test_04_oversized_message_rejected(self):
        """Verify POST /api/chat rejects oversized messages with 422."""
        res = self.client.post("/api/chat", json={"message": "x" * 1050})
        self.assertEqual(res.status_code, 422)

    def test_05_chat_generates_response(self):
        """Verify POST /api/chat generates neural response and persists session."""
        res = self.client.post("/api/chat", json={"message": "Hello", "max_new_tokens": 30})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("reply", data)
        self.assertIn("session_id", data)

    def test_06_image_generation_endpoint(self):
        """Verify POST /api/generate-image generates local artwork."""
        res = self.client.post("/api/generate-image", json={"prompt": "cyberpunk laboratory"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["image_url"].startswith("/static/generated/"))

    def test_07_code_runner_endpoint(self):
        """Verify POST /api/run-code executes Python sandbox code."""
        res = self.client.post("/api/run-code", json={"code": "print(100 + 200)"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["output"].strip(), "300")

    def test_08_sessions_endpoint(self):
        """Verify GET /api/sessions returns conversation history from SQLite."""
        res = self.client.get("/api/sessions")
        self.assertEqual(res.status_code, 200)
        sessions = res.json()
        self.assertIsInstance(sessions, list)

    def test_09_frontend_serves_html(self):
        """Verify GET / serves frontend."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(
            any(k in res.text for k in ["Moon AI", "NanoGPT Lab", "root"]),
            "Frontend HTML should contain Moon AI, NanoGPT Lab, or root mount div"
        )

if __name__ == "__main__":
    unittest.main()
