"""Tests for FastAPI routes."""
import unittest
from fastapi.testclient import TestClient
from app.interface.fastapi_main import app

class TestFastAPIRoutes(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("NeuroAct AI is running", response.json()["message"])