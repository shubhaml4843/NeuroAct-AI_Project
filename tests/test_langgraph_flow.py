"""Tests for LangGraph workflow."""
import unittest
from app.core.langgraph_flow import LangGraphFlow

class TestLangGraphFlow(unittest.TestCase):
    def test_workflow_execution(self):
        flow = LangGraphFlow()
        result = flow.execute_workflow({"task": "test"})
        self.assertEqual(result["status"], "completed")