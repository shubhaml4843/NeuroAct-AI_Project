"""Unit tests for agents."""
import unittest
from app.agents.code_agent import CodeAgent
from app.agents.ml_agent import MLAgent

class TestAgents(unittest.TestCase):
    def test_code_agent(self):
        agent = CodeAgent()
        result = agent.generate_code("print hello world")
        self.assertIn("Generated code", result)
    
    def test_ml_agent(self):
        agent = MLAgent()
        result = agent.train_model("data.csv", {})
        self.assertEqual(result["status"], "training_started")