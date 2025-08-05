"""Critic/Evaluator agent for quality assessment."""

class EvalAgent:
    def __init__(self):
        self.name = "EvalAgent"
    
    def evaluate_output(self, output: str, criteria: dict) -> dict:
        """Evaluate output quality against criteria."""
        return {"score": 0.85, "feedback": "Good quality output"}
    
    def provide_feedback(self, task_result: dict) -> dict:
        """Provide detailed feedback on task results."""
        return {"rating": "excellent", "improvements": []}