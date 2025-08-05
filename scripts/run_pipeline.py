"""CLI runner for local testing."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.langgraph_flow import LangGraphFlow
from app.agents.code_agent import CodeAgent

def main():
    print("Starting NeuroAct AI Pipeline...")
    
    # Initialize workflow
    flow = LangGraphFlow()
    code_agent = CodeAgent()
    flow.add_agent("code", code_agent)
    
    # Execute sample task
    result = flow.execute_workflow({"task": "Generate hello world code"})
    print(f"Result: {result}")

if __name__ == "__main__":
    main()