#!/usr/bin/env python3
"""
NeuroAct AI - Multi-Agent Data Science Workflow System
Main application entry point
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from app.agents.planner_agent import PlannerAgent
from app.mcp.mcp_schema import TaskMessage
from app.utils.logger import get_logger

logger = get_logger(__name__)

class NeuroActApp:
    """Main NeuroAct application"""
    
    def __init__(self):
        self.planner = PlannerAgent()
        logger.info("NeuroAct AI initialized successfully")
    
    def process_query(self, query: str, data_path: str = None) -> Dict[str, Any]:
        """Process user query through the multi-agent system"""
        try:
            # Create task message
            task = TaskMessage.create(
                sender="user",
                receiver="PlannerAgent",
                agent_role="PlannerAgent",
                inputs={
                    "query": query,
                    "data_path": data_path
                },
                dependencies=[]
            )
            
            # Process through planner
            result = self.planner.execute(task)
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "status": "error",
                "errors": [str(e)],
                "query": query
            }
    
    def interactive_mode(self):
        """Run NeuroAct in interactive CLI mode"""
        print("NeuroAct AI - Multi-Agent Data Science System")
        print("=" * 50)
        print("Type your data science queries in natural language")
        print("Commands: 'quit' to exit, 'help' for examples")
        print("=" * 50)
        
        while True:
            try:
                query = input("\nNeuroAct> ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if query.lower() == 'help':
                    self.show_help()
                    continue
                
                if not query:
                    continue
                
                print("\nProcessing...")
                result = self.process_query(query)
                
                # Display result
                self.display_result(result)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def show_help(self):
        """Show example queries"""
        examples = [
            "Load and analyze data from sales.csv",
            "Build a machine learning model for prediction",
            "Create a sentiment analysis for customer reviews",
            "Generate Python code for data cleaning",
            "Visualize the correlation between features",
            "Evaluate model performance with cross-validation"
        ]
        
        print("\nExample Queries:")
        for i, example in enumerate(examples, 1):
            print(f"  {i}. {example}")
    
    def display_result(self, result: Dict[str, Any]):
        """Display formatted result"""
        if result.get("status") == "success":
            print("Success!")
            
            # Show which agent handled the query
            if "selected_agent" in result:
                print(f"Handled by: {result['selected_agent']}")
            
            # Show the main result
            if "result" in result:
                agent_result = result["result"]
                if isinstance(agent_result, dict):
                    if "message" in agent_result:
                        print(f"Message: {agent_result['message']}")
                    if "action" in agent_result:
                        print(f"Action: {agent_result['action']}")
                else:
                    print(f"Result: {agent_result}")
            
            # Show confidence score if available
            if "confidence" in result:
                print(f"Confidence: {result['confidence']:.2f}")
        
        else:
            print("Error occurred:")
            errors = result.get("errors", ["Unknown error"])
            for error in errors:

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Command line mode
        query = " ".join(sys.argv[1:])
        app = NeuroActApp()
        result = app.process_query(query)
        app.display_result(result)
    else:
        # Interactive mode
        app = NeuroActApp()
        app.interactive_mode()

if __name__ == "__main__":
    main()