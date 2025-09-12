""" Smart Critic Agent - Routes queries to specialized agents"""
from app.mcp.mcp_schema import TaskMessage
from app.utils.logger import get_logger, log_execution_time
from app.config import get_ollama_config
from typing import Dict, Any, List, Optional
import subprocess
from pathlib import Path

logger = get_logger(__name__)

class CriticAgent:
    """Smart routing agent that delegates to specialized agents based on query"""
    def __init__(self):
        self.name = "CriticAgent"
        self.ollama_config = get_ollama_config()
        self.available_tools = self._check_available_tools()
        logger.info(f"Initialized {self.name} with tools: {list(self.available_tools.keys())}")
        
    def _check_available_tools(self) -> Dict[str, bool]:
        """Check which external tools are available"""
        tools = {}
        for tool in ["flake8", "bandit", "radon", "pylint"]:
            try:
                subprocess.run([tool, "--version"], capture_output=True, timeout=5)
                tools[tool] = True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                tools[tool] = False
                logger.warning(f"Tool {tool} not available")
        return tools
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities and available tools"""
        return {
            "name": self.name,
            "description": "Smart routing agent for multi-agent coordination",
            "capabilities": [
                "Query intent analysis",
                "Agent routing and selection", 
                "Multi-agent coordination",
                "Comprehensive analysis"
            ],
            "available_tools": self.available_tools,
            "supported_intents": [
                "code_review", "model_review", "data_review", 
                "nlp_review", "visualization_review", "comprehensive"
            ]
        }
    
    @log_execution_time
    def execute(self, task: TaskMessage) -> Dict[str, Any]:
        """Route query to appropriate specialized agent"""
        try:
            query = task.inputs.get("query", "").lower()
            
            # Analyze intent and route to appropriate agent
            intent = self._analyze_intent(query)
            
            if intent == "code_review":
                return self._route_to_code_agent(task)
            elif intent == "model_review":
                return self._route_to_model_evaluation_agent(task)
            elif intent == "data_review":
                return self._route_to_data_agent(task)
            elif intent == "nlp_review":
                return self._route_to_nlp_agent(task)
            elif intent == "visualization_review":
                return self._route_to_visualization_agent(task)
            else:
                return self._comprehensive_review(task)
                
        except Exception as e:
            return {"status": "error", "errors": [str(e)], "agent": self.name}

    def _analyze_intent(self, query: str) -> str:
        """Analyze query to determine which agent should handle it"""
        
        # Code-related queries (expanded keywords)
        code_keywords = ["code", "function", "class", "script", "syntax", "bug", "refactor", "review code", 
                        "check code", "analyze code", "security", "vulnerability", "performance", "style",
                        "generate code", "create code", "execute code", "run code"]
        if any(keyword in query for keyword in code_keywords):
            return "code_review"
        
        # Model-related queries  
        model_keywords = ["model", "accuracy", "overfitting", "metrics", "evaluate", "optimize", "tune",
                         "benchmark", "hyperparameter", "cross validation"]
        if any(keyword in query for keyword in model_keywords):
            return "model_review"
        
        # Data-related queries
        data_keywords = ["data", "dataset", "clean", "preprocess", "outlier", "missing", "eda", "explore"]
        if any(keyword in query for keyword in data_keywords):
            return "data_review"
        
        # NLP-related queries
        nlp_keywords = ["text", "sentiment", "nlp", "embedding", "entity", "summarize", "classify text", "topic"]
        if any(keyword in query for keyword in nlp_keywords):
            return "nlp_review"
        
        # Visualization-related queries
        viz_keywords = ["plot", "chart", "graph", "visualize", "dashboard", "histogram", "scatter"]
        if any(keyword in query for keyword in viz_keywords):
            return "visualization_review"
        
        return "comprehensive"

    def _route_to_data_agent(self, task: TaskMessage) -> Dict[str, Any]:
        """Route to DataAgent for data analysis"""
        try:
            from app.agents.data_agent import DataAgent
            
            data_agent = DataAgent()
            result = data_agent.execute(task)
            
            return {
                "status": "success",
                "routed_to": "DataAgent",
                "analysis_type": "Data Analysis",
                "result": result,
                "message": "Data analysis completed by DataAgent"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [f"DataAgent routing failed: {str(e)}"]}

    def _route_to_code_agent(self, task: TaskMessage) -> Dict[str, Any]:
        """Route to CodeAgent for code analysis"""
        try:
            from app.agents.code_agent import CodeAgent
            
            code_agent = CodeAgent()
            result = code_agent.execute(task)
            
            return {
                "status": "success",
                "routed_to": "CodeAgent",
                "analysis_type": "Code Analysis",
                "result": result,
                "message": "Code analysis completed by CodeAgent"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [f"CodeAgent routing failed: {str(e)}"]}

    def _route_to_model_evaluation_agent(self, task: TaskMessage) -> Dict[str, Any]:
        """Route to ModelEvaluationAgent for model analysis"""
        try:
            from app.agents.model_evaluation_agent import ModelEvaluationAgent
            
            eval_agent = ModelEvaluationAgent()
            result = eval_agent.execute(task)
            
            return {
                "status": "success",
                "routed_to": "ModelEvaluationAgent",
                "analysis_type": "Model Evaluation",
                "result": result,
                "message": "Model evaluation completed by ModelEvaluationAgent"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [f"ModelEvaluationAgent routing failed: {str(e)}"]}

    def _route_to_nlp_agent(self, task: TaskMessage) -> Dict[str, Any]:
        """Route to NLPAgent for NLP analysis"""
        try:
            from app.agents.nlp_agent import NLPAgent
            
            nlp_agent = NLPAgent()
            result = nlp_agent.execute(task)
            
            return {
                "status": "success",
                "routed_to": "NLPAgent",
                "analysis_type": "NLP Analysis",
                "result": result,
                "message": "NLP analysis completed by NLPAgent"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [f"NLPAgent routing failed: {str(e)}"]}

    def _route_to_visualization_agent(self, task: TaskMessage) -> Dict[str, Any]:
        """Route to VisualizationAgent for plotting and charts"""
        try:
            from app.agents.VisualizationAgent import VisualizationAgent
            from app.agents.data_agent import DataAgent
            
            # First get statistical analysis from DataAgent if needed
            query = task.inputs.get("query", "").lower()
            statistical_results = {}
            
            if any(word in query for word in ['normal', 'normality', 'outlier', 'distribution']):
                data_agent = DataAgent()
                data_result = data_agent.perform_normality_and_outlier_analysis(
                    task.inputs.get("data", []), 
                    task.inputs.get("column")
                )
                if data_result.get("status") == "success":
                    statistical_results = data_result.get("statistical_analysis", {})
            
            # Add statistical results to task inputs
            enhanced_inputs = task.inputs.copy()
            enhanced_inputs["statistical_analysis"] = statistical_results
            
            # Create enhanced task
            enhanced_task = TaskMessage(
                task_id=task.task_id,
                inputs=enhanced_inputs
            )
            
            # Route to VisualizationAgent
            viz_agent = VisualizationAgent()
            result = viz_agent.execute(enhanced_task)
            
            return {
                "status": "success",
                "routed_to": "VisualizationAgent",
                "analysis_type": "Data Visualization with Statistical Analysis",
                "result": result,
                "statistical_context": statistical_results,
                "message": "Visualization with statistical analysis completed"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [f"Visualization routing failed: {str(e)}"]}

    def _comprehensive_review(self, task: TaskMessage) -> Dict[str, Any]:
        """Perform comprehensive analysis using multiple agents"""
        try:
            query = task.inputs.get("query", "").lower()
            
            # If query involves visualization, route to visualization agent
            if any(word in query for word in ['plot', 'chart', 'graph', 'visualize']):
                return self._route_to_visualization_agent(task)
            else:
                # Otherwise route to DataAgent as primary handler
                return self._route_to_data_agent(task)
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)], "agent": self.name}