""" Smart Critic Agent - Routes queries to specialized agents"""
from app.mcp.mcp_schema import TaskMessage
from app.utils.logger import get_logger, log_execution_time
from typing import Dict, Any, List, Optional
from app.config import get_ollama_config

logger = get_logger(__name__)

class CriticAgent:
    """Smart routing agent that delegates to specialized agents based on query"""
    def __init__(self):
        self.name = "CriticAgent"
        self.ollama_config = get_ollama_config()
        logger.info(f"Initialized {self.name}")

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

    def _route_to_code_agent(self, task: TaskMessage) -> Dict[str, Any]:
        """Route to CodeAgent for code review"""
        try:
            from app.agents.code_agent import CodeAgent
            
            code_agent = CodeAgent()
            
            # Extract code from inputs
            code_content = task.inputs.get("code_content", task.inputs.get("code", ""))
            
            # Determine the specific action based on query
            query = task.inputs.get("query", "").lower()
            
            if "review" in query or "check" in query or "analyze" in query:
                # Use CodeAgent's review_code method directly
                result = code_agent.review_code({"code": code_content})
            elif "generate" in query or "create" in query:
                # Use CodeAgent's generate_code method
                result = code_agent.generate_code(task.inputs)
            elif "execute" in query or "run" in query:
                # Use CodeAgent's execute_code method
                result = code_agent.execute_code({"code": code_content})
            else:
                # Default to review if code is provided
                if code_content:
                    result = code_agent.review_code({"code": code_content})
                else:
                    result = code_agent.execute(task)
            
            return {
                "status": "success",
                "routed_to": "CodeAgent",
                "analysis_type": "Code Analysis",
                "result": result,
                "message": f"Code analysis completed by CodeAgent: {result.get('action', 'unknown')}"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [f"CodeAgent routing failed: {str(e)}"]}

    def _route_to_model_evaluation_agent(self, task: TaskMessage) -> Dict[str, Any]:
        """Route to ModelEvaluationAgent for model review"""
        try:
            from app.agents.model_evaluation_agent import ModelEvaluationAgent
            
            eval_agent = ModelEvaluationAgent()
            
            # Modify task for model evaluation
            eval_task = TaskMessage(
                task_id=task.task_id,
                agent_role="ModelEvaluationAgent",
                inputs=task.inputs,
                dependencies=task.dependencies
            )
            
            result = eval_agent.execute(eval_task)
            
            return {
                "status": "success",
                "routed_to": "ModelEvaluationAgent",
                "analysis_type": "Model Evaluation",
                "result": result,
                "message": "Model evaluation completed by ModelEvaluationAgent"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [f"ModelEvaluationAgent routing failed: {str(e)}"]}

    def _route_to_data_agent(self, task: TaskMessage) -> Dict[str, Any]:
        """Route to DataAgent for data review"""
        try:
            from app.agents.data_agent import DataAgent
            
            data_agent = DataAgent()
            
            # Modify task for data analysis
            data_task = TaskMessage(
                task_id=task.task_id,
                agent_role="DataAgent",
                inputs={
                    **task.inputs,
                    "query": f"analyze {task.inputs.get('query', '')}"
                },
                dependencies=task.dependencies
            )
            
            result = data_agent.execute(data_task)
            
            return {
                "status": "success",
                "routed_to": "DataAgent",
                "analysis_type": "Data Analysis",
                "result": result,
                "message": "Data analysis completed by DataAgent"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [f"DataAgent routing failed: {str(e)}"]}

    def _route_to_nlp_agent(self, task: TaskMessage) -> Dict[str, Any]:
        """Route to NLPAgent for NLP review"""
        try:
            from app.agents.nlp_agent import NLPAgent
            
            nlp_agent = NLPAgent()
            
            # Modify task for NLP analysis
            nlp_task = TaskMessage(
                task_id=task.task_id,
                agent_role="NLPAgent",
                inputs=task.inputs,
                dependencies=task.dependencies
            )
            
            result = nlp_agent.execute(nlp_task)
            
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
        """Route to VisualizationAgent for visualization review"""
        try:
            from app.agents.VisualizationAgent import VisualizationAgent
            
            viz_agent = VisualizationAgent()
            
            # Modify task for visualization
            viz_task = TaskMessage(
                task_id=task.task_id,
                agent_role="VisualizationAgent",
                inputs=task.inputs,
                dependencies=task.dependencies
            )
            
            result = viz_agent.execute(viz_task)
            
            return {
                "status": "success",
                "routed_to": "VisualizationAgent",
                "analysis_type": "Visualization Analysis",
                "result": result,
                "message": "Visualization analysis completed by VisualizationAgent"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [f"VisualizationAgent routing failed: {str(e)}"]}

    def _comprehensive_review(self, task: TaskMessage) -> Dict[str, Any]:
        """Perform comprehensive review using multiple agents"""
        try:
            query = task.inputs.get("query", "")
            results = {}
            
            # Try multiple agents based on available inputs
            if task.inputs.get("code_content") or task.inputs.get("code_path"):
                code_result = self._route_to_code_agent(task)
                if code_result.get("status") == "success":
                    results["code_review"] = code_result
            
            if task.inputs.get("model_metrics") or task.inputs.get("model"):
                model_result = self._route_to_model_evaluation_agent(task)
                if model_result.get("status") == "success":
                    results["model_evaluation"] = model_result
            
            if task.inputs.get("data_path") or task.inputs.get("dataframe"):
                data_result = self._route_to_data_agent(task)
                if data_result.get("status") == "success":
                    results["data_analysis"] = data_result
            
            # Generate comprehensive summary
            total_agents = len(results)
            successful_reviews = sum(1 for r in results.values() if r.get("status") == "success")
            
            return {
                "status": "success",
                "analysis_type": "Comprehensive Review",
                "agents_used": list(results.keys()),
                "total_agents": total_agents,
                "successful_reviews": successful_reviews,
                "results": results,
                "message": f"Comprehensive review completed using {total_agents} specialized agents"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [f"Comprehensive review failed: {str(e)}"]}

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary from multiple agent results"""
        summary = {
            "overall_status": "success",
            "agents_used": [],
            "key_findings": [],
            "recommendations": []
        }
        
        for agent_name, result in results.items():
            if result.get("status") == "success":
                summary["agents_used"].append(agent_name)
                
                # Extract key findings
                if "issues" in result.get("result", {}):
                    issues = result["result"]["issues"]
                    if issues:
                        summary["key_findings"].append(f"{agent_name}: Found {len(issues)} issues")
                
                # Extract recommendations
                if "recommendations" in result.get("result", {}):
                    recs = result["result"]["recommendations"]
                    summary["recommendations"].extend(recs)
        
        return summary