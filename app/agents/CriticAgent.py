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
                "ai_issues": ai_response.get("issues", []),
                "ai_suggestions": ai_response.get("suggestions", []),
                "ai_quality_score": ai_response.get("score", 70),
                "llm_model": f"ollama-{OLLAMA_CONFIG.get('model', 'llama2')}",
                "analysis_context": agent_name
            }
            
        except Exception as e:
            logger.warning(f"AI review failed: {str(e)}")
            return {
                "ai_issues": [],
                "ai_suggestions": ["AI review unavailable - check Ollama service"],
                "ai_quality_score": 0,
                "error": str(e)
            }
    


    def _query_llm(self, prompt: str) -> Dict[str, Any]:
        """Query Ollama LLM using existing configuration."""
        try:
            import requests
            
            # Use existing Ollama configuration
            ollama_url = f"{OLLAMA_CONFIG['base_url']}/api/generate"
            
            payload = {
                "model": OLLAMA_CONFIG.get("model", "llama2"),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": OLLAMA_CONFIG.get("temperature", 0.1),
                    "top_p": OLLAMA_CONFIG.get("top_p", 0.9),
                    "max_tokens": OLLAMA_CONFIG.get("max_tokens", 500)
                }
            }
            
            response = requests.post(ollama_url, json=payload, timeout=OLLAMA_CONFIG.get("timeout", 30))
            
            if response.status_code == 200:
                result = response.json()
                llm_output = result.get("response", "")
                
                # Parse LLM response (expecting JSON format)
                try:
                    import json
                    parsed_response = json.loads(llm_output)
                    return parsed_response
                except json.JSONDecodeError:
                    return self._fallback_response()
            else:
                logger.warning(f"Ollama API error: {response.status_code}")
                return self._fallback_response()
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama connection failed: {str(e)}")
            return self._fallback_response()
        except Exception as e:
            return {"status": "error", "errors": [f"Comprehensive review failed: {str(e)}"]}

    def _generate_code_fixes(self, issues: List[Dict]) -> Dict[str, str]:
        """Generate actual code fixes."""
        fixes = {}
        for issue in issues:
            issue_type = issue.get("type", "")
            line = issue.get("line", 0)
            
            if issue_type == "dangerous_function":
                if issue.get("function") == "eval":
                    fixes[f"line_{line}"] = "Replace eval() with ast.literal_eval()"
                elif issue.get("function") == "exec":
                    fixes[f"line_{line}"] = "Avoid exec() - use proper function calls"
            elif issue_type == "complex_function":
                fixes[f"line_{line}"] = f"Break down {issue.get('function')}() into smaller functions"
        return fixes

    def _analyze_documentation(self, code_content: str) -> List[Dict[str, Any]]:
        """Analyze documentation quality."""
        issues = []
        try:
            tree = ast.parse(code_content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not ast.get_docstring(node):
                        issues.append({
                            "type": "missing_docstring",
                            "function": node.name,
                            "line": node.lineno,
                            "severity": "medium",
                            "location": str(node.lineno)
                        })
        except SyntaxError:
            pass
        return issues



    def _detect_code_smells(self, code_content: str) -> List[Dict[str, Any]]:
        """Detect code smells and anti-patterns."""
        smells = []
        
        # Long parameter lists
        if re.search(r'def\s+\w+\([^)]{50,}\)', code_content):
            smells.append({
                "type": "long_parameter_list",
                "message": "Function has too many parameters",
                "severity": "medium"
            })
        

        
        return smells

    def _standard_response(self, status: str, action: str, data: Optional[Dict] = None, errors: Optional[List] = None) -> Dict[str, Any]:
        """Generate standardized response with strict schema."""
        return {
            "status": status,
            "action": action,
            "data": data or {},
            "errors": errors or []
        }






            


        
    

