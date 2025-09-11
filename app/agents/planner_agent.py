""" Smart Planner Agent - Query-based agent routing and coordination"""
from app.mcp.mcp_schema import TaskMessage
from app.utils.logger import get_logger, log_execution_time
from typing import Dict, Any, List, Optional
import asyncio
import time

logger = get_logger(__name__)

class PlannerAgent:
    """Smart agent coordinator that routes queries to appropriate agents"""
    def __init__(self):
        self.name = "PlannerAgent"
        self.agent_registry = self._build_agent_registry()
        logger.info(f"Initialized {self.name} with {len(self.agent_registry)} agents")

    def _build_agent_registry(self) -> Dict[str, Dict[str, Any]]:
        """Registry of all available agents with their capabilities"""
        return {
            "DataAgent": {
                "module": "app.agents.data_agent",
                "class": "DataAgent",
                "keywords": ["data", "csv", "clean", "preprocess", "outlier", "missing", "eda", "explore"],
                "capabilities": ["data_cleaning", "data_analysis", "outlier_detection", "eda"]
            },
            "MLAgent": {
                "module": "app.agents.ml_agent", 
                "class": "MLAgent",
                "keywords": ["ml", "machine learning", "xgboost", "random forest", "svm", "clustering", "predict"],
                "capabilities": ["classification", "regression", "clustering", "prediction"]
            },
            "DeepLearningAgent": {
                "module": "app.agents.Deep_learning_Agent",
                "class": "DeepLearningAgent", 
                "keywords": ["neural", "deep", "cnn", "rnn", "lstm", "transformer", "image", "annotation"],
                "capabilities": ["neural_networks", "image_processing", "deep_learning", "annotation"]
            },
            "NLPAgent": {
                "module": "app.agents.nlp_agent",
                "class": "NLPAgent",
                "keywords": ["nlp", "text", "sentiment", "embedding", "entity", "summarize", "classify text"],
                "capabilities": ["text_processing", "sentiment_analysis", "ner", "summarization"]
            },
            "CodeAgent": {
                "module": "app.agents.code_agent",
                "class": "CodeAgent", 
                "keywords": ["code", "python", "generate", "execute", "review", "debug", "script"],
                "capabilities": ["code_generation", "code_execution", "code_review"]
            },
            "VisualizationAgent": {
                "module": "app.agents.VisualizationAgent",
                "class": "VisualizationAgent",
                "keywords": ["plot", "chart", "graph", "visualize", "histogram", "scatter", "dashboard"],
                "capabilities": ["plotting", "visualization", "charts", "graphs"]
            },
            "ModelEvaluationAgent": {
                "module": "app.agents.model_evaluation_agent",
                "class": "ModelEvaluationAgent",
                "keywords": ["evaluate", "metrics", "accuracy", "optimize", "tune", "benchmark", "performance"],
                "capabilities": ["model_evaluation", "optimization", "benchmarking"]
            },
            "CriticAgent": {
                "module": "app.agents.CriticAgent", 
                "class": "CriticAgent",
                "keywords": ["review", "analyze", "check", "critique", "assess", "quality"],
                "capabilities": ["quality_review", "analysis", "routing"]
            },
            "RetrievalAgent": {
                "module": "app.agents.RetrievalAgent",
                "class": "RetrievalAgent",
                "keywords": ["retrieve", "search", "rag", "knowledge", "context", "find"],
                "capabilities": ["data_retrieval", "rag", "search", "context_enhancement"]
            }
        }

    @log_execution_time
    def execute(self, task: TaskMessage) -> Dict[str, Any]:
        """Smart query routing to appropriate agent"""
        try:
            query = task.inputs.get("query", "").lower()
            
            # Analyze query and determine best agent
            best_agent = self._analyze_query_and_route(query, task.inputs)
            
            if best_agent:
                return self._execute_agent(best_agent, task)
            else:
                return self._multi_agent_execution(task)
                
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _analyze_query_and_route(self, query: str, inputs: Dict[str, Any]) -> Optional[str]:
        """Advanced smart query analysis with NLP-based routing"""
        
        # Multi-dimensional analysis
        intent_scores = self._analyze_intent(query)
        keyword_scores = self._analyze_keywords(query)
        context_scores = self._analyze_context(query, inputs)
        semantic_scores = self._analyze_semantics(query)
        
        # Combine all scoring dimensions
        final_scores = {}
        
        for agent_name in self.agent_registry.keys():
            total_score = (
                intent_scores.get(agent_name, 0) * 0.4 +      # Intent weight: 40%
                keyword_scores.get(agent_name, 0) * 0.3 +     # Keyword weight: 30%
                context_scores.get(agent_name, 0) * 0.2 +     # Context weight: 20%
                semantic_scores.get(agent_name, 0) * 0.1      # Semantic weight: 10%
            )
            
            if total_score > 0:
                final_scores[agent_name] = total_score
        
        # Return the highest scoring agent with confidence threshold
        if final_scores:
            best_agent = max(final_scores, key=final_scores.get)
            confidence = final_scores[best_agent]
            
            # Only route if confidence is above threshold
            if confidence >= 2.0:  # Minimum confidence threshold
                logger.info(f"Smart routing: {best_agent} (confidence: {confidence:.2f})")
                return best_agent
            else:
                logger.info(f"Low confidence ({confidence:.2f}), using multi-agent approach")
                return None
        
        return None
    
    def _analyze_intent(self, query: str) -> Dict[str, float]:
        """Analyze user intent from query structure and verbs"""
        intent_scores = {}
        
        # Action-based intent detection
        action_patterns = {
            "DataAgent": ["clean", "preprocess", "explore", "analyze data", "load", "import", "filter"],
            "CodeAgent": ["generate", "create code", "write", "execute", "run", "debug", "review code"],
            "MLAgent": ["train", "predict", "classify", "cluster", "fit model", "machine learning"],
            "DeepLearningAgent": ["neural", "deep learning", "cnn", "rnn", "transformer", "annotate"],
            "NLPAgent": ["analyze text", "sentiment", "summarize", "extract", "classify text"],
            "VisualizationAgent": ["plot", "visualize", "chart", "graph", "show", "display"],
            "ModelEvaluationAgent": ["evaluate", "optimize", "tune", "benchmark", "improve", "assess"],
            "CriticAgent": ["review", "critique", "analyze", "check", "assess", "examine", "inspect"]
        }
        
        for agent, actions in action_patterns.items():
            score = 0
            for action in actions:
                if action in query:
                    score += 5  # High weight for action verbs
                    # Bonus for action at start of query
                    if query.startswith(action):
                        score += 3
            intent_scores[agent] = score
        
        return intent_scores
    
    def _analyze_keywords(self, query: str) -> Dict[str, float]:
        """Enhanced keyword analysis with synonyms and context"""
        keyword_scores = {}
        
        # Enhanced keyword patterns with synonyms
        enhanced_keywords = {
            "DataAgent": {
                "primary": ["data", "dataset", "csv", "dataframe", "table"],
                "secondary": ["clean", "preprocess", "outlier", "missing", "eda", "explore"],
                "synonyms": ["information", "records", "rows", "columns", "file"]
            },
            "CodeAgent": {
                "primary": ["code", "python", "script", "function", "class"],
                "secondary": ["generate", "execute", "debug", "review", "syntax"],
                "synonyms": ["program", "algorithm", "implementation", "logic"]
            },
            "NLPAgent": {
                "primary": ["text", "nlp", "sentiment", "language"],
                "secondary": ["analyze", "extract", "summarize", "classify"],
                "synonyms": ["document", "content", "words", "sentences"]
            },
            "MLAgent": {
                "primary": ["model", "ml", "machine learning", "predict"],
                "secondary": ["train", "fit", "classify", "regression"],
                "synonyms": ["algorithm", "learning", "prediction", "classification"]
            },
            "CriticAgent": {
                "primary": ["review", "critique", "analyze", "check", "assess"],
                "secondary": ["quality", "evaluate", "examine", "inspect", "audit"],
                "synonyms": ["judge", "rate", "grade", "score", "validate"]
            },
            "VisualizationAgent": {
                "primary": ["plot", "chart", "graph", "visualize", "dashboard"],
                "secondary": ["histogram", "scatter", "bar", "line", "pie"],
                "synonyms": ["display", "show", "render", "draw", "create"]
            }
        }
        
        for agent, keyword_groups in enhanced_keywords.items():
            score = 0
            
            # Primary keywords (high weight)
            for keyword in keyword_groups["primary"]:
                if keyword in query:
                    score += 4
                    if f" {keyword} " in f" {query} ":
                        score += 2  # Exact match bonus
            
            # Secondary keywords (medium weight)
            for keyword in keyword_groups["secondary"]:
                if keyword in query:
                    score += 2
            
            # Synonyms (low weight)
            for keyword in keyword_groups["synonyms"]:
                if keyword in query:
                    score += 1
            
            keyword_scores[agent] = score
        
        return keyword_scores
    
    def _analyze_context(self, query: str, inputs: Dict[str, Any]) -> Dict[str, float]:
        """Analyze context from inputs and query structure"""
        context_scores = {}
        
        # Input-based context analysis
        input_patterns = {
            "DataAgent": ["data_path", "csv_file", "dataframe", "dataset", "table"],
            "CodeAgent": ["code", "code_content", "script", "python_code", "function"],
            "ModelEvaluationAgent": ["model", "model_metrics", "accuracy", "performance"],
            "NLPAgent": ["text_data", "text", "documents", "corpus", "sentences"],
            "VisualizationAgent": ["plot_type", "chart_data", "x_axis", "y_axis"]
        }
        
        for agent, input_keys in input_patterns.items():
            score = 0
            for key in input_keys:
                if key in inputs:
                    score += 6  # High weight for relevant inputs
                    # Bonus if input has actual data
                    if inputs[key] and str(inputs[key]).strip():
                        score += 2
            context_scores[agent] = score
        
        return context_scores
    
    def _analyze_semantics(self, query: str) -> Dict[str, float]:
        """Semantic analysis using word relationships and domain knowledge"""
        semantic_scores = {}
        
        # Domain-specific semantic patterns
        semantic_domains = {
            "DataAgent": {
                "data_science": ["pandas", "numpy", "statistics", "analysis"],
                "data_quality": ["clean", "validate", "quality", "integrity"]
            },
            "MLAgent": {
                "algorithms": ["xgboost", "random forest", "svm", "clustering"],
                "workflow": ["pipeline", "training", "validation", "testing"]
            },
            "DeepLearningAgent": {
                "architectures": ["cnn", "rnn", "lstm", "transformer", "bert"],
                "tasks": ["image", "vision", "sequence", "attention"]
            }
        }
        
        for agent, domains in semantic_domains.items():
            score = 0
            for domain, terms in domains.items():
                domain_matches = sum(1 for term in terms if term in query)
                if domain_matches > 0:
                    score += domain_matches * 2
                    if domain_matches >= 2:
                        score += 3
            semantic_scores[agent] = score
        
        return semantic_scores

    def _execute_agent(self, agent_name: str, task: TaskMessage) -> Dict[str, Any]:
        """Execute specific agent"""
        try:
            agent_info = self.agent_registry[agent_name]
            
            # Dynamic import and execution
            module = __import__(agent_info["module"], fromlist=[agent_info["class"]])
            agent_class = getattr(module, agent_info["class"])
            agent = agent_class()
            
            # Execute the agent
            start_time = time.time()
            result = agent.execute(task)
            execution_time = time.time() - start_time
            
            return {
                "status": "success",
                "agent_used": agent_name,
                "execution_time": f"{execution_time:.2f}s",
                "result": result,
                "message": f"Query successfully handled by {agent_name}"
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "agent_used": agent_name,
                "errors": [f"Agent {agent_name} failed: {str(e)}"]
            }

    def _multi_agent_execution(self, task: TaskMessage) -> Dict[str, Any]:
        """Execute multiple agents for complex queries"""
        try:
            query = task.inputs.get("query", "").lower()
            results = {}
            
            # Determine which agents might be relevant
            relevant_agents = self._get_relevant_agents(query, task.inputs)
            
            if not relevant_agents:
                # Fallback to CriticAgent for routing
                return self._execute_agent("CriticAgent", task)
            
            # Execute relevant agents
            for agent_name in relevant_agents[:3]:  # Limit to top 3 agents
                try:
                    result = self._execute_agent(agent_name, task)
                    if result.get("status") == "success":
                        results[agent_name] = result
                except Exception as e:
                    logger.warning(f"Agent {agent_name} failed: {str(e)}")
            
            return {
                "status": "success",
                "execution_type": "multi_agent",
                "agents_used": list(results.keys()),
                "results": results,
                "message": f"Query handled by {len(results)} agents"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _get_relevant_agents(self, query: str, inputs: Dict[str, Any]) -> List[str]:
        """Get list of potentially relevant agents"""
        relevant = []
        
        # Check for data-related content
        if any(word in query for word in ["data", "dataset", "csv"]) or "data_path" in inputs:
            relevant.append("DataAgent")
        
        # Check for code-related content  
        if any(word in query for word in ["code", "python", "script"]) or "code" in inputs:
            relevant.append("CodeAgent")
        
        # Check for ML/model content
        if any(word in query for word in ["model", "predict", "train", "ml"]) or "model" in inputs:
            relevant.extend(["MLAgent", "DeepLearningAgent", "ModelEvaluationAgent"])
        
        # Check for text/NLP content
        if any(word in query for word in ["text", "nlp", "sentiment"]) or "text_data" in inputs:
            relevant.append("NLPAgent")
        
        # Check for visualization content
        if any(word in query for word in ["plot", "chart", "visualize"]):
            relevant.append("VisualizationAgent")
        
        return relevant

    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all registered agents"""
        return {
            agent_name: agent_info["capabilities"] 
            for agent_name, agent_info in self.agent_registry.items()
        }

    def route_query(self, query: str, inputs: Dict[str, Any] = None) -> str:
        """Public method to determine which agent should handle a query"""
        inputs = inputs or {}
        best_agent = self._analyze_query_and_route(query.lower(), inputs)
        return best_agent or "CriticAgent"  # Fallback to CriticAgent

    def execute_pipeline(self, tasks: List[TaskMessage]) -> Dict[str, Any]:
        """Execute a pipeline of tasks with intelligent routing"""
        try:
            results = []
            
            for task in tasks:
                # Route each task to appropriate agent
                result = self.execute(task)
                results.append({
                    "task_id": task.task_id,
                    "result": result
                })
            
            return {
                "status": "success",
                "pipeline_results": results,
                "total_tasks": len(tasks),
                "message": "Pipeline execution completed"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def get_agent_status(self) -> Dict[str, str]:
        """Get status of all registered agents"""
        status = {}
        
        for agent_name, agent_info in self.agent_registry.items():
            try:
                # Try to import the agent
                module = __import__(agent_info["module"], fromlist=[agent_info["class"]])
                getattr(module, agent_info["class"])
                status[agent_name] = "available"
            except Exception as e:
                status[agent_name] = f"unavailable: {str(e)}"
        
        return status
    
    def validate_agents(self) -> Dict[str, Any]:
        """Validate all agents are working properly"""
        validation_results = {}
        
        for agent_name, agent_info in self.agent_registry.items():
            try:
                # Try to create agent instance
                module = __import__(agent_info["module"], fromlist=[agent_info["class"]])
                agent_class = getattr(module, agent_info["class"])
                agent = agent_class()
                
                validation_results[agent_name] = {
                    "status": "valid",
                    "class_name": agent_info["class"],
                    "module": agent_info["module"]
                }
            except Exception as e:
                validation_results[agent_name] = {
                    "status": "invalid",
                    "error": str(e),
                    "class_name": agent_info["class"],
                    "module": agent_info["module"]
                }
        
        return validation_results

    def test_routing(self, test_queries: List[str]) -> Dict[str, Dict[str, Any]]:
        """Advanced routing test with detailed analysis"""
        results = {}
        
        for query in test_queries:
            intent_scores = self._analyze_intent(query)
            keyword_scores = self._analyze_keywords(query)
            best_agent = self.route_query(query)
            
            results[query] = {
                "routed_to": best_agent,
                "intent_analysis": intent_scores,
                "keyword_analysis": keyword_scores,
                "confidence": max(intent_scores.values()) if intent_scores else 0
            }
        
        return results
    
    def explain_routing(self, query: str, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Explain why a query was routed to a specific agent"""
        inputs = inputs or {}
        
        intent_scores = self._analyze_intent(query)
        keyword_scores = self._analyze_keywords(query)
        context_scores = self._analyze_context(query, inputs)
        semantic_scores = self._analyze_semantics(query)
        
        best_agent = self._analyze_query_and_route(query, inputs)
        
        return {
            "query": query,
            "routed_to": best_agent,
            "analysis": {
                "intent_scores": intent_scores,
                "keyword_scores": keyword_scores,
                "context_scores": context_scores,
                "semantic_scores": semantic_scores
            },
            "reasoning": f"Routed to {best_agent}" if best_agent else "Multi-agent approach"
        }