"""ReAct + ToT task breakdown logic.Help  to make  the task into sub task after suing the ReAct it reasoning step by step or plan  and pass them inot planner/agent / vai Mcp"""
# here we define all the agent task

from typing import List, Optional, Dict, Any
from app.mcp.mcp_schema import TaskManager
import uuid
from app.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)

class TaskParser:
    def __init__(self):
        pass
    
    def get_query_complexity(self, query: str) -> str:
        """Enhanced complexity analysis with weighted keywords"""
        high_complexity = ["train", "deep", "neural", "optimize", "hyperparameter", "pipeline"]
        medium_complexity = ["evaluate", "visualize", "clean", "prepare", "model"]
        low_complexity = ["plot", "show", "print", "simple"]
        
        high_matches = sum(2 for kw in high_complexity if kw in query.lower())
        medium_matches = sum(1 for kw in medium_complexity if kw in query.lower())
        low_matches = sum(0.5 for kw in low_complexity if kw in query.lower())
        
        total_score = high_matches + medium_matches + low_matches
        
        if total_score >= 4: return "high"
        elif total_score >= 2: return "medium"
        else: return "low"
    
    def validate_dependencies(self, subtasks: List[TaskManager]) -> bool:
        """Validate that all dependencies exist in task list."""
        task_ids = {task.task_id for task in subtasks}
        for task in subtasks:
            for dep in task.dependencies:
                if dep not in task_ids:
                    return False
        return True
    
    def generate_task_id(self) -> str:
        """Generate unique task ID."""
        return str(uuid.uuid4())[:8]
    
    def _extract_context(self, query: str) -> Dict[str, Any]:
        """Extract key context from query"""
        context = {}
        
        # Extract data source
        if any(word in query.lower() for word in ["csv", "json", "excel", "database"]):
            context["has_data_source"] = True
        
        # Extract model type
        if "classification" in query.lower():
            context["model_type"] = "classification"
        elif "regression" in query.lower():
            context["model_type"] = "regression"
        
        # Extract action type
        if any(word in query.lower() for word in ["build", "create", "develop"]):
            context["action"] = "create"
        elif any(word in query.lower() for word in ["fix", "debug", "error"]):
            context["action"] = "debug"
        
        return context

    @log_execution_time
    def parse(self, user_query: str) -> List[TaskManager]:
        logger.info(f"Parsing query: {user_query}")
        subtasks: List[TaskManager] = []
        query_lower = user_query.lower()
        complexity = self.get_query_complexity(user_query)
        context = self._extract_context(user_query)
        logger.debug(f"Query complexity: {complexity}, Context: {context}")
        
        # --- Planning Agent (for complex queries) ---
        if "plan" in query_lower or "strategy" in query_lower or "workflow" in query_lower or complexity == "high":
            subtasks.append(
                TaskManager(
                    task_id=self.generate_task_id(),
                    sender="user",
                    receiver="PlannerAgent",
                    agent_role="PlannerAgent",
                    inputs={"query": "Create execution plan", "complexity": complexity},
                    tools=["networkx", "graphviz", "workflow_engine", "task_scheduler"],
                    expected_output="Task execution plan and workflow",
                    dependencies=[],
                    metadata={"priority": "highest", "complexity": complexity}
                )
            )

        # --- Data Preparation ---
        if "clean" in query_lower or "prepare" in query_lower or "dataset" in query_lower:
            subtasks.append(
                TaskManager(
                    task_id=self.generate_task_id(),
                    sender="user",
                    receiver="DataAgent",
                    agent_role="DataAgent",
                    inputs={"query": "Prepare and clean dataset"},
                    tools=["pandas", "numpy", "polars", "dask", "pyjanitor", "great_expectations", "pandas_profiling"],
                    expected_output="Cleaned dataset ready for ML/DL",
                    dependencies=[],
                    metadata={"priority": "high"}
                )
            )

        # --- Retrieval for Knowledge/External Data ---
        if "retrieve" in query_lower or "search" in query_lower or "fetch" in query_lower:
            subtasks.append(
                TaskManager(
                    task_id=self.generate_task_id(),
                    sender="user",
                    receiver="RetrievalAgent",
                    agent_role="RetrievalAgent",
                    inputs={"query": "Fetch external knowledge or data"},
                    tools=["requests", "chromadb", "faiss", "pinecone", "weaviate", "langchain", "beautifulsoup4", "scrapy"],
                    expected_output="Relevant retrieved documents/data",
                    dependencies=[],
                    metadata={"priority": "medium"}
                )
            )

        # --- Code Generation ---
        if "code" in query_lower or "generate" in query_lower or "implement" in query_lower or "write" in query_lower:
            subtasks.append(
                TaskManager(
                    task_id=self.generate_task_id(),
                    sender="user",
                    receiver="CodeAgent",
                    agent_role="CodeAgent",
                    inputs={"query": user_query, "action": "generate"},
                    tools=["python_repl", "jupyter", "black", "flake8"],
                    expected_output="Generated code",
                    dependencies=[],
                    metadata={"priority": "medium"}
                )
            )


        # --- Traditional Machine Learning ---
        if "train" in query_lower and "ml" in query_lower:
            subtasks.append(
                TaskManager(
                    task_id=self.generate_task_id(),
                    sender="user",
                    receiver="MLAgent",
                    agent_role="MLAgent",
                    inputs={"query": "Train ML model on dataset"},
                    tools=["scikit-learn", "xgboost", "lightgbm", "catboost", "auto-sklearn", "pycaret", "mlflow"],
                    expected_output="Trained ML model file",
                    dependencies=[st.task_id for st in subtasks if st.agent_role == "DataAgent"],
                    metadata={"priority": "high"}
                )
            )

        # --- Deep Learning ---
        if "train" in query_lower and ("deep" in query_lower or "neural" in query_lower):
            subtasks.append(
                TaskManager(
                    task_id=self.generate_task_id(),
                    sender="user",
                    receiver="DeepLearningAgent",
                    agent_role="DeepLearningAgent",
                    inputs={"query": "Train deep learning model"},
                    tools=["torch", "tensorflow", "pytorch_lightning", "transformers", "accelerate", "wandb", "tensorboard"],
                    expected_output="Trained DL model file",
                    dependencies=[st.task_id for st in subtasks if st.agent_role == "DataAgent"],
                    metadata={"priority": "high"}
                )
            )

        
        # --- Model Evaluation (includes optimization) ---
        if "optimize" in query_lower or "tune" in query_lower or "hyperparameter" in query_lower:
            subtasks.append(
                TaskManager(
                    task_id=self.generate_task_id(),
                    sender="user",
                    receiver="ModelEvaluationAgent",
                    agent_role="ModelEvaluationAgent",
                    inputs={"query": "Optimize training process"},
                    tools=["optuna", "hyperopt", "sklearn.model_selection", "GridSearchCV"],
                    expected_output="Optimized model parameters",
                    dependencies=[st.task_id for st in subtasks if st.agent_role in ["MLAgent", "DeepLearningAgent"]],
                    metadata={"priority": "medium"}
                )
            )

        # --- Evaluation ---
        if "evaluate" in query_lower or "test" in query_lower or "accuracy" in query_lower:
            subtasks.append(
                TaskManager(
                    task_id=self.generate_task_id(),
                    sender="user",
                    receiver="ModelEvaluationAgent",
                    agent_role="ModelEvaluationAgent",
                    inputs={"query": "Evaluate model performance"},
                    tools=["sklearn.metrics", "torchmetrics", "evaluate", "confusion_matrix"],
                    expected_output="Evaluation report (accuracy, F1, etc.)",
                    dependencies=[st.task_id for st in subtasks if st.agent_role in ["MLAgent", "DeepLearningAgent"]],
                    metadata={"priority": "medium"}
                )
            )
        
        # --- Critic / Review ---

        if "review" in query_lower or "verify" in query_lower or "criticize" in query_lower:
            subtasks.append(
                TaskManager(
                    task_id=self.generate_task_id(),
                    sender="user",
                    receiver="CriticAgent",
                    agent_role="CriticAgent",
                    inputs={"query": "Review outputs for quality and correctness"},
                    tools=["pytest", "unittest", "hypothesis", "great_expectations", "deepchecks", "evidently", "alibi_detect"],
                    expected_output="Feedback with strengths/weaknesses",
                    dependencies=[st.task_id for st in subtasks if st.agent_role == "ModelEvaluationAgent"],
                    metadata={"priority": "low"}
                )
            )

        # --- Visualization ---
        if "plot" in query_lower or "visualize" in query_lower or "chart" in query_lower:
            subtasks.append(
                TaskManager(
                    task_id=self.generate_task_id(),
                    sender="user",
                    receiver="VisualizationAgent",
                    agent_role="VisualizationAgent",
                    inputs={"query": "Visualize results"},
                    tools=["matplotlib", "seaborn", "scipy", "plotly", "bokeh", "altair", "streamlit", "dash"],
                    expected_output="Generated charts/plots",
                    dependencies=[st.task_id for st in subtasks if st.agent_role == "ModelEvaluationAgent"],
                    metadata={"priority": "low"}
                )
            )


        # --- Code Execution (Fallback) ---
        if not subtasks:  # default if nothing matched
            subtasks.append(
                TaskManager(
                    task_id=self.generate_task_id(),
                    sender="user",
                    receiver="CodeAgent",
                    agent_role="CodeAgent",
                    inputs={"query": user_query},
                    tools=["python_repl", "jupyter", "black", "flake8", "mypy", "pytest", "ipython"],
                    expected_output="Code execution result",
                    dependencies=[],
                    metadata={"priority": "low"}
                )
            )

        # Validate dependencies before returning
        if not self.validate_dependencies(subtasks):
            # Add fallback task if validation fails
            subtasks = [TaskManager(
                task_id=self.generate_task_id(),
                sender="user",
                receiver="CodeAgent", 
                agent_role="CodeAgent",
                inputs={"query": user_query},
                tools=["python_repl", "jupyter"],
                expected_output="Code execution result",
                dependencies=[],
                metadata={"priority": "low", "fallback": True}
            )]
        
        return subtasks  
