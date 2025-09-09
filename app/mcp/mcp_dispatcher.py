"""MCP Dispatcher - Routes tasks to appropriate agents"""

from typing import Dict, Any
from app.mcp.mcp_schema import TaskManager
from app.utils.logger import get_logger, log_execution_time

# Import all agents
from app.agents.data_agent import DataAgent
from app.agents.ml_agent import MLAgent
from app.agents.Deep_learning_Agent import DeepLearningAgent
from app.agents.eval_agent import EvalAgent
from app.agents.OptimizerAgent import OptimizerAgent
from app.agents.VisualizationAgent import VisualizationAgent
from app.agents.CriticAgent import CriticAgent
from app.agents.RetrievalAgent import RetrievalAgent
from app.agents.code_agent import CodeAgent
from app.agents.planner_agent import PlannerAgent

logger = get_logger(__name__)

class MCPDispatcher:
    def __init__(self):
        self.agents = {
            "DataAgent": DataAgent(),
            "MLAgent": MLAgent(),
            "DeepLearningAgent": DeepLearningAgent(),
            "EvalAgent": EvalAgent(),
            "OptimizerAgent": OptimizerAgent(),
            "VisualizationAgent": VisualizationAgent(),
            "CriticAgent": CriticAgent(),
            "RetrievalAgent": RetrievalAgent(),
            "CodeAgent": CodeAgent(),
            "PlannerAgent": PlannerAgent()
        }
        logger.info("MCP Dispatcher initialized with all agents")

    def validate_task(self, task: TaskManager, completed_tasks: list = None) -> bool:
        """Validate task before dispatch"""
        completed_tasks = completed_tasks or []
        
        # Check if agent exists
        if task.agent_role not in self.agents:
            logger.error(f"Unknown agent: {task.agent_role}")
            return False
        
        # Check dependencies
        if not task.is_ready_to_execute(completed_tasks):
            logger.warning(f"Task {task.task_id} dependencies not met")
            return False
        
        return True
    
    @log_execution_time
    def dispatch(self, task: TaskManager) -> Dict[str, Any]:
        """Dispatch task to appropriate agent"""
        logger.info(f"Dispatching task {task.task_id} to {task.agent_role}")
        
        try:
            # Mark task as in progress
            task.mark_in_progress()
            
            # Get the appropriate agent
            agent = self.agents.get(task.agent_role)
            if not agent:
                raise ValueError(f"Agent {task.agent_role} not found")
            
            # Execute the task
            result = agent.execute(task)
            
            # Mark task as completed
            task.mark_completed(result)
            logger.info(f"Task {task.task_id} completed in {task.execution_time:.2f}s")
            
            return {
                "task_id": task.task_id,
                "status": "completed",
                "result": result,
                "agent": task.agent_role,
                "execution_time": task.execution_time
            }
            
        except Exception as e:
            task.mark_failed(str(e))
            logger.error(f"Task {task.task_id} failed: {str(e)}")
            return {
                "task_id": task.task_id,
                "status": "failed",
                "error": str(e),
                "agent": task.agent_role,
                "execution_time": task.execution_time
            }
    
    def dispatch_batch(self, tasks: list) -> list:
        """Dispatch multiple tasks with dependency management"""
        results = []
        completed_task_ids = []
        
        for task in tasks:
            if self.validate_task(task, completed_task_ids):
                result = self.dispatch(task)
                results.append(result)
                
                if result["status"] == "completed":
                    completed_task_ids.append(task.task_id)
            else:
                results.append({
                    "task_id": task.task_id,
                    "status": "skipped",
                    "reason": "Dependencies not met or invalid agent"
                })
        
        return results

    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed status of all agents"""
        import time
        status = {}
        
        for name, agent in self.agents.items():
            try:
                agent_info = {
                    "status": "ready",
                    "name": getattr(agent, 'name', name),
                    "tools": getattr(agent, 'tools', []),
                    "last_check": time.time()
                }
                
                # Add specific info for PlannerAgent
                if hasattr(agent, 'max_workers'):
                    agent_info["max_workers"] = agent.max_workers
                
                status[name] = agent_info
                
            except Exception as e:
                status[name] = {
                    "status": "error",
                    "error": str(e),
                    "last_check": time.time()
                }
        
        return status