""" Planner agent: NetworkX + LangGraph + Async + Monitoring + Recovery
- Using ReAct and TOT thats tree of thought + memory recall to break into the subtask"""

from typing import List, Dict, Any, Optional
import networkx as nx
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from langgraph.graph import StateGraph, END
from app.mcp.mcp_schema import TaskManager
from app.utils.logger import get_logger, log_execution_time
from dataclasses import dataclass
from enum import Enum

logger = get_logger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class ExecutionMetrics:
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_usage: Optional[float] = None

class PlannerAgent:
    def __init__(self, max_workers: int = 4, max_retries: int = 3):
        self.name = "PlannerAgent"
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.executor_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.workflow = self.build_workflow()
        self.agent_registry = self._build_agent_registry()
    
    @log_execution_time
    def execute(self, task: TaskManager) -> Dict[str, Any]:
        """Main execute method for PlannerAgent"""
        logger.info(f"PlannerAgent executing task: {task.task_id}")
        
        try:
            subtasks = task.inputs.get("subtasks", [])
            if not subtasks:
                return {"status": "error", "message": "No subtasks provided"}
            
            result = asyncio.run(self.execute_advanced(subtasks))
            
            return {
                "status": "success",
                "action": "task_planning",
                "execution_summary": result["execution_summary"],
                "completed_tasks": len(result["completed_tasks"]),
                "failed_tasks": len(result["failed_tasks"]),
                "summary": f"Executed {len(result['completed_tasks'])} tasks successfully"
            }
            
        except Exception as e:
            logger.error(f"PlannerAgent execution failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _build_agent_registry(self) -> Dict[str, Dict[str, Any]]:
        """Complete agent registry with all agents and retry strategies."""
        return {
            "DataAgent": {
                "module": "app.agents.data_agent",
                "timeout": 300,
                "retry_strategy": "exponential_backoff"
            },
            "MLAgent": {
                "module": "app.agents.ml_agent",
                "timeout": 1800,
                "retry_strategy": "linear_backoff"
            },
            "DeepLearningAgent": {
                "module": "app.agents.Deep_learning_Agent",
                "timeout": 3600,
                "retry_strategy": "exponential_backoff"
            },
            "EvalAgent": {
                "module": "app.agents.eval_agent",
                "timeout": 600,
                "retry_strategy": "immediate"
            },
            "CodeAgent": {
                "module": "app.agents.code_agent",
                "timeout": 300,
                "retry_strategy": "immediate"
            },
            "VisualizationAgent": {
                "module": "app.agents.VisualizationAgent",
                "timeout": 300,
                "retry_strategy": "immediate"
            },
            "CriticAgent": {
                "module": "app.agents.CriticAgent",
                "timeout": 600,
                "retry_strategy": "linear_backoff"
            },
            "OptimizerAgent": {
                "module": "app.agents.OptimizerAgent",
                "timeout": 1200,
                "retry_strategy": "exponential_backoff"
            },
            "RetrievalAgent": {
                "module": "app.agents.RetrievalAgent",
                "timeout": 300,
                "retry_strategy": "immediate"
            }
        }

    @log_execution_time
    def build_advanced_plan(self, subtasks: List[TaskManager]) -> Dict[str, Any]:
        """Advanced planning with parallel groups and metrics."""
        logger.info(f"Building advanced execution plan for {len(subtasks)} tasks")
        G = nx.DiGraph()
        
        # Add nodes with enhanced metadata
        for task in subtasks:
            agent_info = self.agent_registry.get(task.agent_role, {})
            G.add_node(
                task.task_id,
                task=task,
                role=task.agent_role,
                status=TaskStatus.PENDING,
                metrics=ExecutionMetrics(start_time=0),
                retry_count=0,
                timeout=agent_info.get("timeout", 300)
            )
        
        # Add dependencies
        for task in subtasks:
            for dep in task.dependencies:
                if dep in G.nodes:
                    G.add_edge(dep, task.task_id)
        
        # Validate DAG
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Cycle detected in task dependencies")
        
        # Find parallel execution groups
        parallel_groups = self._find_parallel_groups(G)
        execution_order = list(nx.topological_sort(G))
        
        return {
            "graph": G,
            "execution_order": execution_order,
            "parallel_groups": parallel_groups,
            "tasks": {t.task_id: t for t in subtasks}
        }
    
    def _find_parallel_groups(self, G: nx.DiGraph) -> List[List[str]]:
        """Find tasks that can run in parallel."""
        parallel_groups = []
        remaining_nodes = set(G.nodes())
        
        while remaining_nodes:
            ready_nodes = [
                node for node in remaining_nodes 
                if all(pred not in remaining_nodes for pred in G.predecessors(node))
            ]
            
            if ready_nodes:
                parallel_groups.append(ready_nodes)
                remaining_nodes -= set(ready_nodes)
            else:
                break
                
        return parallel_groups

    def build_workflow(self) -> StateGraph:
        """Build advanced LangGraph workflow."""
        workflow = StateGraph(dict)
        
        workflow.add_node("validate_deps", self.validate_dependencies)
        workflow.add_node("execute_parallel", self.execute_parallel_tasks)
        workflow.add_node("monitor_progress", self.monitor_execution)
        workflow.add_node("handle_failures", self.handle_task_failures)
        workflow.add_node("finalize", self.finalize_execution)
        
        workflow.set_entry_point("validate_deps")
        workflow.add_edge("validate_deps", "execute_parallel")
        workflow.add_edge("execute_parallel", "monitor_progress")
        workflow.add_conditional_edges(
            "monitor_progress",
            self.should_handle_failures,
            {
                "handle_failures": "handle_failures",
                "finalize": "finalize"
            }
        )
        workflow.add_edge("handle_failures", "execute_parallel")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()

    async def validate_dependencies(self, state: dict) -> dict:
        """Validate all task dependencies."""
        logger.info("Validating task dependencies...")
        plan = state["plan"]
        
        # Validate DAG structure
        if not nx.is_directed_acyclic_graph(plan["graph"]):
            raise ValueError("Invalid task dependencies: cycle detected")
        
        logger.info(f"Dependencies validated for {len(plan['execution_order'])} tasks")
        return state

    async def execute_parallel_tasks(self, state: dict) -> dict:
        """Execute tasks in parallel groups."""
        group_id = state.get("current_parallel_group", 0)
        groups = state["plan"]["parallel_groups"]
        
        if group_id < len(groups):
            tasks = groups[group_id]
            logger.info(f"Executing parallel group {group_id}: {tasks}")
            
            # Execute tasks in parallel
            results = await asyncio.gather(*[
                self.execute_agent_async(state["plan"]["tasks"][tid]) 
                for tid in tasks
            ])
            
            # Process results
            for task_id, result in zip(tasks, results):
                state["results"][task_id] = result
                if result["status"] == TaskStatus.COMPLETED.value:
                    state["completed_tasks"].add(task_id)
                else:
                    state["failed_tasks"].append(task_id)
            
            state["current_parallel_group"] = group_id + 1
        
        return state
    
    async def monitor_execution(self, state: dict) -> dict:
        """Monitor execution progress."""
        completed = len(state["completed_tasks"])
        total = len(state["plan"]["execution_order"])
        failed = len(state["failed_tasks"])
        
        logger.info(f"Progress: {completed}/{total} completed, {failed} failed")
        return state
    
    def should_handle_failures(self, state: dict) -> str:
        """Decide whether to handle failures or finalize."""
        return "handle_failures" if state["failed_tasks"] else "finalize"
    
    async def handle_task_failures(self, state: dict) -> dict:
        """Handle failed tasks with retry logic."""
        failed_tasks = state["failed_tasks"].copy()
        logger.warning(f"Handling {len(failed_tasks)} failed tasks...")
        
        for task_id in failed_tasks:
            node = state["plan"]["graph"].nodes[task_id]
            
            if node["retry_count"] < self.max_retries:
                node["retry_count"] += 1
                agent_info = self.agent_registry.get(node["role"], {})
                retry_strategy = agent_info.get("retry_strategy", "immediate")
                
                # Apply retry delay based on strategy
                if retry_strategy == "exponential_backoff":
                    delay = 2 ** node["retry_count"]
                elif retry_strategy == "linear_backoff":
                    delay = node["retry_count"] * 5
                else:
                    delay = 1
                
                logger.info(f"Retrying task {task_id} in {delay}s (attempt {node['retry_count']})")
                await asyncio.sleep(delay)
                
                # Retry the task
                result = await self.execute_agent_async(node["task"])
                state["results"][task_id] = result
                
                if result["status"] == TaskStatus.COMPLETED.value:
                    state["completed_tasks"].add(task_id)
                    state["failed_tasks"].remove(task_id)
                    logger.info(f"Task {task_id} succeeded on retry")
            else:
                logger.error(f"Task {task_id} failed after {self.max_retries} retries")
        
        return state
    
    async def finalize_execution(self, state: dict) -> dict:
        """Finalize execution and generate report."""
        logger.info("Finalizing execution")
        end_time = time.time()
        total_time = end_time - state["start_time"]
        
        state["execution_summary"] = {
            "total_duration": total_time,
            "success_rate": len(state["completed_tasks"]) / len(state["plan"]["execution_order"]),
            "failed_count": len(state["failed_tasks"])
        }
        
        return state

    async def execute_agent_async(self, task: TaskManager) -> dict:
        """Execute agent asynchronously with timeout and monitoring."""
        agent_info = self.agent_registry.get(task.agent_role, {})
        start_time = time.time()
        
        try:
            # Dynamic import
            module_path = agent_info.get("module")
            if not module_path:
                return {"status": TaskStatus.FAILED.value, "error": f"Unknown agent: {task.agent_role}"}
            
            module = __import__(module_path, fromlist=[task.agent_role])
            agent_class = getattr(module, task.agent_role)
            agent = agent_class()
            
            # Execute with timeout
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.executor_pool,
                    agent.execute,
                    task
                ),
                timeout=agent_info.get("timeout", 300)
            )
            
            end_time = time.time()
            return {
                "status": TaskStatus.COMPLETED.value,
                "result": result,
                "agent": task.agent_role,
                "duration": end_time - start_time
            }
            
        except asyncio.TimeoutError:
            return {
                "status": TaskStatus.FAILED.value,
                "error": f"Task timeout after {agent_info.get('timeout', 300)}s",
                "agent": task.agent_role
            }
        except Exception as e:
            return {
                "status": TaskStatus.FAILED.value,
                "error": str(e),
                "agent": task.agent_role
            }

    @log_execution_time
    async def execute_advanced(self, subtasks: List[TaskManager]) -> Dict[str, Any]:
        """Advanced execution with full monitoring and recovery."""
        logger.info(f"Starting advanced execution of {len(subtasks)} tasks")
        
        # Build advanced plan
        plan = self.build_advanced_plan(subtasks)
        logger.info(f"Parallel groups: {len(plan['parallel_groups'])}")
        
        # Initialize state
        state = {
            "plan": plan,
            "results": {},
            "failed_tasks": [],
            "completed_tasks": set(),
            "current_parallel_group": 0,
            "start_time": time.time()
        }
        
        # Execute workflow
        final_state = await self.workflow.ainvoke(state)
        
        logger.info(f"Advanced execution completed: {final_state['execution_summary']}")
        return final_state