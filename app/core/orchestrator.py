"""Main NeuroAct Orchestrator - Ties everything together"""

from typing import List, Dict, Any
from app.core.task_parser import TaskParser
from app.mcp.mcp_dispatcher import MCPDispatcher
from app.mcp.mcp_schema import TaskManager
from app.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)

class NeuroActOrchestrator:
    def __init__(self):
        self.task_parser = TaskParser()
        self.dispatcher = MCPDispatcher()
        logger.info("NeuroAct Orchestrator initialized")
    
    @log_execution_time
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Main entry point - process user query end-to-end"""
        logger.info(f"Processing query: {user_query}")
        
        try:
            # Step 1: Parse query into tasks
            tasks = self.task_parser.parse(user_query)
            logger.info(f"Parsed {len(tasks)} tasks from query")
            
            # Step 2: Execute tasks respecting dependencies
            results = []
            completed_tasks = set()
            
            # Sort tasks by dependencies (simple topological sort)
            sorted_tasks = self._sort_by_dependencies(tasks)
            
            for task in sorted_tasks:
                # Check if dependencies are completed
                if all(dep in completed_tasks for dep in task.dependencies):
                    result = self.dispatcher.dispatch(task)
                    results.append(result)
                    
                    if result['status'] == 'completed':
                        completed_tasks.add(task.task_id)
                    
                    logger.info(f"Completed task {task.task_id}: {result['status']}")
                else:
                    logger.error(f"Task {task.task_id} dependencies not met")
                    results.append({
                        "task_id": task.task_id,
                        "status": "failed",
                        "error": "Dependencies not completed",
                        "agent": task.agent_role
                    })
            
            # Step 3: Compile final response
            response = self._compile_response(user_query, tasks, results)
            logger.info("Query processing completed successfully")
            
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "query": user_query
            }
    
    def _compile_response(self, query: str, tasks: List[TaskManager], results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile final response from all task results"""
        
        successful_tasks = [r for r in results if r["status"] == "completed"]
        failed_tasks = [r for r in results if r["status"] == "failed"]
        
        response = {
            "status": "completed" if not failed_tasks else "partial",
            "query": query,
            "total_tasks": len(tasks),
            "successful_tasks": len(successful_tasks),
            "failed_tasks": len(failed_tasks),
            "results": results,
            "summary": self._generate_summary(successful_tasks, failed_tasks)
        }
        
        return response
    
    def _generate_summary(self, successful: List[Dict], failed: List[Dict]) -> str:
        """Generate human-readable summary"""
        
        if not failed:
            return f"Successfully completed all {len(successful)} tasks. All agents executed without errors."
        elif not successful:
            return f"All {len(failed)} tasks failed. Please check the error messages."
        else:
            return f"Completed {len(successful)} tasks successfully, {len(failed)} tasks failed. Partial results available."
    
    def _sort_by_dependencies(self, tasks: List[TaskManager]) -> List[TaskManager]:
        """Simple topological sort for task dependencies"""
        # Create task lookup
        task_map = {task.task_id: task for task in tasks}
        
        # Simple dependency sort - tasks with no deps first
        sorted_tasks = []
        remaining_tasks = tasks.copy()
        
        while remaining_tasks:
            # Find tasks with no unmet dependencies
            ready_tasks = []
            for task in remaining_tasks:
                if all(dep in [t.task_id for t in sorted_tasks] for dep in task.dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # If no tasks are ready, add remaining tasks (handles circular deps)
                sorted_tasks.extend(remaining_tasks)
                break
            
            # Add ready tasks and remove from remaining
            sorted_tasks.extend(ready_tasks)
            for task in ready_tasks:
                remaining_tasks.remove(task)
        
        return sorted_tasks
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        agent_status = self.dispatcher.get_agent_status()
        
        return {
            "orchestrator": "ready",
            "task_parser": "ready",
            "dispatcher": "ready",
            "agents": agent_status,
            "total_agents": len(agent_status)
        }