# help to task manager  that receives the task message inot that correct agent

from typing import Dict, Callable
from app.mcp.mcp_schema import TaskMessage
from app.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class McpDispatcher :

    def __init__(self):
        """ Handler Function """
        self.agents: Dict[str, Callable[[TaskMessage], TaskMessage]] = {}

        # Agent Register 
    def register_agent(self, agent_name: str, handler: Callable[[TaskMessage], TaskMessage]):
        """ Register an agent with its handler function """
        self.agents[agent_name] = handler
    
    # Dispatch the message
    @log_execution_time
    def dispatch(self, msg: TaskMessage) -> TaskMessage:
        """ Dispatch the message to the appropriate agent """
        logger.info(f"Dispatching task {msg.task_id} to {msg.receiver}")
        name = msg.receiver
        if name in self.agents:
            handler = self.agents[name]
            logger.debug(f"Found handler for {name}")
            result = handler(msg)
            logger.info(f"Task {msg.task_id} completed successfully")
            return result
        else:
            logger.error(f"Agent {name} not registered")
            raise ValueError(f"Agent {name} not registered.")

McpDispatcher = McpDispatcher ()
