# To manage the task of each task define how the agent works
from pydantic import BaseModel
from typing import List, Dict ,Any
import uuid
import time
from app.utils.logger import get_logger

logger = get_logger(__name__)

#Define the task manager

class TaskMessage(BaseModel):
    task_id: str = str(uuid.uuid4())
    sender :str
    receiver: str
    agent_role :str
    inputs : Dict[str , Any]
    tools : List[str] = []
    expected_output : str = ""
    dependencies : List[str] = []
    metadata : Dict[str , Any] = {}
    status : str = "pending" # pending, in_progress, completed, failed
    

    @staticmethod
    def create(
        sender: str,
        receiver: str,
        agent_role: str,
        inputs: Dict[str, Any],
        tools: List[str] = None,
        expected_output: str = "",
        dependencies: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> "TaskMessage":
        """Helper function to create a new TaskManager with auto ID + timestamp."""
        return TaskMessage(
            task_id=str(uuid.uuid4()),
            sender=sender,
            receiver=receiver,
            agent_role=agent_role,
            inputs=inputs,
            tools=tools or [],
            expected_output=expected_output,
            dependencies=dependencies or [],
            metadata={**(metadata or {}), "timestamp": time.time()},
        )

# Alias for compatibility
TaskManager = TaskMessage