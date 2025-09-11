# To manage the task of each task define how the agent works
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import uuid
import time
from app.utils.logger import get_logger

logger = get_logger(__name__)

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaskResult(BaseModel):
    status: str
    output: Any = None
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TaskMessage(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender: str
    receiver: str
    agent_role: str
    inputs: Dict[str, Any]
    tools: List[str] = Field(default_factory=list)
    expected_output: str = ""
    dependencies: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_duration: Optional[int] = None  # seconds
    max_retries: int = 3
    timeout: Optional[int] = None  # seconds
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None  # seconds
    retry_count: int = 0
    result: Optional[TaskResult] = None
    
    @validator('task_id')
    def validate_task_id(cls, v):
        if not v or len(v) < 4:
            raise ValueError('Task ID must be at least 4 characters')
        return v
    
    @validator('agent_role')
    def validate_agent_role(cls, v):
        valid_agents = [
            'DataAgent', 'MLAgent', 'DeepLearningAgent', 'CodeAgent',
            'VisualizationAgent', 'CriticAgent', 'ModelEvaluationAgent', 
            'RetrievalAgent', 'PlannerAgent', 'NLPAgent'
        ]
        if v not in valid_agents:
            raise ValueError(f'Invalid agent role: {v}')
        return v
    
    def validate_dependencies(self, available_tasks: List[str]) -> bool:
        """Validate that all dependencies exist"""
        return all(dep in available_tasks for dep in self.dependencies)

    def is_ready_to_execute(self, completed_tasks: List[str]) -> bool:
        """Check if all dependencies are completed"""
        return all(dep in completed_tasks for dep in self.dependencies)
    
    def mark_in_progress(self):
        """Mark task as in progress"""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()

    def mark_completed(self, result: Dict[str, Any]):
        """Mark task as completed"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.result = TaskResult(**result)
        if self.started_at:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()
    
    def mark_failed(self, error: str):
        """Mark task as failed"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()
        self.result = TaskResult(status="failed", errors=[error])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return self.dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskMessage":
        """Create from dictionary"""
        return cls(**data)

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
        """Helper function to create a new TaskMessage with auto timestamp."""
        final_metadata = metadata or {}
        final_metadata["timestamp"] = time.time()
        
        return TaskMessage(
            sender=sender,
            receiver=receiver,
            agent_role=agent_role,
            inputs=inputs,
            tools=tools or [],
            expected_output=expected_output,
            dependencies=dependencies or [],
            metadata=final_metadata,
        )

# Alias for compatibility
TaskManager = TaskMessage