"""Data schemas for tasks, context, and communication."""
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class Task:
    id: str
    description: str
    priority: int = 1
    status: str = "pending"

@dataclass
class AgentMessage:
    sender: str
    receiver: str
    content: Dict[str, Any]
    timestamp: str