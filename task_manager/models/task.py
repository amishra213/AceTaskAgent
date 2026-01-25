"""
Task module - Individual task structure definition
"""

from typing import TypedDict, Optional
from .enums import TaskStatus


class Task(TypedDict):
    """Individual task structure with metadata and state information"""
    id: str
    description: str
    status: TaskStatus
    parent_id: Optional[str]
    depth: int
    context: str
    result: Optional[dict]
    error: Optional[str]
    created_at: str
    updated_at: str
