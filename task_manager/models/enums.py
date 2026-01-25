"""
Enums module - Task status and other enumeration types
"""

from enum import Enum


class TaskStatus(str, Enum):
    """Enumeration of possible task statuses"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    BROKEN_DOWN = "broken_down"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"
