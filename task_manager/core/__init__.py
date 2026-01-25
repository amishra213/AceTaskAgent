"""
Core module - Task Manager Agent implementation and workflow
"""

from .agent import TaskManagerAgent
from .workflow import WorkflowBuilder

__all__ = [
    'TaskManagerAgent',
    'WorkflowBuilder',
]
