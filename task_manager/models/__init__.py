"""
Models module - Data structures and enums for Task Manager Agent
"""

from .enums import TaskStatus
from .task import Task
from .state import AgentState, PlanNode, BlackboardEntry, HistoryEntry

# Standardized message formats (NEW - v1.0)
from .messages import (
    MessageEnvelope,
    AgentExecutionRequest,
    AgentExecutionResponse,
    ArtifactMetadata,
    HumanInputRequest,
    HumanInputResponse,
    SystemEvent,
    TempDataSchema,
    CacheEntrySchema,
    LLMRequest,
    LLMResponse,
    ErrorResponse,
    EVENT_TYPE_REGISTRY,
    create_message_envelope,
    create_system_event,
    create_error_response
)

__all__ = [
    # Legacy models
    'TaskStatus',
    'Task',
    'AgentState',
    'PlanNode',
    'BlackboardEntry',
    'HistoryEntry',
    
    # Standardized message formats (v1.0)
    'MessageEnvelope',
    'AgentExecutionRequest',
    'AgentExecutionResponse',
    'ArtifactMetadata',
    'HumanInputRequest',
    'HumanInputResponse',
    'SystemEvent',
    'TempDataSchema',
    'CacheEntrySchema',
    'LLMRequest',
    'LLMResponse',
    'ErrorResponse',
    'EVENT_TYPE_REGISTRY',
    'create_message_envelope',
    'create_system_event',
    'create_error_response'
]

