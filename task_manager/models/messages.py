"""
Standardized Message Formats for TaskManager System

This module defines all message schemas for inter-component communication.
Based on: INTERFACE_STANDARDS.md v1.0

Key Principles:
- All data exchanges use standardized envelopes
- Type safety via TypedDict
- Event-driven architecture support
- Backward compatible with existing code
"""

from typing import TypedDict, Optional, Any, Literal, NotRequired
from datetime import datetime
import uuid


# ============================================================================
# CORE MESSAGE ENVELOPE
# ============================================================================

class MessageEnvelope(TypedDict):
    """
    Universal message wrapper for all system communications.
    
    Usage:
        msg = MessageEnvelope(
            message_id=str(uuid.uuid4()),
            correlation_id="task_1.2.3_exec",
            timestamp=datetime.now().isoformat(),
            source="web_search_agent",
            destination="master_planner",
            message_type="response",
            payload={"status": "success", "data": {...}},
            metadata={"task_id": "task_1.2.3"},
            priority=5,
            retry_count=0,
            ttl_seconds=3600
        )
    """
    # Message Identity
    message_id: str              # UUID v4
    correlation_id: str          # Links related messages (request→response)
    timestamp: str               # ISO 8601 format
    
    # Message Routing
    source: str                  # Origin (e.g., "web_search_agent", "human")
    destination: str             # Target (e.g., "excel_agent", "blackboard")
    message_type: Literal[
        "command",               # Execute action
        "query",                 # Request information
        "event",                 # Notify of occurrence
        "response",              # Reply to command/query
        "error"                  # Error notification
    ]
    
    # Message Content
    payload: dict[str, Any]      # Actual data (schema depends on message_type)
    metadata: dict[str, Any]     # Optional context (task_id, session_id, etc.)
    
    # Message Control
    priority: int                # 1-10 (1=highest, 10=lowest)
    retry_count: int             # For error recovery
    ttl_seconds: NotRequired[Optional[int]]  # Time-to-live for caching


# ============================================================================
# AGENT INTERFACE
# ============================================================================

class AgentExecutionRequest(TypedDict):
    """
    Standard format for requesting agent execution.
    
    Sent by: Master Planner, Workflow Manager
    Received by: Sub-agents (PDF, Excel, Web Search, etc.)
    """
    # Task Context
    task_id: str
    task_description: str
    task_type: Literal["atomic", "composite", "research", "analysis"]
    
    # Execution Parameters
    operation: str               # e.g., "search", "extract_text", "analyze_data"
    parameters: dict[str, Any]   # Operation-specific parameters
    
    # Input Data
    input_data: dict[str, Any]   # Structured input (files, queries, etc.)
    input_context: NotRequired[Optional[dict[str, Any]]]  # Additional context
    
    # Execution Environment
    temp_folder: str             # Where to store intermediate files
    output_folder: str           # Where to store final outputs
    cache_enabled: bool          # Whether to use cached results
    
    # Blackboard Access
    blackboard: list[dict[str, Any]]  # Current blackboard entries (BlackboardEntry compatible)
    relevant_entries: list[str]  # Entry IDs relevant to this task
    
    # Constraints
    timeout_seconds: NotRequired[Optional[int]]
    max_retries: int
    quality_threshold: NotRequired[Optional[float]]


class AgentExecutionResponse(TypedDict):
    """
    Standard format for agent execution results.
    
    Sent by: Sub-agents
    Received by: Master Planner, Workflow Manager
    """
    # Execution Status
    status: Literal["success", "partial_success", "failure", "requires_human_input"]
    success: bool                # Quick boolean check
    
    # Results
    result: dict[str, Any]       # Primary output data
    artifacts: list[dict[str, Any]]  # Generated files/objects
    
    # Metadata
    execution_time_ms: int
    timestamp: str               # ISO 8601
    agent_name: str
    operation: str
    
    # Blackboard Contributions
    blackboard_entries: list[dict[str, Any]]  # New entries to add to blackboard (BlackboardEntry compatible)
    
    # Chain Execution (Event Triggers)
    next_agents: NotRequired[Optional[list[str]]]  # Agents to execute next
    chain_data: NotRequired[Optional[dict]]        # Data for chained agents
    event_triggers: NotRequired[Optional[list[str]]]  # Events raised
    
    # Error Handling
    error: NotRequired[Optional[str]]  # Error message if status != success
    warnings: list[str]                # Non-fatal issues
    
    # Quality Metrics
    confidence_score: NotRequired[Optional[float]]    # 0.0-1.0
    completeness_score: NotRequired[Optional[float]]  # 0.0-1.0


class ArtifactMetadata(TypedDict):
    """Metadata for generated artifacts (files, objects)."""
    type: str                    # "csv", "json", "pdf", "image", etc.
    path: str                    # Absolute path to artifact
    size_bytes: int
    description: str
    mime_type: NotRequired[Optional[str]]
    checksum: NotRequired[Optional[str]]  # SHA-256 hash


# ============================================================================
# HUMAN INTERACTION
# ============================================================================

class HumanInputRequest(TypedDict):
    """Request for human input/review."""
    # Request Identity
    request_id: str
    request_type: Literal[
        "review_required",
        "clarification_needed",
        "decision_needed",
        "approval_required"
    ]
    
    # Context
    task_id: str
    task_description: str
    current_state: dict[str, Any]
    
    # Question/Prompt
    prompt: str                  # Human-readable question
    options: NotRequired[Optional[list[str]]]  # Available choices
    
    # Additional Information
    background: str              # Context for decision
    recommendations: NotRequired[Optional[dict]]  # AI recommendations
    
    # Constraints
    timeout_seconds: NotRequired[Optional[int]]
    default_action: NotRequired[Optional[str]]  # If timeout occurs


class HumanInputResponse(TypedDict):
    """Response from human operator."""
    # Response Identity
    request_id: str              # Links to original request
    response_id: str
    timestamp: str
    
    # Response Data
    response_type: Literal["approval", "rejection", "modification", "clarification"]
    response_data: dict[str, Any]  # Actual response content
    
    # Metadata
    operator_id: NotRequired[Optional[str]]
    response_time_seconds: int
    confidence: NotRequired[Optional[float]]  # How confident human is
    notes: NotRequired[Optional[str]]         # Additional comments


# ============================================================================
# EVENT SYSTEM
# ============================================================================

class SystemEvent(TypedDict):
    """
    Standard event format for event-driven architecture.
    
    Published by: Any component
    Consumed by: Event subscribers (registered via EventBus)
    """
    # Event Identity
    event_id: str                # UUID
    event_type: str              # e.g., "task_completed", "data_extracted"
    event_category: Literal[
        "task_lifecycle",
        "data_flow",
        "agent_execution",
        "human_interaction",
        "system_state"
    ]
    
    # Event Source
    source_agent: str
    source_task_id: NotRequired[Optional[str]]
    
    # Event Payload
    payload: dict[str, Any]
    
    # Event Metadata
    timestamp: str
    severity: Literal["debug", "info", "warning", "error", "critical"]
    
    # Event Propagation
    propagate: bool              # Should event trigger listeners?
    listeners: list[str]         # Which agents should be notified


# Event Type Registry - Maps event types to default listeners
EVENT_TYPE_REGISTRY: dict[str, list[str]] = {
    # Task Lifecycle Events
    "task_created": ["master_planner"],
    "task_started": ["monitoring_service"],
    "task_completed": ["master_planner", "synthesis_agent"],
    "task_failed": ["error_handler", "master_planner"],
    
    # Data Flow Events
    "ocr_results_ready": ["auto_synthesis", "excel_agent"],
    "web_findings_ready": ["auto_synthesis", "data_extraction_agent"],
    "file_generated": ["file_router"],
    "data_extracted": ["blackboard_manager"],
    
    # Agent Execution Events
    "agent_execution_started": ["monitoring_service"],
    "agent_execution_completed": ["monitoring_service", "master_planner"],
    "chain_execution_triggered": ["workflow_manager"],
    
    # Human Interaction Events
    "human_input_requested": ["ui_service"],
    "human_input_received": ["task_manager"],
    
    # System State Events
    "contradiction_detected": ["agentic_debate"],
    "synthesis_required": ["synthesis_agent"],
    "blackboard_updated": ["all_agents"]
}


# ============================================================================
# STORAGE FORMATS
# ============================================================================

class TempDataSchema(TypedDict):
    """Standardized temp file format for TempDataManager."""
    # Metadata
    schema_version: str          # "1.0"
    data_type: str               # "task_result", "cache_entry", "checkpoint"
    created_at: str
    updated_at: str
    
    # Identification
    key: str
    task_id: NotRequired[Optional[str]]
    session_id: str
    
    # Content
    data: dict[str, Any]         # Actual payload
    
    # Lifecycle
    ttl_hours: NotRequired[Optional[int]]
    expires_at: NotRequired[Optional[str]]
    
    # Provenance
    source_agent: NotRequired[Optional[str]]
    source_operation: NotRequired[Optional[str]]


class CacheEntrySchema(TypedDict):
    """Redis cache entry format for RedisCacheManager."""
    # Cache Key Components
    namespace: str               # "task_results", "llm_responses", "web_searches"
    key: str                     # Unique identifier
    
    # Cache Metadata
    cached_at: str
    ttl_seconds: int
    hit_count: int
    
    # Cached Data
    input_hash: str              # SHA-256 of input for validation
    output_data: dict[str, Any]
    
    # Provenance
    agent_name: str
    operation: str
    execution_time_ms: int


# ============================================================================
# LLM COMMUNICATION
# ============================================================================

class LLMRequest(TypedDict):
    """Standardized LLM API request."""
    # Request Identity
    request_id: str
    timestamp: str
    
    # Model Selection
    provider: str                # "google", "openai", "anthropic"
    model: str                   # "gemini-2.0-flash", "gpt-4-turbo"
    
    # Prompt
    system_prompt: NotRequired[Optional[str]]
    user_prompt: str
    context: NotRequired[Optional[list[dict]]]  # Conversation history
    
    # Parameters
    temperature: float
    max_tokens: int
    top_p: NotRequired[Optional[float]]
    
    # Metadata
    task_id: NotRequired[Optional[str]]
    operation: str               # "analyze_task", "synthesize", etc.
    
    # Response Format
    response_format: Literal["text", "json", "structured"]
    json_schema: NotRequired[Optional[dict]]  # For structured output


class LLMResponse(TypedDict):
    """Standardized LLM API response."""
    # Response Identity
    request_id: str              # Links to request
    response_id: str
    timestamp: str
    
    # Response Content
    content: str | dict          # Text or JSON
    finish_reason: str           # "stop", "length", "error"
    
    # Usage Metrics
    tokens_input: int
    tokens_output: int
    tokens_total: int
    cost_usd: NotRequired[Optional[float]]
    
    # Performance
    latency_ms: int
    
    # Quality
    confidence: NotRequired[Optional[float]]
    
    # Provider Info
    provider: str
    model: str
    model_version: NotRequired[Optional[str]]


# ============================================================================
# ERROR HANDLING
# ============================================================================

class ErrorResponse(TypedDict):
    """Standardized error format."""
    # Error Identity
    error_id: str                # UUID for tracking
    timestamp: str
    
    # Error Classification
    error_code: str              # e.g., "AGENT_EXEC_001", "LLM_TIMEOUT_002"
    error_type: Literal[
        "validation_error",
        "execution_error",
        "timeout_error",
        "resource_error",
        "dependency_error"
    ]
    severity: Literal["low", "medium", "high", "critical"]
    
    # Error Details
    message: str                 # Human-readable message
    details: dict[str, Any]      # Technical details
    
    # Context
    source: str                  # Where error occurred
    task_id: NotRequired[Optional[str]]
    operation: NotRequired[Optional[str]]
    
    # Stack Trace
    stack_trace: NotRequired[Optional[str]]
    
    # Recovery
    recoverable: bool
    recovery_suggestions: list[str]
    retry_after_seconds: NotRequired[Optional[int]]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_message_envelope(
    source: str,
    destination: str,
    message_type: Literal["command", "query", "event", "response", "error"],
    payload: dict[str, Any],
    metadata: Optional[dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
    priority: int = 5,
    ttl_seconds: Optional[int] = None
) -> MessageEnvelope:
    """
    Helper function to create standardized message envelope.
    
    Args:
        source: Message source (agent name or "human")
        destination: Message destination
        message_type: Type of message
        payload: Message content
        metadata: Optional metadata
        correlation_id: Optional correlation ID (auto-generated if None)
        priority: Message priority (1-10, default 5)
        ttl_seconds: Time-to-live in seconds
    
    Returns:
        MessageEnvelope instance
    """
    return MessageEnvelope(
        message_id=str(uuid.uuid4()),
        correlation_id=correlation_id or str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        source=source,
        destination=destination,
        message_type=message_type,
        payload=payload,
        metadata=metadata or {},
        priority=priority,
        retry_count=0,
        ttl_seconds=ttl_seconds
    )


def create_system_event(
    event_type: str,
    event_category: Literal[
        "task_lifecycle", "data_flow", "agent_execution",
        "human_interaction", "system_state"
    ],
    source_agent: str,
    payload: dict[str, Any],
    source_task_id: Optional[str] = None,
    severity: Literal["debug", "info", "warning", "error", "critical"] = "info",
    propagate: bool = True,
    listeners: Optional[list[str]] = None
) -> SystemEvent:
    """
    Helper function to create system events.
    
    Args:
        event_type: Type of event (e.g., "task_completed")
        event_category: Category of event
        source_agent: Agent that generated the event
        payload: Event data
        source_task_id: Optional task ID
        severity: Event severity level
        propagate: Whether to propagate to listeners
        listeners: Optional list of listeners (uses registry if None)
    
    Returns:
        SystemEvent instance
    """
    # Use event type registry if listeners not specified
    if listeners is None:
        listeners = EVENT_TYPE_REGISTRY.get(event_type, [])
    
    return SystemEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        event_category=event_category,
        source_agent=source_agent,
        source_task_id=source_task_id,
        payload=payload,
        timestamp=datetime.now().isoformat(),
        severity=severity,
        propagate=propagate,
        listeners=listeners
    )


def create_error_response(
    error_code: str,
    error_type: Literal[
        "validation_error", "execution_error", "timeout_error",
        "resource_error", "dependency_error"
    ],
    message: str,
    source: str,
    severity: Literal["low", "medium", "high", "critical"] = "medium",
    details: Optional[dict[str, Any]] = None,
    task_id: Optional[str] = None,
    operation: Optional[str] = None,
    stack_trace: Optional[str] = None,
    recoverable: bool = True,
    recovery_suggestions: Optional[list[str]] = None,
    retry_after_seconds: Optional[int] = None
) -> ErrorResponse:
    """
    Helper function to create standardized error responses.
    
    Args:
        error_code: Error code (e.g., "AGENT_EXEC_001")
        error_type: Type of error
        message: Human-readable error message
        source: Where error occurred
        severity: Error severity
        details: Technical details
        task_id: Optional task ID
        operation: Optional operation name
        stack_trace: Optional stack trace
        recoverable: Whether error is recoverable
        recovery_suggestions: List of recovery suggestions
        retry_after_seconds: Retry delay
    
    Returns:
        ErrorResponse instance
    """
    return ErrorResponse(
        error_id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        error_code=error_code,
        error_type=error_type,
        severity=severity,
        message=message,
        details=details or {},
        source=source,
        task_id=task_id,
        operation=operation,
        stack_trace=stack_trace,
        recoverable=recoverable,
        recovery_suggestions=recovery_suggestions or [],
        retry_after_seconds=retry_after_seconds
    )
