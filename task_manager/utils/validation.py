"""
Validation Utilities for Standardized Message Formats

Provides validation functions for all standardized schemas.
Use these to ensure data integrity during migration and production.
"""

from typing import Any, Dict, List, Optional, Tuple
import re
from datetime import datetime

from task_manager.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# VALIDATION RESULTS
# ============================================================================

class ValidationResult:
    """Result of a validation check."""
    
    def __init__(self, valid: bool, errors: Optional[List[str]] = None):
        self.valid = valid
        self.errors = errors or []
    
    def __bool__(self) -> bool:
        return self.valid
    
    def __str__(self) -> str:
        if self.valid:
            return "Validation passed"
        return f"Validation failed: {'; '.join(self.errors)}"


# ============================================================================
# AGENT RESPONSE VALIDATION
# ============================================================================

def validate_agent_execution_response(response: Dict[str, Any]) -> ValidationResult:
    """
    Validate AgentExecutionResponse schema.
    
    Args:
        response: Response dict to validate
    
    Returns:
        ValidationResult with any errors
    """
    errors = []
    
    # Required fields
    required_fields = {
        "status", "success", "result", "artifacts",
        "execution_time_ms", "timestamp", "agent_name", "operation"
    }
    
    missing_fields = required_fields - set(response.keys())
    if missing_fields:
        errors.append(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Type validation
    if "success" in response and not isinstance(response["success"], bool):
        errors.append("'success' must be boolean")
    
    if "status" in response and response["status"] not in [
        "success", "partial_success", "failure", "requires_human_input"
    ]:
        errors.append(f"Invalid status: {response['status']}")
    
    if "execution_time_ms" in response and not isinstance(response["execution_time_ms"], int):
        errors.append("'execution_time_ms' must be integer")
    
    if "timestamp" in response and not _is_valid_iso8601(response["timestamp"]):
        errors.append(f"Invalid timestamp format: {response.get('timestamp')}")
    
    if "artifacts" in response and not isinstance(response["artifacts"], list):
        errors.append("'artifacts' must be list")
    
    # Validate artifacts
    if "artifacts" in response:
        for i, artifact in enumerate(response["artifacts"]):
            artifact_errors = _validate_artifact(artifact)
            if artifact_errors:
                errors.append(f"Artifact {i}: {'; '.join(artifact_errors)}")
    
    # Validate confidence scores
    if "confidence_score" in response:
        score = response["confidence_score"]
        if score is not None and (not isinstance(score, (int, float)) or not 0 <= score <= 1):
            errors.append("'confidence_score' must be between 0 and 1")
    
    if "completeness_score" in response:
        score = response["completeness_score"]
        if score is not None and (not isinstance(score, (int, float)) or not 0 <= score <= 1):
            errors.append("'completeness_score' must be between 0 and 1")
    
    return ValidationResult(valid=len(errors) == 0, errors=errors)


def _validate_artifact(artifact: Dict[str, Any]) -> List[str]:
    """Validate artifact metadata."""
    errors = []
    
    required = {"type", "path", "size_bytes", "description"}
    missing = required - set(artifact.keys())
    if missing:
        errors.append(f"Missing artifact fields: {', '.join(missing)}")
    
    if "size_bytes" in artifact and not isinstance(artifact["size_bytes"], int):
        errors.append("'size_bytes' must be integer")
    
    if "size_bytes" in artifact and artifact["size_bytes"] < 0:
        errors.append("'size_bytes' cannot be negative")
    
    return errors


# ============================================================================
# EVENT VALIDATION
# ============================================================================

def validate_system_event(event: Dict[str, Any]) -> ValidationResult:
    """
    Validate SystemEvent schema.
    
    Args:
        event: Event dict to validate
    
    Returns:
        ValidationResult with any errors
    """
    errors = []
    
    # Required fields
    required_fields = {
        "event_id", "event_type", "event_category", "source_agent",
        "payload", "timestamp", "severity", "propagate", "listeners"
    }
    
    missing_fields = required_fields - set(event.keys())
    if missing_fields:
        errors.append(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Validate event_category
    valid_categories = {
        "task_lifecycle", "data_flow", "agent_execution",
        "human_interaction", "system_state"
    }
    if "event_category" in event and event["event_category"] not in valid_categories:
        errors.append(f"Invalid event_category: {event['event_category']}")
    
    # Validate severity
    valid_severities = {"debug", "info", "warning", "error", "critical"}
    if "severity" in event and event["severity"] not in valid_severities:
        errors.append(f"Invalid severity: {event['severity']}")
    
    # Validate timestamp
    if "timestamp" in event and not _is_valid_iso8601(event["timestamp"]):
        errors.append(f"Invalid timestamp format: {event.get('timestamp')}")
    
    # Validate propagate is boolean
    if "propagate" in event and not isinstance(event["propagate"], bool):
        errors.append("'propagate' must be boolean")
    
    # Validate listeners is list
    if "listeners" in event and not isinstance(event["listeners"], list):
        errors.append("'listeners' must be list")
    
    return ValidationResult(valid=len(errors) == 0, errors=errors)


# ============================================================================
# STORAGE SCHEMA VALIDATION
# ============================================================================

def validate_temp_data_schema(data: Dict[str, Any]) -> ValidationResult:
    """
    Validate TempDataSchema.
    
    Args:
        data: Temp data dict to validate
    
    Returns:
        ValidationResult with any errors
    """
    errors = []
    
    required_fields = {
        "schema_version", "data_type", "created_at", "updated_at",
        "key", "session_id", "data"
    }
    
    missing_fields = required_fields - set(data.keys())
    if missing_fields:
        errors.append(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Validate schema_version format (e.g., "1.0")
    if "schema_version" in data and not re.match(r'^\d+\.\d+$', data["schema_version"]):
        errors.append(f"Invalid schema_version format: {data.get('schema_version')}")
    
    # Validate timestamps
    for field in ["created_at", "updated_at", "expires_at"]:
        if field in data and data[field] and not _is_valid_iso8601(data[field]):
            errors.append(f"Invalid {field} format: {data.get(field)}")
    
    # Validate TTL
    if "ttl_hours" in data:
        ttl = data["ttl_hours"]
        if ttl is not None and (not isinstance(ttl, int) or ttl < 0):
            errors.append("'ttl_hours' must be non-negative integer or None")
    
    return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_cache_entry_schema(entry: Dict[str, Any]) -> ValidationResult:
    """
    Validate CacheEntrySchema.
    
    Args:
        entry: Cache entry dict to validate
    
    Returns:
        ValidationResult with any errors
    """
    errors = []
    
    required_fields = {
        "namespace", "key", "cached_at", "ttl_seconds", "hit_count",
        "input_hash", "output_data", "agent_name", "operation", "execution_time_ms"
    }
    
    missing_fields = required_fields - set(entry.keys())
    if missing_fields:
        errors.append(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Validate cached_at
    if "cached_at" in entry and not _is_valid_iso8601(entry["cached_at"]):
        errors.append(f"Invalid cached_at format: {entry.get('cached_at')}")
    
    # Validate numeric fields
    if "ttl_seconds" in entry and (not isinstance(entry["ttl_seconds"], int) or entry["ttl_seconds"] < 0):
        errors.append("'ttl_seconds' must be non-negative integer")
    
    if "hit_count" in entry and (not isinstance(entry["hit_count"], int) or entry["hit_count"] < 0):
        errors.append("'hit_count' must be non-negative integer")
    
    if "execution_time_ms" in entry and (not isinstance(entry["execution_time_ms"], int) or entry["execution_time_ms"] < 0):
        errors.append("'execution_time_ms' must be non-negative integer")
    
    # Validate input_hash format (should be SHA-256 hex)
    if "input_hash" in entry and not re.match(r'^[a-f0-9]{64}$', entry["input_hash"].lower()):
        errors.append("'input_hash' should be SHA-256 hex (64 characters)")
    
    return ValidationResult(valid=len(errors) == 0, errors=errors)


# ============================================================================
# ERROR RESPONSE VALIDATION
# ============================================================================

def validate_error_response(error: Dict[str, Any]) -> ValidationResult:
    """
    Validate ErrorResponse schema.
    
    Args:
        error: Error dict to validate
    
    Returns:
        ValidationResult with any errors
    """
    errors = []
    
    required_fields = {
        "error_id", "timestamp", "error_code", "error_type", "severity",
        "message", "details", "source", "recoverable", "recovery_suggestions"
    }
    
    missing_fields = required_fields - set(error.keys())
    if missing_fields:
        errors.append(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Validate error_type
    valid_types = {
        "validation_error", "execution_error", "timeout_error",
        "resource_error", "dependency_error"
    }
    if "error_type" in error and error["error_type"] not in valid_types:
        errors.append(f"Invalid error_type: {error['error_type']}")
    
    # Validate severity
    valid_severities = {"low", "medium", "high", "critical"}
    if "severity" in error and error["severity"] not in valid_severities:
        errors.append(f"Invalid severity: {error['severity']}")
    
    # Validate timestamp
    if "timestamp" in error and not _is_valid_iso8601(error["timestamp"]):
        errors.append(f"Invalid timestamp format: {error.get('timestamp')}")
    
    # Validate recoverable is boolean
    if "recoverable" in error and not isinstance(error["recoverable"], bool):
        errors.append("'recoverable' must be boolean")
    
    # Validate recovery_suggestions is list
    if "recovery_suggestions" in error and not isinstance(error["recovery_suggestions"], list):
        errors.append("'recovery_suggestions' must be list")
    
    # Validate retry_after_seconds
    if "retry_after_seconds" in error:
        retry = error["retry_after_seconds"]
        if retry is not None and (not isinstance(retry, int) or retry < 0):
            errors.append("'retry_after_seconds' must be non-negative integer or None")
    
    return ValidationResult(valid=len(errors) == 0, errors=errors)


# ============================================================================
# HUMAN INPUT VALIDATION
# ============================================================================

def validate_human_input_request(request: Dict[str, Any]) -> ValidationResult:
    """
    Validate HumanInputRequest schema.
    
    Args:
        request: Request dict to validate
    
    Returns:
        ValidationResult with any errors
    """
    errors = []
    
    required_fields = {
        "request_id", "request_type", "task_id", "task_description",
        "current_state", "prompt", "background"
    }
    
    missing_fields = required_fields - set(request.keys())
    if missing_fields:
        errors.append(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Validate request_type
    valid_types = {
        "review_required", "clarification_needed",
        "decision_needed", "approval_required"
    }
    if "request_type" in request and request["request_type"] not in valid_types:
        errors.append(f"Invalid request_type: {request['request_type']}")
    
    # Validate timeout_seconds
    if "timeout_seconds" in request:
        timeout = request["timeout_seconds"]
        if timeout is not None and (not isinstance(timeout, int) or timeout < 0):
            errors.append("'timeout_seconds' must be non-negative integer or None")
    
    return ValidationResult(valid=len(errors) == 0, errors=errors)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _is_valid_iso8601(timestamp: str) -> bool:
    """Check if string is valid ISO 8601 timestamp."""
    try:
        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return True
    except (ValueError, AttributeError):
        return False


# ============================================================================
# BATCH VALIDATION
# ============================================================================

def validate_all_schemas(data: Dict[str, Any], schema_type: str) -> ValidationResult:
    """
    Validate data against specified schema type.
    
    Args:
        data: Data to validate
        schema_type: One of: 'agent_response', 'event', 'temp_data',
                    'cache_entry', 'error', 'human_request'
    
    Returns:
        ValidationResult
    """
    validators = {
        "agent_response": validate_agent_execution_response,
        "event": validate_system_event,
        "temp_data": validate_temp_data_schema,
        "cache_entry": validate_cache_entry_schema,
        "error": validate_error_response,
        "human_request": validate_human_input_request
    }
    
    if schema_type not in validators:
        return ValidationResult(
            valid=False,
            errors=[f"Unknown schema type: {schema_type}"]
        )
    
    return validators[schema_type](data)


# ============================================================================
# LOGGING VALIDATION DECORATOR
# ============================================================================

def log_validation_errors(func):
    """Decorator to log validation errors."""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if not result.valid:
            logger.warning(f"Validation failed in {func.__name__}: {result.errors}")
        return result
    return wrapper
