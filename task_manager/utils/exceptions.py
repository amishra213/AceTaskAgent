"""
Standardized Exception Hierarchy for TaskManager

This module provides a comprehensive exception hierarchy for consistent error handling
across all agents and modules in the TaskManager system.

Exception Categories:
- Configuration Errors: Issues with settings, environment, or initialization
- Validation Errors: Input validation failures
- Execution Errors: Runtime operation failures
- Resource Errors: File, network, or external resource issues
- Agent-Specific Errors: Specialized exceptions for different agent types

Usage:
    from task_manager.utils.exceptions import (
        TaskManagerError,
        AgentExecutionError,
        InvalidParameterError
    )
    
    # In your agent:
    if not query:
        raise InvalidParameterError("query", "Search query cannot be empty")
    
    try:
        result = perform_operation()
    except Exception as e:
        raise AgentExecutionError(
            "search",
            "Failed to execute web search",
            original_error=e
        )
"""

from typing import Optional, Any, Dict


# ============================================================================
# Base Exception
# ============================================================================

class TaskManagerError(Exception):
    """
    Base exception for all TaskManager errors.
    
    All custom exceptions should inherit from this class to enable
    centralized error handling and logging.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base exception.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for categorization
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }
    
    def __str__(self) -> str:
        """String representation of the error."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# ============================================================================
# Configuration and Initialization Errors
# ============================================================================

class ConfigurationError(TaskManagerError):
    """Raised when there's an issue with configuration or settings."""
    
    def __init__(
        self,
        setting_name: str,
        message: str,
        expected_value: Optional[Any] = None,
        actual_value: Optional[Any] = None
    ):
        details = {"setting_name": setting_name}
        if expected_value is not None:
            details["expected_value"] = str(expected_value)
        if actual_value is not None:
            details["actual_value"] = str(actual_value)
        
        super().__init__(
            message=f"Configuration error for '{setting_name}': {message}",
            error_code="CONFIG_ERROR",
            details=details
        )
        self.setting_name = setting_name


class MissingDependencyError(TaskManagerError):
    """Raised when a required dependency is not installed."""
    
    def __init__(
        self,
        package_name: str,
        install_command: Optional[str] = None,
        purpose: Optional[str] = None
    ):
        message = f"Required package '{package_name}' is not installed"
        if purpose:
            message += f" (needed for {purpose})"
        if install_command:
            message += f"\nInstall with: {install_command}"
        
        super().__init__(
            message=message,
            error_code="MISSING_DEPENDENCY",
            details={
                "package_name": package_name,
                "install_command": install_command,
                "purpose": purpose
            }
        )
        self.package_name = package_name


class EnvironmentError(TaskManagerError):
    """Raised when environment setup is invalid or incomplete."""
    
    def __init__(
        self,
        variable_name: str,
        message: str,
        suggestion: Optional[str] = None
    ):
        full_message = f"Environment error for '{variable_name}': {message}"
        if suggestion:
            full_message += f"\nSuggestion: {suggestion}"
        
        super().__init__(
            message=full_message,
            error_code="ENV_ERROR",
            details={
                "variable_name": variable_name,
                "suggestion": suggestion
            }
        )
        self.variable_name = variable_name


# ============================================================================
# Validation Errors
# ============================================================================

class ValidationError(TaskManagerError):
    """Base class for validation errors."""
    pass


class InvalidParameterError(ValidationError):
    """Raised when a parameter value is invalid."""
    
    def __init__(
        self,
        parameter_name: str,
        message: str,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None
    ):
        details = {"parameter_name": parameter_name}
        if expected_type:
            details["expected_type"] = expected_type
        if actual_value is not None:
            details["actual_value"] = str(actual_value)
        
        super().__init__(
            message=f"Invalid parameter '{parameter_name}': {message}",
            error_code="INVALID_PARAM",
            details=details
        )
        self.parameter_name = parameter_name


class MissingParameterError(ValidationError):
    """Raised when a required parameter is missing."""
    
    def __init__(
        self,
        parameter_name: str,
        context: Optional[str] = None
    ):
        message = f"Missing required parameter: '{parameter_name}'"
        if context:
            message += f" in {context}"
        
        super().__init__(
            message=message,
            error_code="MISSING_PARAM",
            details={"parameter_name": parameter_name, "context": context}
        )
        self.parameter_name = parameter_name


class InvalidOperationError(ValidationError):
    """Raised when an unsupported operation is requested."""
    
    def __init__(
        self,
        operation: str,
        supported_operations: Optional[list] = None,
        agent_name: Optional[str] = None
    ):
        message = f"Unsupported operation: '{operation}'"
        if agent_name:
            message += f" for {agent_name}"
        if supported_operations:
            message += f"\nSupported operations: {', '.join(supported_operations)}"
        
        super().__init__(
            message=message,
            error_code="INVALID_OPERATION",
            details={
                "operation": operation,
                "supported_operations": supported_operations,
                "agent_name": agent_name
            }
        )
        self.operation = operation


class SchemaValidationError(ValidationError):
    """Raised when data doesn't match expected schema."""
    
    def __init__(
        self,
        schema_name: str,
        validation_errors: list,
        data_sample: Optional[Dict[str, Any]] = None
    ):
        message = f"Schema validation failed for '{schema_name}'"
        if validation_errors:
            message += f"\nErrors: {'; '.join(str(e) for e in validation_errors)}"
        
        super().__init__(
            message=message,
            error_code="SCHEMA_VALIDATION",
            details={
                "schema_name": schema_name,
                "validation_errors": validation_errors,
                "data_sample": data_sample
            }
        )
        self.schema_name = schema_name
        self.validation_errors = validation_errors


# ============================================================================
# Execution Errors
# ============================================================================

class ExecutionError(TaskManagerError):
    """Base class for execution-time errors."""
    pass


class AgentExecutionError(ExecutionError):
    """Raised when an agent fails to execute a task."""
    
    def __init__(
        self,
        operation: str,
        message: str,
        agent_name: Optional[str] = None,
        task_id: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        full_message = f"Agent execution failed for operation '{operation}': {message}"
        if agent_name:
            full_message = f"[{agent_name}] {full_message}"
        if original_error:
            full_message += f"\nCaused by: {str(original_error)}"
        
        super().__init__(
            message=full_message,
            error_code="AGENT_EXEC_ERROR",
            details={
                "operation": operation,
                "agent_name": agent_name,
                "task_id": task_id,
                "original_error": str(original_error) if original_error else None
            }
        )
        self.operation = operation
        self.agent_name = agent_name
        self.original_error = original_error


class TimeoutError(ExecutionError):
    """Raised when an operation exceeds time limit."""
    
    def __init__(
        self,
        operation: str,
        timeout_seconds: float,
        message: Optional[str] = None
    ):
        default_message = f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        full_message = message or default_message
        
        super().__init__(
            message=full_message,
            error_code="TIMEOUT",
            details={
                "operation": operation,
                "timeout_seconds": timeout_seconds
            }
        )
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class RetryExhaustedError(ExecutionError):
    """Raised when all retry attempts have been exhausted."""
    
    def __init__(
        self,
        operation: str,
        max_retries: int,
        last_error: Optional[Exception] = None
    ):
        message = f"Operation '{operation}' failed after {max_retries} retry attempts"
        if last_error:
            message += f"\nLast error: {str(last_error)}"
        
        super().__init__(
            message=message,
            error_code="RETRY_EXHAUSTED",
            details={
                "operation": operation,
                "max_retries": max_retries,
                "last_error": str(last_error) if last_error else None
            }
        )
        self.operation = operation
        self.max_retries = max_retries
        self.last_error = last_error


# ============================================================================
# Resource Errors
# ============================================================================

class ResourceError(TaskManagerError):
    """Base class for resource-related errors."""
    pass


class FileOperationError(ResourceError):
    """Raised when file operations fail."""
    
    def __init__(
        self,
        file_path: str,
        operation: str,
        message: str,
        original_error: Optional[Exception] = None
    ):
        full_message = f"File operation '{operation}' failed for '{file_path}': {message}"
        if original_error:
            full_message += f"\nCaused by: {str(original_error)}"
        
        super().__init__(
            message=full_message,
            error_code="FILE_OP_ERROR",
            details={
                "file_path": file_path,
                "operation": operation,
                "original_error": str(original_error) if original_error else None
            }
        )
        self.file_path = file_path
        self.operation = operation


class NetworkError(ResourceError):
    """Raised when network operations fail."""
    
    def __init__(
        self,
        url: str,
        message: str,
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None
    ):
        full_message = f"Network error for '{url}': {message}"
        if status_code:
            full_message += f" (status code: {status_code})"
        if original_error:
            full_message += f"\nCaused by: {str(original_error)}"
        
        super().__init__(
            message=full_message,
            error_code="NETWORK_ERROR",
            details={
                "url": url,
                "status_code": status_code,
                "original_error": str(original_error) if original_error else None
            }
        )
        self.url = url
        self.status_code = status_code


class CacheError(ResourceError):
    """Raised when cache operations fail."""
    
    def __init__(
        self,
        cache_key: str,
        operation: str,
        message: str,
        original_error: Optional[Exception] = None
    ):
        full_message = f"Cache operation '{operation}' failed for key '{cache_key}': {message}"
        if original_error:
            full_message += f"\nCaused by: {str(original_error)}"
        
        super().__init__(
            message=full_message,
            error_code="CACHE_ERROR",
            details={
                "cache_key": cache_key,
                "operation": operation,
                "original_error": str(original_error) if original_error else None
            }
        )
        self.cache_key = cache_key


# ============================================================================
# Agent-Specific Errors
# ============================================================================

class WebSearchError(AgentExecutionError):
    """Raised when web search operations fail."""
    
    def __init__(
        self,
        query: str,
        message: str,
        backend: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            operation="web_search",
            message=f"Search failed for query '{query}': {message}",
            agent_name="web_search_agent",
            original_error=original_error
        )
        self.query = query
        self.backend = backend
        if backend:
            self.details["backend"] = backend


class ScrapingError(AgentExecutionError):
    """Raised when web scraping operations fail."""
    
    def __init__(
        self,
        url: str,
        message: str,
        scraping_method: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            operation="web_scrape",
            message=f"Scraping failed for URL '{url}': {message}",
            agent_name="web_search_agent",
            original_error=original_error
        )
        self.url = url
        if scraping_method:
            self.details["scraping_method"] = scraping_method


class PDFOperationError(AgentExecutionError):
    """Raised when PDF operations fail."""
    
    def __init__(
        self,
        operation: str,
        file_path: str,
        message: str,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            operation=operation,
            message=f"PDF operation failed for '{file_path}': {message}",
            agent_name="pdf_agent",
            original_error=original_error
        )
        self.file_path = file_path


class ExcelOperationError(AgentExecutionError):
    """Raised when Excel operations fail."""
    
    def __init__(
        self,
        operation: str,
        file_path: str,
        message: str,
        sheet_name: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        full_message = f"Excel operation failed for '{file_path}': {message}"
        if sheet_name:
            full_message += f" (sheet: {sheet_name})"
        
        super().__init__(
            operation=operation,
            message=full_message,
            agent_name="excel_agent",
            original_error=original_error
        )
        self.file_path = file_path
        self.sheet_name = sheet_name


class OCRError(AgentExecutionError):
    """Raised when OCR operations fail."""
    
    def __init__(
        self,
        file_path: str,
        message: str,
        ocr_engine: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            operation="ocr",
            message=f"OCR failed for '{file_path}': {message}",
            agent_name="ocr_image_agent",
            original_error=original_error
        )
        self.file_path = file_path
        if ocr_engine:
            self.details["ocr_engine"] = ocr_engine


class CodeExecutionError(AgentExecutionError):
    """Raised when code execution fails."""
    
    def __init__(
        self,
        message: str,
        code_snippet: Optional[str] = None,
        execution_output: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            operation="code_execution",
            message=f"Code execution failed: {message}",
            agent_name="code_interpreter_agent",
            original_error=original_error
        )
        if code_snippet:
            self.details["code_snippet"] = code_snippet[:200]  # Limit size
        if execution_output:
            self.details["execution_output"] = execution_output[:200]


class DataExtractionError(AgentExecutionError):
    """Raised when data extraction fails."""
    
    def __init__(
        self,
        file_path: str,
        message: str,
        extraction_type: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            operation="data_extraction",
            message=f"Data extraction failed for '{file_path}': {message}",
            agent_name="data_extraction_agent",
            original_error=original_error
        )
        self.file_path = file_path
        if extraction_type:
            self.details["extraction_type"] = extraction_type


class LLMError(AgentExecutionError):
    """Raised when LLM operations fail."""
    
    def __init__(
        self,
        provider: str,
        message: str,
        model: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        full_message = f"LLM error with provider '{provider}': {message}"
        if model:
            full_message += f" (model: {model})"
        
        super().__init__(
            operation="llm_call",
            message=full_message,
            original_error=original_error
        )
        self.provider = provider
        self.model = model
        if provider:
            self.details["provider"] = provider
        if model:
            self.details["model"] = model


# ============================================================================
# Convenience Functions
# ============================================================================

def wrap_exception(
    original_error: Exception,
    operation: str,
    context: Optional[Dict[str, Any]] = None
) -> TaskManagerError:
    """
    Wrap a generic exception in an appropriate TaskManager exception.
    
    Args:
        original_error: The original exception to wrap
        operation: The operation that was being performed
        context: Additional context about the error
    
    Returns:
        An appropriate TaskManagerError subclass
    """
    context = context or {}
    
    # If already a TaskManager error, return as-is
    if isinstance(original_error, TaskManagerError):
        return original_error
    
    # Map common exception types to TaskManager exceptions
    if isinstance(original_error, (IOError, OSError)):
        file_path = context.get("file_path", "unknown")
        return FileOperationError(
            file_path=file_path,
            operation=operation,
            message=str(original_error),
            original_error=original_error
        )
    
    elif isinstance(original_error, ImportError):
        package_name = context.get("package_name", "unknown")
        return MissingDependencyError(
            package_name=package_name,
            purpose=operation,
            install_command=context.get("install_command")
        )
    
    elif isinstance(original_error, ValueError):
        param_name = context.get("parameter_name", "unknown")
        return InvalidParameterError(
            parameter_name=param_name,
            message=str(original_error)
        )
    
    elif isinstance(original_error, KeyError):
        param_name = str(original_error).strip("'\"")
        return MissingParameterError(
            parameter_name=param_name,
            context=operation
        )
    
    # Default: wrap in generic AgentExecutionError
    return AgentExecutionError(
        operation=operation,
        message=str(original_error),
        agent_name=context.get("agent_name"),
        task_id=context.get("task_id"),
        original_error=original_error
    )


__all__ = [
    # Base
    "TaskManagerError",
    
    # Configuration
    "ConfigurationError",
    "MissingDependencyError",
    "EnvironmentError",
    
    # Validation
    "ValidationError",
    "InvalidParameterError",
    "MissingParameterError",
    "InvalidOperationError",
    "SchemaValidationError",
    
    # Execution
    "ExecutionError",
    "AgentExecutionError",
    "TimeoutError",
    "RetryExhaustedError",
    
    # Resources
    "ResourceError",
    "FileOperationError",
    "NetworkError",
    "CacheError",
    
    # Agent-Specific
    "WebSearchError",
    "ScrapingError",
    "PDFOperationError",
    "ExcelOperationError",
    "OCRError",
    "CodeExecutionError",
    "DataExtractionError",
    "LLMError",
    
    # Utilities
    "wrap_exception",
]
