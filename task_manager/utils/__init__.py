"""
Utilities module - Helper functions and utilities
"""

from .logger import get_logger
from .comprehensive_logger import ComprehensiveLogger, TaskLogger
from .ai_context_logger import AIContextLogger, get_ai_logger, ai_log_info, ai_log_error, ai_log_warning, LogCategory
from .prompt_builder import PromptBuilder
from .llm_client import LLMClient, ResponseWrapper
from .rate_limiter import RateLimiter, global_rate_limiter
from .input_context import InputContext, FileInfo
from .temp_manager import TempDataManager, TempDataEntry
from .redis_cache import RedisCacheManager

# Standardization utilities (NEW - Week 1-2)
from .compatibility import (
    legacy_response_to_standard,
    standard_response_to_legacy,
    legacy_temp_data_to_standard,
    standard_temp_data_to_legacy,
    legacy_cache_to_standard,
    exception_to_error_response,
    is_standard_response,
    is_legacy_response,
    auto_convert_response
)
from .validation import (
    ValidationResult,
    validate_agent_execution_response,
    validate_system_event,
    validate_temp_data_schema,
    validate_cache_entry_schema,
    validate_error_response,
    validate_human_input_request,
    validate_all_schemas
)

# Exception hierarchy (NEW - Standardized error handling)
from .exceptions import (
    # Base
    TaskManagerError,
    # Configuration
    ConfigurationError,
    MissingDependencyError,
    EnvironmentError,
    # Validation
    ValidationError,
    InvalidParameterError,
    MissingParameterError,
    InvalidOperationError,
    SchemaValidationError,
    # Execution
    ExecutionError,
    AgentExecutionError,
    TimeoutError,
    RetryExhaustedError,
    # Resources
    ResourceError,
    FileOperationError,
    NetworkError,
    CacheError,
    # Agent-Specific
    WebSearchError,
    ScrapingError,
    PDFOperationError,
    ExcelOperationError,
    OCRError,
    CodeExecutionError,
    DataExtractionError,
    LLMError,
    # Utilities
    wrap_exception,
)

__all__ = [
    'get_logger',
    'ComprehensiveLogger',
    'TaskLogger',
    'AIContextLogger',
    'get_ai_logger',
    'ai_log_info',
    'ai_log_error',
    'ai_log_warning',
    'LogCategory',
    'PromptBuilder',
    'LLMClient',
    'ResponseWrapper',
    'RateLimiter',
    'global_rate_limiter',
    'InputContext',
    'FileInfo',
    'TempDataManager',
    'TempDataEntry',
    'RedisCacheManager',
    
    # Standardization utilities
    'legacy_response_to_standard',
    'standard_response_to_legacy',
    'legacy_temp_data_to_standard',
    'standard_temp_data_to_legacy',
    'legacy_cache_to_standard',
    'exception_to_error_response',
    'is_standard_response',
    'is_legacy_response',
    'auto_convert_response',
    'ValidationResult',
    'validate_agent_execution_response',
    'validate_system_event',
    'validate_temp_data_schema',
    'validate_cache_entry_schema',
    'validate_error_response',
    'validate_human_input_request',
    'validate_all_schemas',
    
    # Exception hierarchy
    'TaskManagerError',
    'ConfigurationError',
    'MissingDependencyError',
    'EnvironmentError',
    'ValidationError',
    'InvalidParameterError',
    'MissingParameterError',
    'InvalidOperationError',
    'SchemaValidationError',
    'ExecutionError',
    'AgentExecutionError',
    'TimeoutError',
    'RetryExhaustedError',
    'ResourceError',
    'FileOperationError',
    'NetworkError',
    'CacheError',
    'WebSearchError',
    'ScrapingError',
    'PDFOperationError',
    'ExcelOperationError',
    'OCRError',
    'CodeExecutionError',
    'DataExtractionError',
    'LLMError',
    'wrap_exception',
]
