"""
Comprehensive Logging System with File, Console, and Langfuse Integration

Features:
- Configurable log folder (via .env)
- Console and file logging
- Langfuse integration for observability
- Structured logging with context
- Log rotation and archival
- Performance metrics tracking
- Unicode support on Windows
"""

import logging
import logging.handlers
import sys
import os
import codecs
from pathlib import Path
from typing import Optional, Dict, Any, Literal, cast
from datetime import datetime
import json
import traceback

# Fix Unicode support on Windows BEFORE any logging is configured
if sys.platform == 'win32':
    try:
        # Check encoding before wrapping
        if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
            # Wrap stdout with UTF-8 codec writer for Windows
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'backslashreplace')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'backslashreplace')
    except (AttributeError, TypeError):
        # In case buffer attribute is not available
        pass

try:
    from langfuse import Langfuse
    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False


class SafeStreamHandler(logging.StreamHandler):
    """
    Custom StreamHandler that safely handles Unicode on Windows.
    Uses 'replace' error handling to avoid UnicodeEncodeError.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Handle encoding carefully on Windows
            if hasattr(stream, 'buffer'):
                # Use buffer with UTF-8 encoding and replace errors
                stream.buffer.write((msg + self.terminator).encode('utf-8', errors='replace'))
                stream.buffer.flush()
            else:
                # Fallback to regular write with error handling
                try:
                    stream.write(msg + self.terminator)
                except UnicodeEncodeError:
                    # Replace problematic characters with '?'
                    safe_msg = msg.encode(stream.encoding or 'utf-8', errors='replace').decode(stream.encoding or 'utf-8')
                    stream.write(safe_msg + self.terminator)
                self.flush()
        except Exception:
            # Silently ignore encoding errors - don't call handleError which prints to stderr
            pass


class ComprehensiveLogger:
    """
    Centralized logging system with file, console, and Langfuse support.
    
    Usage:
        logger = ComprehensiveLogger.get_logger("my_module")
        logger.info("Message", extra={"user": "john"})
        
        # With Langfuse
        logger.create_trace("operation_name", metadata={"priority": "high"})
    """
    
    _instance = None
    _loggers = {}
    _langfuse_client = None
    _log_folder = None
    _config = {}
    
    def __init__(self):
        """Initialize the logging system."""
        pass
    
    @classmethod
    def initialize(
        cls,
        log_folder: Optional[str] = None,
        log_level: str = "INFO",
        enable_console: bool = True,
        enable_file: bool = True,
        enable_langfuse: bool = False,
        langfuse_public_key: Optional[str] = None,
        langfuse_secret_key: Optional[str] = None,
        langfuse_host: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> None:
        """
        Initialize the comprehensive logging system.
        
        Args:
            log_folder: Folder for log files (default: ./logs)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            enable_console: Enable console logging
            enable_file: Enable file logging
            enable_langfuse: Enable Langfuse integration
            langfuse_public_key: Langfuse public key
            langfuse_secret_key: Langfuse secret key
            langfuse_host: Langfuse host URL
            max_bytes: Max file size before rotation (default: 10MB)
            backup_count: Number of backup files to keep
        """
        cls._log_folder = log_folder or "./logs"
        cls._config = {
            "log_level": log_level,
            "enable_console": enable_console,
            "enable_file": enable_file,
            "enable_langfuse": enable_langfuse,
            "max_bytes": max_bytes,
            "backup_count": backup_count,
        }
        
        # Create log folder if needed
        if enable_file:
            Path(cls._log_folder).mkdir(parents=True, exist_ok=True)
        
        # Initialize Langfuse if enabled
        if enable_langfuse and HAS_LANGFUSE:
            try:
                cls._langfuse_client = Langfuse(
                    public_key=langfuse_public_key,
                    secret_key=langfuse_secret_key,
                    host=langfuse_host
                )
                logging.getLogger("ComprehensiveLogger").info(
                    "✓ Langfuse initialized successfully"
                )
            except Exception as e:
                logging.getLogger("ComprehensiveLogger").warning(
                    f"Failed to initialize Langfuse: {e}"
                )
                cls._langfuse_client = None
        elif enable_langfuse and not HAS_LANGFUSE:
            logging.getLogger("ComprehensiveLogger").warning(
                "Langfuse enabled but not installed. Install with: pip install langfuse"
            )
    
    @classmethod
    def get_logger(cls, name: str) -> "TaskLogger":
        """
        Get or create a logger instance.
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            TaskLogger instance
        """
        if name not in cls._loggers:
            cls._loggers[name] = TaskLogger(name, cls._log_folder, cls._config, cls._langfuse_client)
        
        return cls._loggers[name]
    
    @classmethod
    def create_trace(
        cls,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ):
        """
        Create a trace for observability (via Langfuse).
        
        Args:
            name: Trace name
            metadata: Optional metadata dict
            trace_id: Optional trace ID (not used in start_span)
            
        Returns:
            Span context manager or None if Langfuse not available
        """
        if cls._langfuse_client:
            try:
                return cls._langfuse_client.start_span(
                    name=name,
                    input=metadata
                )
            except Exception as e:
                logging.getLogger("ComprehensiveLogger").error(f"Trace creation failed: {e}")
        
        return None
    
    @classmethod
    def flush(cls):
        """Flush all loggers and Langfuse client."""
        for logger in cls._loggers.values():
            logger.flush()
        
        if cls._langfuse_client:
            cls._langfuse_client.flush()


class TaskLogger:
    """
    Individual logger instance with file, console, and Langfuse support.
    """
    
    def __init__(
        self,
        name: str,
        log_folder: Optional[str],
        config: Dict[str, Any],
        langfuse_client: Optional["Langfuse"] = None
    ):
        """Initialize logger instance."""
        self.name = name
        self.log_folder = log_folder or "./logs"
        self.config = config
        self.langfuse_client = langfuse_client
        self.logger = logging.getLogger(name)
        self.logger.setLevel(config.get("log_level", "INFO"))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        if config.get("enable_console"):
            self._add_console_handler()
        
        # File handler with rotation
        if config.get("enable_file"):
            self._add_file_handler()
    
    def _add_console_handler(self) -> None:
        """Add console handler with Unicode-safe output."""
        # Use custom SafeStreamHandler instead of regular StreamHandler
        handler = SafeStreamHandler(sys.stdout)
        handler.setLevel(self.config.get("log_level", "INFO"))
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def _add_file_handler(self) -> None:
        """Add rotating file handler with UTF-8 encoding."""
        # Ensure log folder exists
        Path(self.log_folder).mkdir(parents=True, exist_ok=True)
        
        log_file = os.path.join(self.log_folder, f"{self.name}.log")
        
        try:
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.get("max_bytes", 10 * 1024 * 1024),
                backupCount=self.config.get("backup_count", 5),
                encoding='utf-8'  # Ensure UTF-8 encoding for Unicode support
            )
            handler.setLevel(self.config.get("log_level", "INFO"))
            
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        except Exception as e:
            self.logger.error(f"Failed to add file handler: {e}")
    
    def debug(self, message: str, extra: Optional[Dict] = None, **kwargs):
        """Log debug message."""
        self._log("DEBUG", message, extra, **kwargs)
    
    def info(self, message: str, extra: Optional[Dict] = None, **kwargs):
        """Log info message."""
        self._log("INFO", message, extra, **kwargs)
    
    def warning(self, message: str, extra: Optional[Dict] = None, **kwargs):
        """Log warning message."""
        self._log("WARNING", message, extra, **kwargs)
    
    def error(self, message: str, extra: Optional[Dict] = None, **kwargs):
        """Log error message."""
        self._log("ERROR", message, extra, **kwargs)
    
    def critical(self, message: str, extra: Optional[Dict] = None, **kwargs):
        """Log critical message."""
        self._log("CRITICAL", message, extra, **kwargs)
    
    def _log(
        self,
        level: str,
        message: str,
        extra: Optional[Dict] = None,
        **kwargs
    ) -> None:
        """
        Internal logging method with Langfuse integration.
        
        Args:
            level: Log level
            message: Log message
            extra: Extra context dict
            **kwargs: Additional arguments for Langfuse
        """
        # Log to standard logger
        log_func = getattr(self.logger, level.lower())
        
        if extra:
            message = f"{message} | {json.dumps(extra)}"
        
        log_func(message)
        
        # Log to Langfuse if available
        if self.langfuse_client:
            try:
                # Map log levels to Langfuse levels
                level_map = {
                    'debug': 'DEBUG',
                    'info': 'DEFAULT',
                    'warning': 'WARNING',
                    'error': 'ERROR',
                    'critical': 'ERROR'
                }
                langfuse_level = cast(Literal['DEBUG', 'DEFAULT', 'WARNING', 'ERROR'], level_map.get(level.lower(), 'DEFAULT'))
                
                self.langfuse_client.create_event(
                    name=f"{self.name}.{level.lower()}",
                    input=extra or {},
                    output=message,
                    level=langfuse_level
                )
            except Exception as e:
                self.logger.debug(f"Langfuse event failed: {e}")
    
    def log_exception(self, message: str, exc: Optional[Exception] = None) -> None:
        """
        Log exception with full traceback.
        
        Args:
            message: Error message
            exc: Exception object (uses current exception if None)
        """
        if exc:
            tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        else:
            tb = traceback.format_exc().split('\n')
        
        full_message = f"{message}\n{''.join(tb)}"
        self.logger.error(full_message)
        
        # Log to Langfuse
        if self.langfuse_client:
            try:
                self.langfuse_client.create_event(
                    name=f"{self.name}.exception",
                    input={"message": message},
                    output=full_message,
                    level='ERROR'
                )
            except Exception as e:
                self.logger.debug(f"Langfuse exception event failed: {e}")
    
    def log_performance(
        self,
        operation: str,
        duration_seconds: float,
        success: bool = True,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Log performance metrics.
        
        Args:
            operation: Operation name
            duration_seconds: Duration in seconds
            success: Whether operation succeeded
            metadata: Additional metadata
        """
        status = "✓" if success else "✗"
        log_message = f"{status} {operation} completed in {duration_seconds:.2f}s"
        
        extra = metadata or {}
        extra.update({
            "operation": operation,
            "duration_seconds": duration_seconds,
            "success": success
        })
        
        log_func = self.info if success else self.warning
        log_func(log_message, extra=extra)
        
        # Log to Langfuse
        if self.langfuse_client:
            try:
                self.langfuse_client.create_event(
                    name=f"{self.name}.performance",
                    input={"operation": operation},
                    output=extra,
                    level='DEFAULT'
                )
            except Exception as e:
                self.logger.debug(f"Langfuse performance event failed: {e}")
    
    def create_trace(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ):
        """
        Create a Langfuse span for request/operation tracking.
        
        Args:
            name: Span name
            metadata: Optional metadata
            trace_id: Optional trace ID (not used in start_span)
            
        Returns:
            Span context manager or None
        """
        if self.langfuse_client:
            try:
                return self.langfuse_client.start_span(
                    name=name,
                    input=metadata
                )
            except Exception as e:
                self.logger.error(f"Trace creation failed: {e}")
        
        return None
    
    def flush(self) -> None:
        """Flush all handlers."""
        for handler in self.logger.handlers:
            handler.flush()


# Convenience function for backward compatibility
def get_logger(name: str, log_folder: Optional[str] = None) -> TaskLogger:
    """
    Get a logger instance (backward compatible function).
    
    Args:
        name: Logger name
        log_folder: Optional log folder path
        
    Returns:
        TaskLogger instance
    """
    if log_folder and not ComprehensiveLogger._log_folder:
        ComprehensiveLogger.initialize(log_folder=log_folder)
    
    return ComprehensiveLogger.get_logger(name)
