"""
Logger module - Logging configuration and utilities

This module provides logging configuration with support for:
- File and console logging
- Configuration from .env
- Langfuse integration
- Structured logging with context
- Unicode support on Windows
"""

import logging
import sys
import os
import codecs
from typing import Optional, Union, Any

# Fix Unicode support on Windows BEFORE any logging is configured
# This must be done at module import time
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

# Flag to track if ComprehensiveLogger has been initialized
_comprehensive_logger_initialized = False


def _ensure_comprehensive_logger_initialized():
    """
    Initialize ComprehensiveLogger with .env configuration on first use.
    This is called automatically by get_logger().
    """
    global _comprehensive_logger_initialized
    
    if _comprehensive_logger_initialized:
        return
    
    _comprehensive_logger_initialized = True
    
    try:
        from .comprehensive_logger import ComprehensiveLogger
        from task_manager.config import EnvConfig
        
        # Load environment configuration
        EnvConfig.load_env_file()
        
        # Get logging settings from environment
        log_folder = os.getenv("AGENT_LOG_FOLDER", "./logs")
        log_level = os.getenv("AGENT_LOG_LEVEL", "INFO")
        enable_console = os.getenv("AGENT_ENABLE_CONSOLE_LOGGING", "true").lower() == "true"
        enable_file = os.getenv("AGENT_ENABLE_FILE_LOGGING", "true").lower() == "true"
        enable_langfuse = os.getenv("ENABLE_LANGFUSE", "false").lower() == "true"
        max_bytes = int(os.getenv("AGENT_LOG_MAX_BYTES", "10485760"))  # 10MB default
        backup_count = int(os.getenv("AGENT_LOG_BACKUP_COUNT", "5"))
        
        # Langfuse configuration
        langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        langfuse_host = os.getenv("LANGFUSE_BASE_URL")
        
        # Initialize ComprehensiveLogger
        ComprehensiveLogger.initialize(
            log_folder=log_folder,
            log_level=log_level,
            enable_console=enable_console,
            enable_file=enable_file,
            enable_langfuse=enable_langfuse,
            langfuse_public_key=langfuse_public_key,
            langfuse_secret_key=langfuse_secret_key,
            langfuse_host=langfuse_host,
            max_bytes=max_bytes,
            backup_count=backup_count
        )
        
    except Exception as e:
        # Fallback to basic logging if ComprehensiveLogger initialization fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.getLogger(__name__).warning(
            f"Failed to initialize ComprehensiveLogger: {e}. Using basic logging."
        )


def get_logger(name: str, level: Optional[str] = None) -> Any:
    """
    Get or create a logger with standard formatting and .env configuration.
    
    This function initializes the ComprehensiveLogger system on first call,
    which loads configuration from .env and enables file and console logging.
    
    Args:
        name: Logger name (typically __name__)
        level: Optional logging level override (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance with file and console handlers (TaskLogger or basic Logger)
    """
    # Initialize comprehensive logger on first call
    _ensure_comprehensive_logger_initialized()
    
    # Try to use ComprehensiveLogger
    try:
        from .comprehensive_logger import ComprehensiveLogger as CL
        logger = CL.get_logger(name)
        return logger
    except Exception:
        # Fallback to basic logger if ComprehensiveLogger fails
        logger = logging.getLogger(name)
        
        # Only configure if not already configured
        if not logger.handlers:
            level = level or os.getenv("AGENT_LOG_LEVEL", "INFO")
            logger.setLevel(getattr(logging, level))
            
            # Create console handler
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(getattr(logging, level))
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            
            logger.addHandler(handler)
        
        return logger
