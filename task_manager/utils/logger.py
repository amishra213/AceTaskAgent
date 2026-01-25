"""
Logger module - Logging configuration and utilities
"""

import logging
import sys
from typing import Optional


# Configure root logger
def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get or create a logger with standard formatting.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        level = level or "INFO"
        logger.setLevel(getattr(logging, level))
        
        # Create console handler with formatting
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
