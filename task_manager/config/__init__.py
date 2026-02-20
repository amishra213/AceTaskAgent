"""
Configuration module - Settings and configuration management
"""

from .agent_config import AgentConfig, LLMConfig, LLMProvider, FolderConfig, RateLimitConfig
from .config_properties import ConfigProperties

# EnvConfig is an alias for ConfigProperties — kept for backward compatibility
# with all existing imports: `from task_manager.config import EnvConfig`
EnvConfig = ConfigProperties

__all__ = [
    'AgentConfig',
    'LLMConfig',
    'LLMProvider',
    'FolderConfig',
    'RateLimitConfig',
    'EnvConfig',
    'ConfigProperties',
]
