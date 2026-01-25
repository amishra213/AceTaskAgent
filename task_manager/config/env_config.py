"""
Environment configuration - Load settings from .env files
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import json

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


class EnvConfig:
    """
    Load and manage configuration from environment variables and .env files.
    
    Supports multiple sources with priority:
    1. Environment variables (highest priority)
    2. .env file in current/specified directory
    3. .env.example as fallback
    """
    
    @staticmethod
    def load_env_file(path: Optional[str] = None) -> bool:
        """
        Load environment variables from .env file.
        
        Args:
            path: Path to .env file (default: search current dir and parents)
        
        Returns:
            True if file was loaded, False otherwise
        """
        if not HAS_DOTENV:
            print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
            return False
        
        # Search for .env file
        if path:
            env_path = Path(path)
        else:
            # Search in current dir and up to 3 levels up
            env_path = None
            current = Path.cwd()
            for _ in range(4):  # Current dir + 3 parent levels
                potential_path = current / ".env"
                if potential_path.exists():
                    env_path = potential_path
                    break
                if current.parent == current:  # Stop at filesystem root
                    break
                current = current.parent
        
        if env_path and env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded environment from {env_path}")
            return True
        
        return False
    
    @staticmethod
    def get(key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable with optional default."""
        return os.getenv(key, default)
    
    @staticmethod
    def get_bool(key: str, default: bool = False) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    @staticmethod
    def get_int(key: str, default: int = 0) -> int:
        """Get integer environment variable."""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    @staticmethod
    def get_float(key: str, default: float = 0.0) -> float:
        """Get float environment variable."""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            return default
    
    @staticmethod
    def get_json(key: str, default: Optional[Dict] = None) -> Optional[Dict]:
        """Get JSON environment variable."""
        value = os.getenv(key)
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    
    @staticmethod
    def check_required(*keys: str) -> bool:
        """
        Check if required environment variables are set.
        
        Args:
            *keys: Environment variable names to check
        
        Returns:
            True if all are set, False otherwise
        """
        missing = [key for key in keys if not os.getenv(key)]
        
        if missing:
            print(f"❌ Missing required environment variables: {', '.join(missing)}")
            return False
        
        return True
    
    @staticmethod
    def show_config_template(llm_provider: str = "anthropic") -> str:
        """
        Show .env template for configuration.
        
        Args:
            llm_provider: LLM provider to show config for
        
        Returns:
            Template as string
        """
        templates = {
            "anthropic": """
# Anthropic Configuration
ANTHROPIC_API_KEY=sk-ant-...
AGENT_LLM_PROVIDER=anthropic
AGENT_LLM_MODEL=claude-sonnet-4-20250514
AGENT_LOG_LEVEL=INFO
AGENT_MAX_ITERATIONS=100
AGENT_ENABLE_SEARCH=true
AGENT_TIMEOUT=30
AGENT_MAX_RETRIES=3
""",
            "openai": """
# OpenAI Configuration
OPENAI_API_KEY=sk-...
AGENT_LLM_PROVIDER=openai
AGENT_LLM_MODEL=gpt-4-turbo
AGENT_LOG_LEVEL=INFO
AGENT_MAX_ITERATIONS=100
AGENT_ENABLE_SEARCH=true
AGENT_TIMEOUT=30
AGENT_MAX_RETRIES=3
""",
            "google": """
# Google Generative AI Configuration
GOOGLE_API_KEY=AIza...
AGENT_LLM_PROVIDER=google
AGENT_LLM_MODEL=gemini-pro
AGENT_LOG_LEVEL=INFO
AGENT_MAX_ITERATIONS=100
AGENT_ENABLE_SEARCH=true
AGENT_TIMEOUT=30
AGENT_MAX_RETRIES=3

# Folder Configuration
AGENT_INPUT_FOLDER=./input
AGENT_OUTPUT_FOLDER=./output
AGENT_TEMP_FOLDER=./temp
AGENT_AUTO_CREATE_FOLDERS=true
""",
            "groq": """
# Groq Configuration (Fast inference with Llama models)
LLM_API_KEY=gsk_...
AGENT_LLM_PROVIDER=groq
AGENT_LLM_MODEL=llama-3.3-70b-versatile
AGENT_LOG_LEVEL=INFO
AGENT_MAX_ITERATIONS=100
AGENT_ENABLE_SEARCH=true
AGENT_TIMEOUT=30
AGENT_MAX_RETRIES=3

# Folder Configuration
AGENT_INPUT_FOLDER=./input
AGENT_OUTPUT_FOLDER=./output
AGENT_TEMP_FOLDER=./temp
AGENT_AUTO_CREATE_FOLDERS=true
""",
            "local": """
# Local LLM Configuration (Ollama)
AGENT_LLM_PROVIDER=local
AGENT_LLM_MODEL=llama2
AGENT_LLM_BASE_URL=http://localhost:11434
AGENT_LOG_LEVEL=INFO
AGENT_MAX_ITERATIONS=100
AGENT_ENABLE_SEARCH=false
AGENT_TIMEOUT=30
AGENT_MAX_RETRIES=3

# Folder Configuration
AGENT_INPUT_FOLDER=./input
AGENT_OUTPUT_FOLDER=./output
AGENT_TEMP_FOLDER=./temp
AGENT_AUTO_CREATE_FOLDERS=true
""",
        }
        
        return templates.get(llm_provider, templates["anthropic"])
