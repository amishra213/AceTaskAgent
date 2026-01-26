"""
Agent configuration - Settings for the Task Manager Agent
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from pathlib import Path
import os
from enum import Enum

if TYPE_CHECKING:
    from pathlib import Path as PathType


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """
    Configuration for LLM provider and model.
    
    Attributes:
        provider: LLM provider (anthropic, openai, google, groq, deepseek, local)
        model_name: Model identifier for the provider
        api_key: API key for the provider (reads from LLM_API_KEY env if not provided)
        base_url: Base URL for API (useful for local/custom servers)
        temperature: Temperature for response generation (0-2)
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
        use_native_sdk: Use native provider SDK instead of LangChain wrapper
        api_version: API version to use (provider-specific, e.g., v1alpha, v1beta)
        api_base_url: Base URL for LLM API endpoint (reads from LLM_API_BASE_URL env)
        api_endpoint_path: Path for API endpoint (reads from LLM_API_ENDPOINT_PATH env)
        extra_params: Additional provider-specific parameters
    """
    
    provider: str = "anthropic"
    model_name: str = "claude-sonnet-4-20250514"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    timeout: int = 30
    use_native_sdk: bool = False
    api_version: str = "v1alpha"
    api_base_url: Optional[str] = None
    api_endpoint_path: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and set up LLM configuration."""
        # Validate provider
        valid_providers = [p.value for p in LLMProvider]
        if self.provider not in valid_providers:
            raise ValueError(f"Provider must be one of {valid_providers}, got {self.provider}")
        
        # Validate temperature
        if not 0 <= self.temperature <= 2:
            raise ValueError(f"Temperature must be between 0 and 2, got {self.temperature}")
        
        # Get API key from environment if not provided
        if not self.api_key:
            # Try generic LLM_API_KEY first, then provider-specific
            self.api_key = os.getenv('LLM_API_KEY')
            if not self.api_key:
                env_var = self._get_env_var_for_provider()
                self.api_key = os.getenv(env_var)
            
            if not self.api_key and self.provider != "local":
                env_var = self._get_env_var_for_provider()
                raise ValueError(
                    f"API key not provided and LLM_API_KEY or {env_var} environment variable not set. "
                    f"Set it via config or environment: export LLM_API_KEY=your-key"
                )
    
    def _get_env_var_for_provider(self) -> str:
        """Get environment variable name for provider (fallback only)."""
        env_vars = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "local": "LOCAL_LLM_URL"
        }
        return env_vars.get(self.provider, f"{self.provider.upper()}_API_KEY")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding API key for security."""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "base_url": self.base_url,
        }


@dataclass
class RateLimitConfig:
    """
    Configuration for LLM rate limiting.
    
    Attributes:
        requests_per_minute: Maximum requests per minute (0 = unlimited)
        requests_per_second: Maximum requests per second (0 = unlimited, overrides RPM)
        min_request_delay: Minimum delay between requests in seconds (0 = no delay)
    """
    requests_per_minute: int = 60
    requests_per_second: int = 0
    min_request_delay: float = 0.5
    
    def __post_init__(self):
        """Validate rate limit configuration."""
        if self.requests_per_minute < 0:
            raise ValueError("requests_per_minute cannot be negative")
        if self.requests_per_second < 0:
            raise ValueError("requests_per_second cannot be negative")
        if self.min_request_delay < 0:
            raise ValueError("min_request_delay cannot be negative")
    
    @classmethod
    def from_env(cls) -> "RateLimitConfig":
        """Create rate limit config from environment variables."""
        return cls(
            requests_per_minute=int(os.getenv('LLM_RATE_LIMIT_RPM', '60')),
            requests_per_second=int(os.getenv('LLM_RATE_LIMIT_RPS', '0')),
            min_request_delay=float(os.getenv('LLM_MIN_REQUEST_DELAY', '0.5')),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "requests_per_minute": self.requests_per_minute,
            "requests_per_second": self.requests_per_second,
            "min_request_delay": self.min_request_delay,
        }


@dataclass
class FolderConfig:
    """
    Configuration for input, output, and temp folders.
    
    Attributes:
        input_folder: Path to input folder for user-provided data files
        output_folder: Path to output folder for final results
        temp_folder: Path to temp folder for work-in-progress data
        auto_create: Whether to auto-create folders if they don't exist
    """
    input_folder: str = "./input"
    output_folder: str = "./output"
    temp_folder: str = "./temp"
    auto_create: bool = True
    
    def __post_init__(self):
        """Initialize and optionally create folders."""
        # Convert to absolute paths
        self._input_path = Path(self.input_folder).resolve()
        self._output_path = Path(self.output_folder).resolve()
        self._temp_path = Path(self.temp_folder).resolve()
        
        # Auto-create folders if enabled
        if self.auto_create:
            self._input_path.mkdir(parents=True, exist_ok=True)
            self._output_path.mkdir(parents=True, exist_ok=True)
            self._temp_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def input_path(self) -> Path:
        """Get absolute input folder path."""
        return Path(self.input_folder).resolve()
    
    @property
    def output_path(self) -> Path:
        """Get absolute output folder path."""
        return Path(self.output_folder).resolve()
    
    @property
    def temp_path(self) -> Path:
        """Get absolute temp folder path."""
        return Path(self.temp_folder).resolve()
    
    def get_input_files(self, pattern: str = "*") -> List[Path]:
        """
        Get list of files in input folder matching pattern.
        
        Args:
            pattern: Glob pattern to match files (default: "*" for all files)
        
        Returns:
            List of Path objects for matching files
        """
        return list(self.input_path.glob(pattern))
    
    def get_output_file(self, filename: str) -> Path:
        """
        Get path for an output file.
        
        Args:
            filename: Name of the output file
        
        Returns:
            Full path to the output file
        """
        return self.output_path / filename
    
    def get_temp_file(self, filename: str) -> Path:
        """
        Get path for a temp file.
        
        Args:
            filename: Name of the temp file
        
        Returns:
            Full path to the temp file
        """
        return self.temp_path / filename
    
    @classmethod
    def from_env(cls, prefix: str = "AGENT_") -> "FolderConfig":
        """Create folder config from environment variables."""
        return cls(
            input_folder=os.getenv(f"{prefix}INPUT_FOLDER", "./input"),
            output_folder=os.getenv(f"{prefix}OUTPUT_FOLDER", "./output"),
            temp_folder=os.getenv(f"{prefix}TEMP_FOLDER", "./temp"),
            auto_create=os.getenv(f"{prefix}AUTO_CREATE_FOLDERS", "true").lower() == "true",
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_folder": str(self.input_path),
            "output_folder": str(self.output_path),
            "temp_folder": str(self.temp_path),
            "auto_create": self.auto_create,
        }


@dataclass
class AgentConfig:
    """
    Configuration settings for the Task Manager Agent.
    
    Attributes:
        llm: LLM configuration (default: Anthropic Claude Sonnet)
        rate_limit: Rate limiting configuration for LLM calls
        folders: Folder configuration for input/output/temp directories
        max_iterations: Maximum iterations before stopping (default: 100)
        enable_search: Whether to enable web search capability (default: True)
        log_level: Logging level (default: 'INFO')
        timeout: Request timeout in seconds (default: 30)
        max_retries: Maximum retries for failed operations (default: 3)
        debug: Enable debug mode with detailed logging (default: False)
    """
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    folders: FolderConfig = field(default_factory=FolderConfig)
    max_iterations: int = 100
    enable_search: bool = True
    log_level: str = "INFO"
    timeout: int = 30
    max_retries: int = 3
    debug: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        
        if self.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            raise ValueError(f"Invalid log_level: {self.log_level}")
        
        if self.timeout < 1:
            raise ValueError("timeout must be at least 1 second")
        
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        
        # Ensure LLM is a LLMConfig instance
        if isinstance(self.llm, dict):
            self.llm = LLMConfig(**self.llm)
        
        # Ensure rate_limit is a RateLimitConfig instance
        if isinstance(self.rate_limit, dict):
            self.rate_limit = RateLimitConfig(**self.rate_limit)
        
        # Ensure folders is a FolderConfig instance
        if isinstance(self.folders, dict):
            self.folders = FolderConfig(**self.folders)
        if isinstance(self.rate_limit, dict):
            self.rate_limit = RateLimitConfig(**self.rate_limit)
    
    @classmethod
    def from_env(cls, prefix: str = "AGENT_") -> "AgentConfig":
        """
        Create configuration from environment variables.
        
        Args:
            prefix: Prefix for environment variables (default: "AGENT_")
        
        Returns:
            Configured AgentConfig instance
            
        Example:
            export AGENT_LOG_LEVEL=DEBUG
            export AGENT_MAX_ITERATIONS=50
            export AGENT_INPUT_FOLDER=./data/input
            export AGENT_OUTPUT_FOLDER=./data/output
            export ANTHROPIC_API_KEY=sk-...
            config = AgentConfig.from_env()
        """
        return cls(
            llm=LLMConfig(
                provider=os.getenv(f"{prefix}LLM_PROVIDER", "anthropic"),
                model_name=os.getenv(f"{prefix}LLM_MODEL", "claude-sonnet-4-20250514"),
                api_key=os.getenv("LLM_API_KEY"),  # Read from LLM_API_KEY env var
                base_url=os.getenv("LLM_API_BASE_URL"),  # Read from LLM_API_BASE_URL env var
                api_base_url=os.getenv("LLM_API_BASE_URL"),  # Also set api_base_url for compatibility
                temperature=float(os.getenv(f"{prefix}LLM_TEMPERATURE", "0.2")),
                max_tokens=int(os.getenv(f"{prefix}LLM_MAX_TOKENS")) if os.getenv(f"{prefix}LLM_MAX_TOKENS") else None,
                timeout=int(os.getenv(f"{prefix}TIMEOUT", "30")),
            ),
            rate_limit=RateLimitConfig.from_env(),
            folders=FolderConfig.from_env(prefix),
            max_iterations=int(os.getenv(f"{prefix}MAX_ITERATIONS", "100")),
            enable_search=os.getenv(f"{prefix}ENABLE_SEARCH", "true").lower() == "true",
            log_level=os.getenv(f"{prefix}LOG_LEVEL", "INFO"),
            timeout=int(os.getenv(f"{prefix}TIMEOUT", "30")),
            max_retries=int(os.getenv(f"{prefix}MAX_RETRIES", "3")),
            debug=os.getenv(f"{prefix}DEBUG", "false").lower() == "true",
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration
        
        Returns:
            Configured AgentConfig instance
            
        Example:
            config_dict = {
                "llm": {
                    "provider": "openai",
                    "model_name": "gpt-4",
                    "api_key": "sk-..."
                },
                "max_iterations": 50,
                "log_level": "DEBUG"
            }
            config = AgentConfig.from_dict(config_dict)
        """
        llm_config = config_dict.pop("llm", {})
        if isinstance(llm_config, dict):
            llm_config = LLMConfig(**llm_config)
        
        folders_config = config_dict.pop("folders", {})
        if isinstance(folders_config, dict):
            folders_config = FolderConfig(**folders_config)
        
        return cls(llm=llm_config, folders=folders_config, **config_dict)
    
    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Args:
            include_secrets: Whether to include API keys (default: False)
        
        Returns:
            Dictionary representation of config
        """
        result = {
            "llm": self.llm.to_dict(),
            "rate_limit": self.rate_limit.to_dict(),
            "folders": self.folders.to_dict(),
            "max_iterations": self.max_iterations,
            "enable_search": self.enable_search,
            "log_level": self.log_level,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "debug": self.debug,
        }
        
        if include_secrets and self.llm.api_key:
            result["llm"]["api_key"] = self.llm.api_key
        
        return result
