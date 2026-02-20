"""
Configuration Properties â€” single source of truth for all app configuration.

Reads config.properties and provides access to every setting. Also exposes
env-var-style helpers (get_env, get_bool_env, â€¦) so this class fully replaces
the former EnvConfig without requiring any call-site changes.
EnvConfig is exported as an alias for this class from task_manager/config/__init__.py.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List


class ConfigProperties:
    """
    Unified configuration loader and accessor.

    Reads config.properties once at startup, injects all plain-key values
    into os.environ, and exposes both file-based and env-var accessors.

    Quick usage::

        # File-based (dot-notation and plain keys from config.properties)
        ConfigProperties.get("deepseek.api.key")
        ConfigProperties.get_int("app.port", 8550)

        # Env-var accessors (reads os.environ, respects OS overrides)
        ConfigProperties.get_env("AGENT_LLM_PROVIDER")
        ConfigProperties.get_logging_config()

        # Bootstrap (call once at process start)
        ConfigProperties.load_env_file()   # load + inject into os.environ
    """

    _instance: Optional["ConfigProperties"] = None
    _properties: Dict[str, str] = {}
    _loaded: bool = False

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Optional[str] = None) -> "ConfigProperties":
        """
        Parse config.properties and return the singleton instance.

        Args:
            path: Explicit path to config.properties; auto-discovered if omitted.
        """
        if cls._instance and cls._loaded:
            return cls._instance

        cls._instance = cls()
        cls._properties = {}

        config_path = Path(path) if path else cls._find_config_file()

        if config_path and config_path.exists():
            cls._parse_file(config_path)

        cls._loaded = True
        return cls._instance

    @classmethod
    def load_env_file(cls, path: Optional[str] = None) -> bool:
        """
        Load config.properties and inject all plain keys into os.environ.

        Drop-in replacement for the former ``EnvConfig.load_env_file()``.

        Args:
            path: Explicit path to config.properties; auto-discovered if omitted.

        Returns:
            True if config.properties was found and loaded.
        """
        try:
            cls.load(path)
            cls.load_to_env()
            props = cls.all_properties()
            if props:
                cfg_path = cls._find_config_file()
                if cfg_path:
                    print(f"Loaded environment from {cfg_path}")
            return bool(props)
        except Exception as exc:
            print(f"Warning: Failed to load config.properties: {exc}")
            return False

    @classmethod
    def load_to_env(cls) -> None:
        """
        Populate os.environ from config.properties (plain keys only).

        - OS/container env vars already set are **never** overwritten.
        - Dot-notation keys (e.g. ``deepseek.api.key``) are skipped â€” they are
          not valid env-var identifiers and must be accessed via :meth:`get`.
        """
        if not cls._loaded:
            cls.load()

        for key, value in cls._properties.items():
            if "." in key:
                continue
            if key not in os.environ:
                os.environ[key] = value

    @classmethod
    def reload(cls, path: Optional[str] = None) -> "ConfigProperties":
        """Force a fresh re-parse of config.properties."""
        cls._loaded = False
        cls._properties = {}
        cls._instance = None
        return cls.load(path)

    # ------------------------------------------------------------------
    # File-based accessors  (read from the parsed properties dict)
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, key: str, default: Optional[str] = None) -> Optional[str]:
        """Return the value for *key* from config.properties (or *default*)."""
        if not cls._loaded:
            cls.load()
        return cls._properties.get(key, default)

    @classmethod
    def get_bool(cls, key: str, default: bool = False) -> bool:
        """Return a boolean value from config.properties."""
        val = cls.get(key)
        if val is None:
            return default
        return val.lower() in ("true", "1", "yes", "on")

    @classmethod
    def get_int(cls, key: str, default: int = 0) -> int:
        """Return an integer value from config.properties."""
        val = cls.get(key)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            return default

    @classmethod
    def get_float(cls, key: str, default: float = 0.0) -> float:
        """Return a float value from config.properties."""
        val = cls.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except ValueError:
            return default

    @classmethod
    def get_section(cls, prefix: str) -> Dict[str, str]:
        """Return all keys/values under dot-notation *prefix* as a flat dict."""
        if not cls._loaded:
            cls.load()
        prefix_dot = prefix if prefix.endswith(".") else prefix + "."
        return {
            key[len(prefix_dot):]: val
            for key, val in cls._properties.items()
            if key.startswith(prefix_dot)
        }

    @classmethod
    def all_properties(cls) -> Dict[str, str]:
        """Return a copy of every loaded property."""
        if not cls._loaded:
            cls.load()
        return dict(cls._properties)

    # ------------------------------------------------------------------
    # Env-var-style accessors  (reads os.environ â€” respects OS overrides)
    # These mirror the former EnvConfig.get* methods exactly.
    # ------------------------------------------------------------------

    @staticmethod
    def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
        """Get an environment variable (same as ``os.getenv``)."""
        return os.getenv(key, default)

    @staticmethod
    def get_bool_env(key: str, default: bool = False) -> bool:
        """Get a boolean from an environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")

    @staticmethod
    def get_int_env(key: str, default: int = 0) -> int:
        """Get an integer from an environment variable."""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default

    @staticmethod
    def get_float_env(key: str, default: float = 0.0) -> float:
        """Get a float from an environment variable."""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            return default

    @staticmethod
    def get_json_env(key: str, default: Optional[Dict] = None) -> Optional[Dict]:
        """Get a JSON-encoded environment variable."""
        value = os.getenv(key)
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    # Kept for backward-compat (EnvConfig.get / get_bool / get_int / get_float
    # read from os.environ in the legacy code). After load_to_env() the values
    # are present in both the properties dict and os.environ, so the file-based
    # versions above return the same results. The aliases below ensure that any
    # code path which calls EnvConfig.get("UPPER_CASE_KEY") still works.

    # ------------------------------------------------------------------
    # Required-key validation
    # ------------------------------------------------------------------

    @staticmethod
    def check_required(*keys: str) -> bool:
        """
        Assert that all *keys* are present in os.environ.

        Prints a diagnostic message and returns False if any are missing.
        """
        missing = [key for key in keys if not os.getenv(key)]
        if missing:
            print(f"âŒ Missing required environment variables: {', '.join(missing)}")
            return False
        return True

    # ------------------------------------------------------------------
    # Logging configuration helper
    # ------------------------------------------------------------------

    @staticmethod
    def get_logging_config() -> Dict[str, Any]:
        """
        Return a ``ComprehensiveLogger.initialize()``-compatible dict
        built from the current environment (populated by ``load_to_env``).
        """
        return {
            "log_folder":          os.getenv("AGENT_LOG_FOLDER", "./logs"),
            "log_level":           os.getenv("AGENT_LOG_LEVEL", "INFO"),
            "enable_console":      os.getenv("AGENT_ENABLE_CONSOLE_LOGGING", "true").lower() in ("true", "1", "yes"),
            "enable_file":         os.getenv("AGENT_ENABLE_FILE_LOGGING", "true").lower() in ("true", "1", "yes"),
            "enable_langfuse":     os.getenv("ENABLE_LANGFUSE", "false").lower() in ("true", "1", "yes"),
            "langfuse_public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
            "langfuse_secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
            "langfuse_host":       os.getenv("LANGFUSE_BASE_URL"),
            "max_bytes":           int(os.getenv("AGENT_LOG_MAX_BYTES", "10485760")),
            "backup_count":        int(os.getenv("AGENT_LOG_BACKUP_COUNT", "5")),
        }

    # ------------------------------------------------------------------
    # Config template helper (for dev onboarding / docs)
    # ------------------------------------------------------------------

    @staticmethod
    def show_config_template(llm_provider: str = "deepseek") -> str:
        """Return a config.properties starter template for *llm_provider*."""
        templates: Dict[str, str] = {
            "deepseek": """\
LLM_API_KEY=sk-...
AGENT_LLM_PROVIDER=deepseek
AGENT_LLM_MODEL=deepseek-chat
LLM_API_BASE_URL=https://api.deepseek.com
LLM_API_ENDPOINT_PATH=v1
deepseek.api.key=sk-...
deepseek.api.base_url=https://api.deepseek.com
deepseek.api.model=deepseek-chat
""",
            "anthropic": """\
LLM_API_KEY=sk-ant-...
AGENT_LLM_PROVIDER=anthropic
AGENT_LLM_MODEL=claude-sonnet-4-20250514
""",
            "openai": """\
LLM_API_KEY=sk-...
AGENT_LLM_PROVIDER=openai
AGENT_LLM_MODEL=gpt-4-turbo
""",
            "google": """\
LLM_API_KEY=AIza...
AGENT_LLM_PROVIDER=google
AGENT_LLM_MODEL=gemini-2.5-flash
""",
            "groq": """\
LLM_API_KEY=gsk_...
AGENT_LLM_PROVIDER=groq
AGENT_LLM_MODEL=llama-3.3-70b-versatile
""",
            "local": """\
AGENT_LLM_PROVIDER=local
AGENT_LLM_MODEL=llama2
AGENT_LLM_BASE_URL=http://localhost:11434
""",
        }
        base = templates.get(llm_provider, templates["deepseek"])
        common = """\
AGENT_LOG_LEVEL=INFO
AGENT_MAX_ITERATIONS=100
AGENT_ENABLE_SEARCH=true
AGENT_TIMEOUT=30
AGENT_MAX_RETRIES=3
AGENT_INPUT_FOLDER=./input_folder
AGENT_OUTPUT_FOLDER=./output_folder
AGENT_TEMP_FOLDER=./temp_folder
AGENT_AUTO_CREATE_FOLDERS=true
"""
        return base + common

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _find_config_file(cls) -> Optional[Path]:
        """Search for config.properties starting from the project root."""
        fixed = Path(__file__).parent.parent.parent / "config.properties"
        if fixed.exists():
            return fixed

        current = Path.cwd()
        for _ in range(4):
            candidate = current / "config.properties"
            if candidate.exists():
                return candidate
            if current.parent == current:
                break
            current = current.parent

        return None

    @classmethod
    def _parse_file(cls, path: Path) -> None:
        """Parse a Java-style .properties file into ``_properties``."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("!"):
                    continue
                for sep in ("=", ":"):
                    if sep in line:
                        key, value = line.split(sep, 1)
                        cls._properties[key.strip()] = value.strip()
                        break


