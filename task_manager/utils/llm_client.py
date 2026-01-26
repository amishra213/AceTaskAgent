"""
Generic LLM Client Wrapper

This module provides a unified wrapper for any LLM provider (Google Gemini, OpenAI, Anthropic, etc.)
configured entirely from environment variables and configuration files.

Supports:
- Native provider SDKs (when available and enabled)
- LangChain wrappers (universal fallback)
- Flexible configuration from .env and code
- Multi-provider support with zero code changes
- Global rate limiting for API calls

Example usage:
    from task_manager.utils.llm_client import LLMClient
    
    # Uses environment variables: LLM_API_KEY, AGENT_LLM_PROVIDER, etc.
    client = LLMClient()
    response = client.generate_content("Explain AI")
    print(response)
"""

import os
import logging
from typing import Optional, Union, List, Dict, Any
from dotenv import load_dotenv

from .rate_limiter import global_rate_limiter
from .exceptions import (
    ConfigurationError,
    MissingDependencyError,
    LLMError,
    InvalidParameterError
)

logger = logging.getLogger(__name__)


class ResponseWrapper:
    """Wrapper for responses to provide LangChain-compatible interface."""
    
    def __init__(self, text: str):
        self.content = text
    
    def __str__(self):
        return self.content


class LLMClient:
    """
    Generic LLM Client - works with any provider.
    
    Configuration is read from environment variables:
    - LLM_API_KEY: API key for the provider
    - AGENT_LLM_PROVIDER: Provider name (google, openai, anthropic, ollama)
    - AGENT_LLM_MODEL: Model name (e.g., gemini-2.5-flash, gpt-4-turbo, claude-3-sonnet)
    - LLM_API_BASE_URL: Custom API base URL (optional)
    - LLM_API_ENDPOINT_PATH: Custom endpoint path (optional)
    - LLM_API_VERSION: API version (optional)
    - USE_NATIVE_SDK: Use native SDK if available (true/false)
    """
    
    PROVIDER_DEFAULTS = {
        'google': {
            'model': 'gemini-2.5-flash',
            'api_base_url': 'https://generativelanguage.googleapis.com',
            'api_endpoint_path': 'v1beta',
            'api_version': 'v1alpha',
            'native_sdk_package': 'google-generativeai',
            'langchain_package': 'langchain-google-genai',
        },
        'openai': {
            'model': 'gpt-4-turbo',
            'api_base_url': 'https://api.openai.com',
            'api_endpoint_path': 'v1',
            'api_version': '2024-10-01',
            'native_sdk_package': None,  # OpenAI doesn't have distinct "native" SDK
            'langchain_package': 'langchain-openai',
        },
        'anthropic': {
            'model': 'claude-3-5-sonnet-20241022',
            'api_base_url': 'https://api.anthropic.com',
            'api_endpoint_path': 'v1',
            'api_version': '2024-06-01',
            'native_sdk_package': 'anthropic',
            'langchain_package': 'langchain-anthropic',
        },
        'groq': {
            'model': 'llama-3.3-70b-versatile',
            'api_base_url': 'https://api.groq.com/openai',
            'api_endpoint_path': 'v1',
            'api_version': None,
            'native_sdk_package': None,  # Uses OpenAI-compatible API
            'langchain_package': 'langchain-groq',
        },
        'ollama': {
            'model': 'llama2',
            'api_base_url': 'http://localhost:11434',
            'api_endpoint_path': 'api',
            'api_version': None,
            'native_sdk_package': None,
            'langchain_package': 'langchain-community',
        },
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        use_native_sdk: bool = False,
        api_base_url: Optional[str] = None,
        api_endpoint_path: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        """
        Initialize LLM Client.
        
        Configuration sources (in precedence order):
        1. Direct parameters (if provided)
        2. Environment variables (LLM_*, AGENT_*)
        3. Provider defaults
        
        Args:
            api_key: API key for the provider (or from LLM_API_KEY env)
            provider: Provider name: google, openai, anthropic, ollama (or from AGENT_LLM_PROVIDER env)
            model: Model name (or from AGENT_LLM_MODEL env)
            temperature: Temperature for generation (0.0-2.0)
            use_native_sdk: Use native provider SDK if available
            api_base_url: Custom API base URL
            api_endpoint_path: Custom API endpoint path
            api_version: Custom API version
        
        Raises:
            ValueError: If required configuration is missing
            ImportError: If required packages not installed
        """
        load_dotenv()
        
        # Read configuration from environment
        self.api_key = api_key or os.getenv('LLM_API_KEY')
        self.provider = (provider or os.getenv('AGENT_LLM_PROVIDER', 'google')).lower()
        self.temperature = temperature
        self.use_native_sdk = use_native_sdk or os.getenv('USE_NATIVE_SDK', 'false').lower() == 'true'
        
        # Validate provider
        if self.provider not in self.PROVIDER_DEFAULTS:
            raise InvalidParameterError(
                parameter_name="provider",
                message=f"Unsupported provider: {self.provider}. "
                f"Supported: {list(self.PROVIDER_DEFAULTS.keys())}"
            )
        
        # Get provider defaults
        defaults = self.PROVIDER_DEFAULTS[self.provider]
        
        # Set model
        self.model = model or os.getenv('AGENT_LLM_MODEL', defaults['model'])
        
        # Set API configuration
        self.api_base_url = api_base_url or os.getenv('LLM_API_BASE_URL', defaults['api_base_url'])
        self.api_endpoint_path = api_endpoint_path or os.getenv('LLM_API_ENDPOINT_PATH', defaults['api_endpoint_path'])
        self.api_version = api_version or os.getenv('LLM_API_VERSION', defaults.get('api_version'))
        
        # Construct full endpoint
        if self.api_endpoint_path:
            self.api_endpoint = f"{self.api_base_url}/{self.api_endpoint_path}"
        else:
            self.api_endpoint = self.api_base_url
        
        # Validate API key
        if not self.api_key:
            raise ConfigurationError(
                setting_name="LLM_API_KEY",
                message="API key not found. Set LLM_API_KEY environment variable or pass api_key parameter."
            )
        
        logger.info(f"Initializing LLM Client")
        logger.debug(f"  Provider: {self.provider}")
        logger.debug(f"  Model: {self.model}")
        logger.debug(f"  API Base URL: {self.api_base_url}")
        logger.debug(f"  API Endpoint: {self.api_endpoint_path}")
        logger.debug(f"  Use Native SDK: {self.use_native_sdk}")
        
        self.client = None
        self._initialize_client()
    
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client based on provider and configuration."""
        # Try native SDK first if enabled
        if self.use_native_sdk and self.provider == 'google':
            try:
                self._initialize_google_native()
                return
            except ImportError as e:
                logger.warning(f"Native Google SDK not available: {e}. Falling back to LangChain.")
        
        # Fall back to LangChain wrapper
        self._initialize_langchain_wrapper()
    
    
    def _initialize_google_native(self):
        """Initialize native Google GenAI SDK."""
        try:
            from google import genai
            
            self.client = genai.Client(
                api_key=self.api_key,
                http_options={'api_version': self.api_version or 'v1alpha'}
            )
            
            logger.info(f"✓ Initialized native Google GenAI SDK (API: {self.api_version})")
            logger.info(f"  Model: {self.model}")
            
        except ImportError:
            raise MissingDependencyError(
                package_name="google-generativeai",
                install_command="pip install google-generativeai",
                purpose="Google GenAI SDK"
            )
    
    
    def _initialize_langchain_wrapper(self):
        """Initialize LangChain wrapper for the provider."""
        logger.debug(f"Initializing LangChain wrapper for {self.provider}")
        
        if self.provider == 'google':
            self._init_langchain_google()
        elif self.provider == 'openai':
            self._init_langchain_openai()
        elif self.provider == 'anthropic':
            self._init_langchain_anthropic()
        elif self.provider == 'groq':
            self._init_langchain_groq()
        elif self.provider == 'ollama':
            self._init_langchain_ollama()
        else:
            raise InvalidParameterError(
                parameter_name="provider",
                message=f"Unsupported provider: {self.provider}"
            )
    
    
    def _init_langchain_google(self):
        """Initialize LangChain Google wrapper."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            self.client = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.api_key,
                temperature=self.temperature
            )
            
            logger.info(f"✓ Initialized LangChain ChatGoogleGenerativeAI")
            logger.info(f"  Model: {self.model}")
            
        except ImportError:
            raise MissingDependencyError(
                package_name="langchain-google-genai",
                install_command="pip install langchain-google-genai",
                purpose="LangChain Google wrapper"
            )
    
    
    def _init_langchain_openai(self):
        """Initialize LangChain OpenAI wrapper."""
        try:
            from langchain_openai import ChatOpenAI
            
            self.client = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,  # type: ignore
                temperature=self.temperature
            )
            
            logger.info(f"✓ Initialized LangChain ChatOpenAI")
            logger.info(f"  Model: {self.model}")
            
        except ImportError:
            raise MissingDependencyError(
                package_name="langchain-openai",
                install_command="pip install langchain-openai",
                purpose="LangChain OpenAI wrapper"
            )
    
    
    def _init_langchain_anthropic(self):
        """Initialize LangChain Anthropic wrapper."""
        try:
            from langchain_anthropic import ChatAnthropic
            
            kwargs = {
                'model_name': self.model,
                'temperature': self.temperature
            }
            
            # Only set api_key if provided (Anthropic defaults to env var)
            if self.api_key and isinstance(self.api_key, str):
                kwargs['api_key'] = self.api_key
            
            self.client = ChatAnthropic(**kwargs)
            
            logger.info(f"✓ Initialized LangChain ChatAnthropic")
            logger.info(f"  Model: {self.model}")
            
        except ImportError:
            raise MissingDependencyError(
                package_name="langchain-anthropic",
                install_command="pip install langchain-anthropic",
                purpose="LangChain Anthropic wrapper"
            )
    
    
    def _init_langchain_groq(self):
        """Initialize LangChain Groq wrapper (OpenAI-compatible API)."""
        try:
            from langchain_groq import ChatGroq
            
            self.client = ChatGroq(
                model=self.model,
                api_key=self.api_key,  # type: ignore
                temperature=self.temperature
            )
            
            logger.info(f"✓ Initialized LangChain ChatGroq")
            logger.info(f"  Model: {self.model}")
            
        except ImportError:
            # Fall back to OpenAI client with custom base URL (Groq is OpenAI-compatible)
            logger.warning("langchain-groq not installed, falling back to OpenAI-compatible client")
            try:
                from langchain_openai import ChatOpenAI
                
                self.client = ChatOpenAI(
                    model=self.model,
                    api_key=self.api_key,  # type: ignore
                    base_url="https://api.groq.com/openai/v1",
                    temperature=self.temperature
                )
                
                logger.info(f"✓ Initialized Groq via OpenAI-compatible client")
                logger.info(f"  Model: {self.model}")
                logger.info(f"  Base URL: https://api.groq.com/openai/v1")
                
            except ImportError:
                raise MissingDependencyError(
                    package_name="langchain-groq or langchain-openai",
                    install_command="pip install langchain-groq OR pip install langchain-openai",
                    purpose="Groq API client"
                )
    
    
    def _init_langchain_ollama(self):
        """Initialize LangChain Ollama wrapper."""
        try:
            from langchain_community.llms import Ollama
            
            self.client = Ollama(
                model=self.model,
                base_url=self.api_base_url,
                temperature=self.temperature
            )
            
            logger.info(f"✓ Initialized LangChain Ollama")
            logger.info(f"  Model: {self.model}")
            logger.info(f"  Base URL: {self.api_base_url}")
            
        except ImportError:
            raise MissingDependencyError(
                package_name="langchain-community",
                install_command="pip install langchain-community",
                purpose="Ollama LLM wrapper"
            )
    
    
    def invoke(self, messages: List[Any], **kwargs) -> ResponseWrapper:
        """
        LangChain-compatible invoke method.
        
        Args:
            messages: List of LangChain messages
            **kwargs: Additional parameters
        
        Returns:
            ResponseWrapper with .content attribute
        """
        try:
            # Apply global rate limiting before making the request
            wait_time = global_rate_limiter.wait()
            if wait_time > 0:
                logger.debug(f"Rate limiter delayed request by {wait_time:.2f}s")
            
            if self.provider == 'google' and hasattr(self.client, 'models'):
                # Native SDK
                return self._invoke_native(messages, **kwargs)
            else:
                # LangChain wrapper
                return self._invoke_langchain(messages, **kwargs)
        
        except Exception as e:
            logger.error(f"Error in invoke: {str(e)}")
            raise
    
    
    def _invoke_native(self, messages: List[Any], **kwargs) -> ResponseWrapper:
        """Invoke using native SDK."""
        try:
            # Build request contents from messages
            request_contents = []
            
            for msg in messages:
                msg_type = msg.__class__.__name__
                msg_content = msg.content if hasattr(msg, 'content') else str(msg)
                
                if msg_type == 'SystemMessage':
                    request_contents.append({
                        "role": "user",
                        "parts": [{"text": msg_content}]
                    })
                elif msg_type in ['HumanMessage', 'UserMessage']:
                    request_contents.append({
                        "role": "user",
                        "parts": [{"text": msg_content}]
                    })
            
            response = self.client.models.generate_content(  # type: ignore
                model=self.model,
                contents=request_contents
            )
            
            return ResponseWrapper(response.text or "")
        
        except Exception as e:
            logger.error(f"Error in native SDK invoke: {str(e)}")
            raise
    
    
    def _invoke_langchain(self, messages: List[Any], **kwargs) -> ResponseWrapper:
        """Invoke using LangChain wrapper."""
        try:
            response = self.client.invoke(messages, **kwargs)  # type: ignore
            content = response.content if hasattr(response, 'content') else str(response)
            # Ensure content is a string
            if isinstance(content, str):
                return ResponseWrapper(content)
            else:
                return ResponseWrapper(str(content))
        
        except Exception as e:
            logger.error(f"Error in LangChain invoke: {str(e)}")
            raise
    
    
    def generate_content(
        self,
        contents: Union[str, List[str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate content using the LLM.
        
        Args:
            contents: Prompt or list of prompts
            system_prompt: System message for context
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text response
        """
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            
            if isinstance(contents, list):
                full_content = ' '.join(str(c) for c in contents)
            else:
                full_content = contents
            
            messages.append(HumanMessage(content=full_content))
            
            response = self.invoke(messages, **kwargs)
            return response.content
        
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            raise
    
    
    @staticmethod
    def get_provider_defaults(provider: str) -> Dict[str, Any]:
        """Get default configuration for a provider."""
        if provider not in LLMClient.PROVIDER_DEFAULTS:
            raise ValueError(f"Unknown provider: {provider}")
        return LLMClient.PROVIDER_DEFAULTS[provider]
    
    
    @staticmethod
    def list_supported_providers() -> List[str]:
        """List all supported providers."""
        return list(LLMClient.PROVIDER_DEFAULTS.keys())
