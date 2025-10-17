"""
LLM Provider Abstraction Layer

This module provides a unified interface for different LLM providers,
abstracting away the specific implementation details of each provider.

Supported providers:
- Gemini (Google Generative AI)
- Qwen (OpenAI-compatible API)

Usage:
    from core.llm_provider import LLMProviderFactory
    
    # Get the configured LLM provider
    provider = LLMProviderFactory.get_provider()
    
    # Synchronous call
    response = provider.generate_text("Hello, world!")
    
    # Asynchronous call
    response = await provider.generate_text_async("Hello, world!")
"""

import os
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text synchronously.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters specific to the provider
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """
        Generate text asynchronously.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters specific to the provider
            
        Returns:
            Generated text response
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the provider."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        pass


class GeminiProvider(LLMProvider):
    """Gemini LLM provider implementation."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-preview-05-20"):
        """
        Initialize Gemini provider.
        
        Args:
            api_key: Gemini API key
            model: Model name to use
        """
        self.api_key = api_key
        self.model = model
        self._initialized = False
        self._client = None
        
    def _ensure_initialized(self):
        """Ensure the Gemini client is initialized."""
        if not self._initialized:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai
                self._initialized = True
                logger.info(f"Initialized Gemini provider with model: {self.model}")
            except ImportError:
                raise ImportError("google-generativeai package is required for Gemini provider")
            except Exception as e:
                raise Exception(f"Failed to initialize Gemini provider: {str(e)}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Gemini synchronously."""
        self._ensure_initialized()
        
        try:
            logger.info(f"Calling LLM: {self.provider_name} ({self.model_name})")
            model = self._client.GenerativeModel(self.model)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation error: {str(e)}")
            raise Exception(f"Gemini generation failed: {str(e)}")
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using Gemini asynchronously."""
        self._ensure_initialized()
        
        try:
            logger.info(f"Calling LLM: {self.provider_name} ({self.model_name})")
            model = self._client.GenerativeModel(self.model)
            response = await model.generate_content_async(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini async generation error: {str(e)}")
            raise Exception(f"Gemini async generation failed: {str(e)}")
    
    @property
    def provider_name(self) -> str:
        return "Gemini"
    
    @property
    def model_name(self) -> str:
        return self.model


class QwenProvider(LLMProvider):
    """Qwen (OpenAI-compatible) LLM provider implementation."""
    
    def __init__(self, api_key: str, base_url: str, model: str = "qwen3-max"):
        """
        Initialize Qwen provider.
        
        Args:
            api_key: API key for Qwen service
            base_url: Base URL for the API
            model: Model name to use
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._client = None
        
    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                logger.info(f"Initialized Qwen provider with model: {self.model}")
            except ImportError:
                raise ImportError("openai package is required for Qwen provider")
            except Exception as e:
                raise Exception(f"Failed to initialize Qwen provider: {str(e)}")
        return self._client
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Qwen synchronously (runs async in sync context)."""
        logger.info(f"Calling LLM: {self.provider_name} ({self.model_name})")
        # Since OpenAI client is async-only, we need to run it in an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Event loop is already running, we can't use run_until_complete
                raise RuntimeError("This event loop is already running")
        except RuntimeError as e:
            if "This event loop is already running" in str(e):
                # Re-raise the specific error to be caught by callers
                raise e
            else:
                # No event loop exists, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.generate_async(prompt, **kwargs))
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Generate text using Qwen asynchronously."""
        logger.info(f"Calling LLM: {self.provider_name} ({self.model_name})")
        client = self._get_client()
        
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Qwen generation error: {str(e)}")
            raise Exception(f"Qwen generation failed: {str(e)}")
    
    @property
    def provider_name(self) -> str:
        return "Qwen"
    
    @property
    def model_name(self) -> str:
        return self.model


class LLMProviderFactory:
    """Factory class for creating LLM providers."""
    
    _instance = None
    _provider = None
    
    @classmethod
    def get_provider(cls, provider_type: Optional[str] = None) -> LLMProvider:
        """
        Get the configured LLM provider.
        
        Args:
            provider_type: Optional provider type override ('gemini' or 'qwen')
            
        Returns:
            Configured LLM provider instance
        """
        if cls._provider is None or provider_type is not None:
            cls._provider = cls._create_provider(provider_type)
        return cls._provider
    
    @classmethod
    def _create_provider(cls, provider_type: Optional[str] = None) -> LLMProvider:
        """Create a new provider instance based on configuration."""
        from .config import LLM_CONFIG
        
        # Use provided type or fall back to config
        provider_type = provider_type or LLM_CONFIG.get("provider", "gemini")
        
        if provider_type.lower() == "gemini":
            gemini_config = LLM_CONFIG.get("gemini", {})
            api_key = gemini_config.get("api_key") or os.getenv("GEMINI_API_KEY")
            model = gemini_config.get("model", "gemini-2.5-flash")
            
            if not api_key:
                raise ValueError("Gemini API key not found in config or environment")
            
            return GeminiProvider(api_key=api_key, model=model)
            
        elif provider_type.lower() == "qwen":
            qwen_config = LLM_CONFIG.get("qwen", {})
            api_key = qwen_config.get("api_key") or os.getenv("DASHSCOPE_API_KEY")
            base_url = qwen_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            model = qwen_config.get("model", "qwen3-max")
            
            if not api_key:
                raise ValueError("Qwen API key not found in config or environment")
            
            return QwenProvider(api_key=api_key, base_url=base_url, model=model)
            
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
    
    @classmethod
    def reset(cls):
        """Reset the factory (useful for testing)."""
        cls._provider = None


# Convenience functions for backward compatibility and ease of use
def get_llm_provider() -> LLMProvider:
    """Get the default LLM provider."""
    return LLMProviderFactory.get_provider()


def generate_text(prompt: str, **kwargs) -> str:
    """Generate text using the default provider (synchronous)."""
    provider = get_llm_provider()
    return provider.generate(prompt, **kwargs)


async def generate_text_async(prompt: str, **kwargs) -> str:
    """Generate text using the default provider (asynchronous)."""
    provider = get_llm_provider()
    return await provider.generate_async(prompt, **kwargs)