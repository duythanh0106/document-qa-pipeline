"""
Provider factory for selecting API or local LLM implementations.
"""
from typing import Optional

from config import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_PROVIDER,
    OLLAMA_TIMEOUT,
    PROVIDER_CONFIGS,
    PROVIDER_KIND_MAP,
)
from providers.base import LLMProvider
from providers.llm_api.anthropic_provider import AnthropicProvider
from providers.llm_api.deepseek_provider import DeepSeekProvider
from providers.llm_api.openai_provider import OpenAIProvider
from providers.llm_local.ollama_provider import OllamaProvider


def normalize_provider_name(provider_name: str) -> str:
    if provider_name in {"api", "cloud"}:
        return DEFAULT_PROVIDER
    if provider_name in {"local", "ollama"}:
        return "ollama"
    return provider_name


def create_provider(
    provider_name: str,
    api_key: Optional[str] = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_timeout: int = OLLAMA_TIMEOUT,
    base_url: Optional[str] = None,
) -> LLMProvider:
    normalized = normalize_provider_name(provider_name)
    provider_kind = PROVIDER_KIND_MAP.get(normalized)

    if provider_kind == "api":
        config = PROVIDER_CONFIGS.get(normalized)
        if not config:
            raise ValueError(f"Unsupported API provider: {provider_name}")
        if normalized == "anthropic":
            if not api_key:
                raise ValueError("API key required for anthropic provider")
            return AnthropicProvider(api_key=api_key, model=config.model)
        if normalized == "deepseek":
            if not api_key:
                raise ValueError("API key required for deepseek provider")
            # Use custom base_url if provided, otherwise use config default
            effective_base_url = base_url or config.base_url
            return DeepSeekProvider(
                api_key=api_key,
                model=config.model,
                base_url=effective_base_url,
            )
        if not api_key:
            raise ValueError(f"API key required for {normalized} provider")
        # Use custom base_url if provided, otherwise use config default
        effective_base_url = base_url or config.base_url
        return OpenAIProvider(
            provider_name=normalized,
            api_key=api_key,
            model=config.model,
            base_url=effective_base_url,
        )

    if provider_kind == "local":
        return OllamaProvider(
            base_url=ollama_url,
            model=ollama_model,
            timeout=ollama_timeout,
        )

    raise ValueError(f"Unsupported provider: {provider_name}")