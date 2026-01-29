"""
Configuration defaults for RAG question generation.
"""
from dataclasses import dataclass
from typing import Dict, Optional

DEFAULT_PROVIDER = "deepseek"
DEFAULT_NUM_QUESTIONS = 20
DEFAULT_OUTPUT = "rag_eval.json"

MAX_RETRY = 2
MAX_TOKENS = 9999
TEMPERATURE = 0.3

DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 60


@dataclass(frozen=True)
class ProviderConfig:
    """Model configuration for a provider."""

    model: str
    base_url: Optional[str] = None


PROVIDER_CONFIGS: Dict[str, ProviderConfig] = {
    "deepseek": ProviderConfig(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
    ),
    "openai": ProviderConfig(model="gpt-4o-mini"),
    "anthropic": ProviderConfig(model="claude-sonnet-4-20250514"),
}

PROVIDER_KIND_MAP: Dict[str, str] = {
    "deepseek": "api",
    "openai": "api",
    "anthropic": "api",
    "ollama": "local",
}
