"""
Anthropic API provider implementation.
"""
from typing import Any, Dict, List

from providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic Claude chat API."""

    def __init__(self, api_key: str, model: str) -> None:
        super().__init__()
        self.api_key = api_key
        self.model = model

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        self.clear_error()
        try:
            import anthropic
        except ImportError:
            print("❌ Anthropic library not installed. Run: pip install anthropic")
            self.last_error = "missing_anthropic_library"
            return ""

        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            max_tokens = kwargs.get("max_tokens")
            payload: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens if max_tokens is not None else 4000,
                "messages": messages,
            }
            message = client.messages.create(**payload)
            if not message.content:
                return ""
            return message.content[0].text

        except Exception as exc:
            print(f"❌ Anthropic API error: {exc}")
            self.last_error = str(exc)
            return ""
