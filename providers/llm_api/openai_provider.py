"""
OpenAI API provider implementation.
"""
from typing import Any, Dict, List, Optional

from providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI chat API."""

    def __init__(
        self,
        provider_name: str,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.provider_name = provider_name
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        self.clear_error()
        try:
            from openai import OpenAI
        except ImportError:
            print("❌ OpenAI library not installed. Run: pip install openai")
            self.last_error = "missing_openai_library"
            return ""

        try:
            client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            client = OpenAI(**client_kwargs)
            params: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }
            if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
                params["max_tokens"] = kwargs["max_tokens"]
            if "temperature" in kwargs and kwargs["temperature"] is not None:
                params["temperature"] = kwargs["temperature"]

            response = client.chat.completions.create(**params)
            return response.choices[0].message.content or ""

        except Exception as exc:
            if self.provider_name == "deepseek":
                print(f"❌ DeepSeek API error: {exc}")
            else:
                print(f"❌ OpenAI API error: {exc}")
            self.last_error = str(exc)
            return ""
