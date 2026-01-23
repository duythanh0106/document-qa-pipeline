"""
DeepSeek API provider implementation.
"""
import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from providers.base import LLMProvider

DEFAULT_TIMEOUT = 360


def call_openai_compat(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]


class DeepSeekProvider(LLMProvider):
    """Provider for DeepSeek OpenAI-compatible chat API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.deepseek.com"
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        self.clear_error()
        try:
            max_tokens = kwargs.get("max_tokens")
            temperature = kwargs.get("temperature")
            timeout = kwargs.get("timeout", self.timeout)

            if max_tokens is None:
                max_tokens = 4000
            if temperature is None:
                temperature = 0.7

            return call_openai_compat(
                base_url=self.base_url,
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception as exc:
            print(f"❌ DeepSeek API error: {exc}")
            self.last_error = str(exc)
            return ""
