"""
Ollama local provider implementation.
"""
import json
import urllib.error
import urllib.request
from typing import Any, Dict, List

from providers.base import LLMProvider


class OllamaProvider(LLMProvider):
    """Provider for Ollama local chat API."""

    def __init__(self, base_url: str, model: str, timeout: int = 60) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        self.clear_error()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if "options" in kwargs and kwargs["options"] is not None:
            payload["options"] = kwargs["options"]

        url = f"{self.base_url}/api/chat"
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, method="POST")
        request.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")

            response_data = json.loads(body)
            message = response_data.get("message", {})
            return message.get("content", "")

        except Exception as exc:
            print(f"❌ Ollama API error: {exc}")
            self.last_error = str(exc)
            return ""


def list_models(base_url: str, timeout: int = 10) -> List[str]:
    """Return available Ollama models, or empty list on error."""

    url = f"{base_url.rstrip('/')}/api/tags"
    request = urllib.request.Request(url, method="GET")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")

        data = json.loads(body)
        models = data.get("models", [])
        return [model.get("name") for model in models if model.get("name")]

    except Exception:
        return []
