"""
Provider interface for LLM backends.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMProvider(ABC):
    """Common interface for LLM providers."""

    def __init__(self) -> None:
        self.last_error: Optional[str] = None

    def clear_error(self) -> None:
        self.last_error = None

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Return raw text response. Empty string on failure."""

        raise NotImplementedError
