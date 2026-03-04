"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int


class BaseLLMProvider(ABC):
    """Base class for all LLM providers."""

    @abstractmethod
    def complete(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt for context.

        Returns:
            LLMResponse with generated content and metadata.
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the provider is available and healthy.

        Returns:
            True if the provider is available, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the LLM provider."""
        pass
