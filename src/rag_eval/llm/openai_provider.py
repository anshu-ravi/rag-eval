"""OpenAI LLM provider implementation."""

from __future__ import annotations

import logging

from openai import OpenAI

from rag_eval.llm.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key.
            model: Model to use (default: gpt-4o-mini).
        """
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def complete(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """Generate completion using OpenAI API.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.

        Returns:
            LLMResponse with completion.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            provider=self.provider_name,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    def health_check(self) -> bool:
        """Check if OpenAI API is accessible.

        Returns:
            True if API is accessible, False otherwise.
        """
        try:
            # Try a simple list models call to check connectivity
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "openai"
