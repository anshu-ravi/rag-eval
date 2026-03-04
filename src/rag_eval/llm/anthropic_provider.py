"""Anthropic LLM provider implementation."""

from __future__ import annotations

import logging

from anthropic import Anthropic

from rag_eval.llm.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM provider."""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5") -> None:
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key.
            model: Model to use (default: claude-haiku-4-5).
        """
        self.model = model
        self.client = Anthropic(api_key=api_key)

    def complete(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """Generate completion using Anthropic API.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.

        Returns:
            LLMResponse with completion.
        """
        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)

        # Extract text from content blocks
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return LLMResponse(
            content=content,
            model=response.model,
            provider=self.provider_name,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )

    def health_check(self) -> bool:
        """Check if Anthropic API is accessible.

        Returns:
            True if API is accessible, False otherwise.
        """
        try:
            # Try a simple message to check connectivity
            self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return False

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "anthropic"
