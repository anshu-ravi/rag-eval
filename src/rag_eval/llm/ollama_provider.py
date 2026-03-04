"""Ollama LLM provider implementation."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from rag_eval.llm.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
    ) -> None:
        """Initialize Ollama provider.

        Args:
            base_url: Ollama server base URL.
            model: Model to use (default: llama3.2).
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.generate_url = f"{self.base_url}/api/generate"

    def complete(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        """Generate completion using Ollama API.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.

        Returns:
            LLMResponse with completion.
        """
        # Combine system and user prompts
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
        }

        with httpx.Client(timeout=120.0) as client:
            response = client.post(self.generate_url, json=payload)
            response.raise_for_status()
            data = response.json()

        # Extract token counts (if available)
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return LLMResponse(
            content=data.get("response", ""),
            model=self.model,
            provider=self.provider_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def health_check(self) -> bool:
        """Check if Ollama service is accessible.

        Returns:
            True if Ollama service is accessible, False otherwise.
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "ollama"
