"""Configuration management using pydantic-settings."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM API Keys
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama base URL"
    )

    # LLM Provider Configuration
    llm_provider: str = Field(default="openai", description="LLM provider: openai | anthropic | ollama")
    llm_model: str | None = Field(default=None, description="Optional model override")

    # Vector Store
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")

    # Embeddings and Retrieval
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence transformer model for embeddings"
    )
    top_k: int = Field(default=10, description="Number of documents to retrieve")
    chunk_size: int = Field(default=512, description="Chunk size in tokens")
    chunk_overlap: int = Field(default=50, description="Chunk overlap in tokens")
    semantic_similarity_threshold: float = Field(
        default=0.85, description="Threshold for semantic chunking"
    )

    def get_llm_model_for_provider(self) -> str:
        """Get the LLM model to use, applying provider defaults if not overridden."""
        if self.llm_model:
            return self.llm_model

        # Provider defaults
        defaults = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-haiku-4-5",
            "ollama": "llama3.2",
        }
        return defaults.get(self.llm_provider, "gpt-4o-mini")


# Global config instance
config = Config()
