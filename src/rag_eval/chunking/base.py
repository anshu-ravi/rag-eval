"""Abstract base class for chunking strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llama_index.core import Document


class BaseChunker(ABC):
    """Base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks.

        Args:
            documents: List of LlamaIndex Document objects to chunk.

        Returns:
            List of chunked Document objects.
        """
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the name of the chunking strategy."""
        pass
