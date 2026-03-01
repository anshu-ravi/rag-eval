"""Semantic chunking strategy using embedding similarity."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from llama_index.core import Document

from rag_eval.chunking.base import BaseChunker


class SemanticChunker(BaseChunker):
    """Semantic chunking based on embedding similarity between sentences.

    Splits text into sentences, computes embeddings, and creates chunk
    boundaries where cosine similarity drops below a threshold.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        sentence_split_pattern: str = ". ",
    ) -> None:
        """Initialize semantic chunker.

        Args:
            embedding_model: Name of sentence-transformers model to use.
            similarity_threshold: Minimum cosine similarity to keep sentences together.
            sentence_split_pattern: Pattern to split text into sentences.
        """
        self.embedding_model_name = embedding_model
        self.similarity_threshold = similarity_threshold
        self.sentence_split_pattern = sentence_split_pattern
        self.model = SentenceTransformer(embedding_model)

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents using semantic similarity.

        Args:
            documents: List of documents to chunk.

        Returns:
            List of semantically chunked documents.
        """
        chunked_docs = []

        for doc in documents:
            chunks = self._semantic_split(doc.text)
            for chunk_text in chunks:
                chunk_doc = Document(
                    text=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_strategy": self.strategy_name,
                        "source_doc_id": doc.doc_id,
                    },
                    doc_id=str(uuid.uuid4()),
                )
                chunked_docs.append(chunk_doc)

        return chunked_docs

    def _semantic_split(self, text: str) -> list[str]:
        """Split text based on semantic similarity.

        Args:
            text: Text to split.

        Returns:
            List of semantically coherent chunks.
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return [text]

        # Compute embeddings for each sentence
        embeddings = self.model.encode(sentences, convert_to_numpy=True)

        # Find split points based on similarity drops
        split_indices = self._find_split_points(embeddings)

        # Create chunks from split points
        chunks = []
        start_idx = 0
        for split_idx in split_indices:
            chunk = self.sentence_split_pattern.join(sentences[start_idx:split_idx])
            if chunk.strip():
                chunks.append(chunk)
            start_idx = split_idx

        # Don't forget the last chunk
        if start_idx < len(sentences):
            chunk = self.sentence_split_pattern.join(sentences[start_idx:])
            if chunk.strip():
                chunks.append(chunk)

        return chunks if chunks else [text]

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Text to split.

        Returns:
            List of sentences.
        """
        # Simple sentence splitting - can be improved with nltk/spacy
        sentences = text.split(self.sentence_split_pattern)
        # Filter out empty sentences and strip whitespace
        return [s.strip() for s in sentences if s.strip()]

    def _find_split_points(self, embeddings: NDArray[Any]) -> list[int]:
        """Find indices where similarity drops below threshold.

        Args:
            embeddings: Array of sentence embeddings.

        Returns:
            List of indices where splits should occur.
        """
        split_points = []

        for i in range(len(embeddings) - 1):
            # Compute cosine similarity between adjacent sentences
            similarity = self._cosine_similarity(embeddings[i], embeddings[i + 1])

            # If similarity drops below threshold, mark as split point
            if similarity < self.similarity_threshold:
                split_points.append(i + 1)

        return split_points

    def _cosine_similarity(self, vec1: NDArray[Any], vec2: NDArray[Any]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine similarity score.
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    @property
    def strategy_name(self) -> str:
        """Return strategy name."""
        return "semantic"
