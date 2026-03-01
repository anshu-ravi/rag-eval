"""Recursive character text splitting strategy."""

from __future__ import annotations

import uuid

from llama_index.core import Document

from rag_eval.chunking.base import BaseChunker


class RecursiveChunker(BaseChunker):
    """Recursive character text splitter.

    Splits on separators in order: ["\\n\\n", "\\n", " "], recursively
    applying the next separator if chunks are still too large.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        """Initialize recursive chunker.

        Args:
            chunk_size: Target size of each chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
            separators: List of separators to try in order. Defaults to ["\\n\\n", "\\n", " "].
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " "]

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents using recursive character splitting.

        Args:
            documents: List of documents to chunk.

        Returns:
            List of chunked documents.
        """
        chunked_docs = []
        for doc in documents:
            chunks = self._split_text(doc.text, self.separators)
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

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the given separators.

        Args:
            text: Text to split.
            separators: List of separators to try.

        Returns:
            List of text chunks.
        """
        if not separators or len(text) <= self.chunk_size:
            # Base case: no more separators or text is small enough
            return self._create_chunks_from_text(text)

        # Try splitting with the first separator
        separator = separators[0]
        splits = text.split(separator)

        # If we only got one split (separator not found), try next separator
        if len(splits) == 1:
            return self._split_text(text, separators[1:])

        # Process splits and combine them respecting chunk_size
        chunks = []
        current_chunk = ""

        for i, split in enumerate(splits):
            # Add separator back (except for last split)
            split_with_sep = split + (separator if i < len(splits) - 1 else "")

            if len(current_chunk) + len(split_with_sep) <= self.chunk_size:
                current_chunk += split_with_sep
            else:
                # Current chunk is full, save it
                if current_chunk:
                    chunks.append(current_chunk)

                # If this split is still too large, recursively split it
                if len(split) > self.chunk_size:
                    sub_chunks = self._split_text(split, separators[1:])
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split_with_sep

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        # Apply overlap
        return self._apply_overlap(chunks)

    def _create_chunks_from_text(self, text: str) -> list[str]:
        """Create chunks from text without any separator (fallback for long strings).

        Args:
            text: Text to chunk.

        Returns:
            List of chunks.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Apply overlap to chunks.

        Args:
            chunks: List of chunks without overlap.

        Returns:
            List of chunks with overlap applied.
        """
        if not chunks or self.chunk_overlap == 0:
            return chunks

        overlapped = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                # Take last chunk_overlap chars from previous chunk
                prev_overlap = chunks[i - 1][-self.chunk_overlap :]
                overlapped.append(prev_overlap + chunk)

        return overlapped

    @property
    def strategy_name(self) -> str:
        """Return strategy name."""
        return "recursive"
