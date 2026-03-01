"""Fixed-size token chunking strategy."""

from __future__ import annotations

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from rag_eval.chunking.base import BaseChunker


class FixedChunker(BaseChunker):
    """Fixed-size token chunking using LlamaIndex SentenceSplitter."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        """Initialize fixed chunker.

        Args:
            chunk_size: Target size of each chunk in tokens.
            chunk_overlap: Number of tokens to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents into fixed-size chunks.

        Args:
            documents: List of documents to chunk.

        Returns:
            List of chunked documents.
        """
        chunked_nodes = []
        for doc in documents:
            nodes = self.splitter.get_nodes_from_documents([doc])
            # Convert nodes back to documents, preserving metadata
            for node in nodes:
                chunk_doc = Document(
                    text=node.get_content(),
                    metadata={
                        **doc.metadata,
                        "chunk_strategy": self.strategy_name,
                        "source_doc_id": doc.doc_id,
                    },
                    doc_id=node.node_id,
                )
                chunked_nodes.append(chunk_doc)

        return chunked_nodes

    @property
    def strategy_name(self) -> str:
        """Return strategy name."""
        return "fixed"
