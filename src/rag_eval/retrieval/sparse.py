"""Sparse retrieval using BM25."""

from __future__ import annotations

import logging
from typing import Any

from llama_index.core import Document
from rank_bm25 import BM25Okapi

from rag_eval.retrieval.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


class SparseRetriever(BaseRetriever):
    """Sparse retrieval using BM25 algorithm."""

    def __init__(self) -> None:
        """Initialize BM25 retriever."""
        self.bm25: BM25Okapi | None = None
        self.documents: list[Document] = []
        self.tokenized_corpus: list[list[str]] = []

    def index(self, documents: list[Document]) -> None:
        """Index documents for BM25 retrieval.

        Args:
            documents: List of documents to index.
        """
        logger.info(f"Indexing {len(documents)} documents with BM25...")

        self.documents = documents

        # Tokenize documents (simple whitespace tokenization)
        self.tokenized_corpus = [self._tokenize(doc.text) for doc in documents]

        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info("BM25 indexing complete.")

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve documents using BM25.

        Args:
            query: Search query.
            top_k: Number of documents to retrieve.

        Returns:
            List of retrieval results sorted by BM25 score (descending).
        """
        if self.bm25 is None:
            raise ValueError("Index not created. Call index() first.")

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k document indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        # Create retrieval results
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            # Use source_doc_id from metadata for metrics matching
            doc_id = doc.metadata.get("source_doc_id", doc.doc_id)
            result = RetrievalResult(
                doc_id=doc_id,
                score=float(scores[idx]),
                text=doc.text,
            )
            results.append(result)

        return results

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.
        """
        # Simple whitespace tokenization with lowercasing
        # Can be improved with stemming/lemmatization
        return text.lower().split()

    @property
    def strategy_name(self) -> str:
        """Return strategy name."""
        return "sparse"
