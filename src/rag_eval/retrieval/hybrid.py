"""Hybrid retrieval combining dense and sparse methods with RRF fusion."""

from __future__ import annotations

import logging
from collections import defaultdict

from llama_index.core import Document

from rag_eval.retrieval.base import BaseRetriever, RetrievalResult
from rag_eval.retrieval.dense import DenseRetriever
from rag_eval.retrieval.sparse import SparseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """Hybrid retrieval using Reciprocal Rank Fusion (RRF) of dense and sparse methods."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        rrf_k: int = 60,
    ) -> None:
        """Initialize hybrid retriever.

        Args:
            dense_retriever: Dense retriever instance.
            sparse_retriever: Sparse retriever instance.
            rrf_k: Constant for RRF formula (typically 60).
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k

    def index(self, documents: list[Document]) -> None:
        """Index documents in both dense and sparse retrievers.

        Args:
            documents: List of documents to index.
        """
        logger.info("Indexing documents for hybrid retrieval...")
        self.dense_retriever.index(documents)
        self.sparse_retriever.index(documents)
        logger.info("Hybrid indexing complete.")

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve documents using hybrid RRF fusion.

        Args:
            query: Search query.
            top_k: Number of documents to retrieve.

        Returns:
            List of retrieval results sorted by RRF score (descending).
        """
        # Retrieve from both methods (get more than top_k for better fusion)
        retrieval_k = top_k * 2

        dense_results = self.dense_retriever.retrieve(query, top_k=retrieval_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=retrieval_k)

        # Apply RRF fusion
        fused_results = self._reciprocal_rank_fusion(
            dense_results, sparse_results, top_k=top_k
        )

        return fused_results

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Fuse results using Reciprocal Rank Fusion.

        RRF formula: score(d) = Σ 1 / (k + rank_i(d))
        where k is a constant (typically 60) and rank_i(d) is the rank
        of document d in retrieval method i.

        Args:
            dense_results: Results from dense retrieval.
            sparse_results: Results from sparse retrieval.
            top_k: Number of top results to return.

        Returns:
            Fused and re-ranked results.
        """
        # Create rank mappings: doc_id -> rank (1-indexed)
        dense_ranks: dict[str, int] = {
            result.doc_id: rank + 1 for rank, result in enumerate(dense_results)
        }
        sparse_ranks: dict[str, int] = {
            result.doc_id: rank + 1 for rank, result in enumerate(sparse_results)
        }

        # Collect all unique documents
        all_doc_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        # Compute RRF scores
        rrf_scores: dict[str, float] = {}
        for doc_id in all_doc_ids:
            score = 0.0

            # Add contribution from dense retrieval
            if doc_id in dense_ranks:
                score += 1.0 / (self.rrf_k + dense_ranks[doc_id])

            # Add contribution from sparse retrieval
            if doc_id in sparse_ranks:
                score += 1.0 / (self.rrf_k + sparse_ranks[doc_id])

            rrf_scores[doc_id] = score

        # Sort by RRF score descending
        sorted_doc_ids = sorted(
            rrf_scores.keys(),
            key=lambda doc_id: rrf_scores[doc_id],
            reverse=True,
        )[:top_k]

        # Create result objects with RRF scores
        # Get text from original results
        doc_texts: dict[str, str] = {}
        for result in dense_results + sparse_results:
            if result.doc_id not in doc_texts:
                doc_texts[result.doc_id] = result.text

        fused_results = []
        for doc_id in sorted_doc_ids:
            result = RetrievalResult(
                doc_id=doc_id,
                score=rrf_scores[doc_id],
                text=doc_texts.get(doc_id, ""),
            )
            fused_results.append(result)

        return fused_results

    @property
    def strategy_name(self) -> str:
        """Return strategy name."""
        return "hybrid"
