"""Retrieval evaluation metrics: MRR, NDCG, Hit Rate.

Implemented from scratch to demonstrate understanding of the metrics.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from rag_eval.retrieval.base import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""

    mrr_at_k: float
    ndcg_at_k: float
    hit_rate_at_k: float
    num_queries: int

    # Per-query scores for variance analysis
    per_query_mrr: list[float]
    per_query_ndcg: list[float]
    per_query_hit: list[float]


class RetrievalEvaluator:
    """Evaluate retrieval performance using standard IR metrics."""

    def __init__(self, k: int = 10) -> None:
        """Initialize evaluator.

        Args:
            k: Cutoff for metrics (e.g., MRR@10, NDCG@10).
        """
        self.k = k

    def evaluate(
        self,
        qrels: dict[str, dict[str, int]],
        results: dict[str, list[RetrievalResult]],
    ) -> RetrievalMetrics:
        """Evaluate retrieval results against ground truth qrels.

        Args:
            qrels: Ground truth relevance judgments.
                   Format: {query_id: {doc_id: relevance_score}}
            results: Retrieval results for each query.
                     Format: {query_id: [RetrievalResult, ...]}

        Returns:
            RetrievalMetrics with averaged scores and per-query scores.
        """
        mrr_scores = []
        ndcg_scores = []
        hit_scores = []

        for query_id in qrels.keys():
            if query_id not in results:
                logger.warning(f"No results for query {query_id}, skipping...")
                continue

            retrieved_docs = results[query_id][: self.k]
            relevant_docs = qrels[query_id]

            # Compute metrics for this query
            mrr = self._compute_mrr(retrieved_docs, relevant_docs)
            ndcg = self._compute_ndcg(retrieved_docs, relevant_docs, self.k)
            hit = self._compute_hit_rate(retrieved_docs, relevant_docs)

            mrr_scores.append(mrr)
            ndcg_scores.append(ndcg)
            hit_scores.append(hit)

        # Compute averages
        avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
        avg_hit_rate = sum(hit_scores) / len(hit_scores) if hit_scores else 0.0

        return RetrievalMetrics(
            mrr_at_k=avg_mrr,
            ndcg_at_k=avg_ndcg,
            hit_rate_at_k=avg_hit_rate,
            num_queries=len(mrr_scores),
            per_query_mrr=mrr_scores,
            per_query_ndcg=ndcg_scores,
            per_query_hit=hit_scores,
        )

    def _compute_mrr(
        self,
        retrieved: list[RetrievalResult],
        relevant: dict[str, int],
    ) -> float:
        """Compute Mean Reciprocal Rank for a single query.

        MRR = 1 / rank of first relevant document (0 if none found)

        Args:
            retrieved: List of retrieved documents (ranked).
            relevant: Dictionary of relevant doc_ids with relevance scores.

        Returns:
            MRR score for this query.
        """
        for rank, result in enumerate(retrieved, start=1):
            if result.doc_id in relevant and relevant[result.doc_id] > 0:
                return 1.0 / rank

        return 0.0

    def _compute_ndcg(
        self,
        retrieved: list[RetrievalResult],
        relevant: dict[str, int],
        k: int,
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain.

        DCG@k = Σ (2^rel_i - 1) / log2(i + 1)
        NDCG@k = DCG@k / IDCG@k

        Args:
            retrieved: List of retrieved documents (ranked).
            relevant: Dictionary of relevant doc_ids with relevance scores.
            k: Cutoff.

        Returns:
            NDCG@k score for this query.
        """
        # Compute DCG
        dcg = 0.0
        for i, result in enumerate(retrieved[:k], start=1):
            relevance = relevant.get(result.doc_id, 0)
            # DCG formula: (2^rel - 1) / log2(i + 1)
            dcg += (2**relevance - 1) / math.log2(i + 1)

        # Compute ideal DCG (IDCG)
        # Sort relevance scores in descending order
        ideal_relevances = sorted(relevant.values(), reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevances, start=1):
            idcg += (2**relevance - 1) / math.log2(i + 1)

        # Avoid division by zero
        if idcg == 0:
            return 0.0

        return dcg / idcg

    def _compute_hit_rate(
        self,
        retrieved: list[RetrievalResult],
        relevant: dict[str, int],
    ) -> float:
        """Compute Hit Rate (also known as Recall@k or Success@k).

        Hit Rate = 1 if at least one relevant doc in top-k, else 0

        Args:
            retrieved: List of retrieved documents (ranked).
            relevant: Dictionary of relevant doc_ids with relevance scores.

        Returns:
            Hit rate (1.0 or 0.0) for this query.
        """
        for result in retrieved:
            if result.doc_id in relevant and relevant[result.doc_id] > 0:
                return 1.0

        return 0.0


def compute_retrieval_metrics(
    qrels: dict[str, dict[str, int]],
    results: dict[str, list[RetrievalResult]],
    k: int = 10,
) -> RetrievalMetrics:
    """Convenience function to compute retrieval metrics.

    Args:
        qrels: Ground truth relevance judgments.
        results: Retrieval results for each query.
        k: Cutoff for metrics.

    Returns:
        RetrievalMetrics with all computed metrics.
    """
    evaluator = RetrievalEvaluator(k=k)
    return evaluator.evaluate(qrels, results)
