"""Tests for evaluation metrics.

Hand-crafted test cases to validate metric implementations.
"""

from __future__ import annotations

import pytest

from rag_eval.evaluation.retrieval_metrics import RetrievalEvaluator, compute_retrieval_metrics
from rag_eval.retrieval.base import RetrievalResult


class TestRetrievalMetrics:
    """Tests for retrieval metrics implementation."""

    @pytest.fixture
    def simple_qrels(self) -> dict[str, dict[str, int]]:
        """Create simple ground truth qrels."""
        return {
            "q1": {"doc1": 1, "doc2": 1, "doc3": 0},
            "q2": {"doc4": 2, "doc5": 1},
            "q3": {"doc6": 1},
        }

    @pytest.fixture
    def simple_results(self) -> dict[str, list[RetrievalResult]]:
        """Create simple retrieval results."""
        return {
            "q1": [
                RetrievalResult(doc_id="doc1", score=0.9, text="text1"),
                RetrievalResult(doc_id="doc3", score=0.8, text="text3"),
                RetrievalResult(doc_id="doc2", score=0.7, text="text2"),
            ],
            "q2": [
                RetrievalResult(doc_id="doc7", score=0.95, text="text7"),
                RetrievalResult(doc_id="doc4", score=0.85, text="text4"),
                RetrievalResult(doc_id="doc5", score=0.75, text="text5"),
            ],
            "q3": [
                RetrievalResult(doc_id="doc8", score=0.9, text="text8"),
                RetrievalResult(doc_id="doc9", score=0.8, text="text9"),
            ],
        }

    def test_mrr_perfect_ranking(self) -> None:
        """Test MRR with perfect ranking (relevant doc at rank 1)."""
        evaluator = RetrievalEvaluator(k=10)

        qrels = {"q1": {"doc1": 1}}
        results = {
            "q1": [
                RetrievalResult(doc_id="doc1", score=1.0, text="text"),
            ]
        }

        metrics = evaluator.evaluate(qrels, results)
        assert metrics.mrr_at_k == 1.0

    def test_mrr_second_rank(self) -> None:
        """Test MRR when relevant doc is at rank 2."""
        evaluator = RetrievalEvaluator(k=10)

        qrels = {"q1": {"doc2": 1}}
        results = {
            "q1": [
                RetrievalResult(doc_id="doc1", score=1.0, text="text1"),
                RetrievalResult(doc_id="doc2", score=0.9, text="text2"),
            ]
        }

        metrics = evaluator.evaluate(qrels, results)
        assert metrics.mrr_at_k == 0.5  # 1/2

    def test_mrr_no_relevant(self) -> None:
        """Test MRR when no relevant docs are retrieved."""
        evaluator = RetrievalEvaluator(k=10)

        qrels = {"q1": {"doc3": 1}}
        results = {
            "q1": [
                RetrievalResult(doc_id="doc1", score=1.0, text="text1"),
                RetrievalResult(doc_id="doc2", score=0.9, text="text2"),
            ]
        }

        metrics = evaluator.evaluate(qrels, results)
        assert metrics.mrr_at_k == 0.0

    def test_ndcg_perfect_ranking(self) -> None:
        """Test NDCG with perfect ranking."""
        evaluator = RetrievalEvaluator(k=3)

        # Perfect ranking: most relevant docs first
        qrels = {"q1": {"doc1": 2, "doc2": 1, "doc3": 0}}
        results = {
            "q1": [
                RetrievalResult(doc_id="doc1", score=1.0, text="text1"),
                RetrievalResult(doc_id="doc2", score=0.9, text="text2"),
                RetrievalResult(doc_id="doc3", score=0.8, text="text3"),
            ]
        }

        metrics = evaluator.evaluate(qrels, results)
        # With perfect ranking, NDCG should be 1.0
        assert metrics.ndcg_at_k == pytest.approx(1.0, abs=0.01)

    def test_ndcg_reversed_ranking(self) -> None:
        """Test NDCG with reversed (worst) ranking."""
        evaluator = RetrievalEvaluator(k=2)

        qrels = {"q1": {"doc1": 2, "doc2": 1}}
        results = {
            "q1": [
                RetrievalResult(doc_id="doc2", score=1.0, text="text2"),
                RetrievalResult(doc_id="doc1", score=0.9, text="text1"),
            ]
        }

        metrics = evaluator.evaluate(qrels, results)
        # NDCG should be less than 1.0 for non-optimal ranking
        assert metrics.ndcg_at_k < 1.0
        assert metrics.ndcg_at_k > 0.0

    def test_hit_rate_has_relevant(self) -> None:
        """Test Hit Rate when at least one relevant doc is retrieved."""
        evaluator = RetrievalEvaluator(k=10)

        qrels = {"q1": {"doc2": 1}}
        results = {
            "q1": [
                RetrievalResult(doc_id="doc1", score=1.0, text="text1"),
                RetrievalResult(doc_id="doc2", score=0.9, text="text2"),
                RetrievalResult(doc_id="doc3", score=0.8, text="text3"),
            ]
        }

        metrics = evaluator.evaluate(qrels, results)
        assert metrics.hit_rate_at_k == 1.0

    def test_hit_rate_no_relevant(self) -> None:
        """Test Hit Rate when no relevant docs are retrieved."""
        evaluator = RetrievalEvaluator(k=10)

        qrels = {"q1": {"doc4": 1}}
        results = {
            "q1": [
                RetrievalResult(doc_id="doc1", score=1.0, text="text1"),
                RetrievalResult(doc_id="doc2", score=0.9, text="text2"),
                RetrievalResult(doc_id="doc3", score=0.8, text="text3"),
            ]
        }

        metrics = evaluator.evaluate(qrels, results)
        assert metrics.hit_rate_at_k == 0.0

    def test_metrics_averaging(
        self,
        simple_qrels: dict[str, dict[str, int]],
        simple_results: dict[str, list[RetrievalResult]],
    ) -> None:
        """Test that metrics are correctly averaged across queries."""
        evaluator = RetrievalEvaluator(k=10)
        metrics = evaluator.evaluate(simple_qrels, simple_results)

        # Should have 3 queries
        assert metrics.num_queries == 3

        # Should have per-query scores
        assert len(metrics.per_query_mrr) == 3
        assert len(metrics.per_query_ndcg) == 3
        assert len(metrics.per_query_hit) == 3

        # Averages should be between 0 and 1
        assert 0.0 <= metrics.mrr_at_k <= 1.0
        assert 0.0 <= metrics.ndcg_at_k <= 1.0
        assert 0.0 <= metrics.hit_rate_at_k <= 1.0

    def test_compute_retrieval_metrics_convenience(self) -> None:
        """Test the convenience function."""
        qrels = {"q1": {"doc1": 1}}
        results = {
            "q1": [
                RetrievalResult(doc_id="doc1", score=1.0, text="text"),
            ]
        }

        metrics = compute_retrieval_metrics(qrels, results, k=10)

        assert metrics.mrr_at_k == 1.0
        assert metrics.ndcg_at_k == pytest.approx(1.0, abs=0.01)
        assert metrics.hit_rate_at_k == 1.0

    def test_cutoff_k_respected(self) -> None:
        """Test that cutoff k is respected."""
        evaluator = RetrievalEvaluator(k=2)

        qrels = {"q1": {"doc3": 1}}
        results = {
            "q1": [
                RetrievalResult(doc_id="doc1", score=1.0, text="text1"),
                RetrievalResult(doc_id="doc2", score=0.9, text="text2"),
                RetrievalResult(doc_id="doc3", score=0.8, text="text3"),  # Outside k=2
            ]
        }

        metrics = evaluator.evaluate(qrels, results)

        # doc3 is at rank 3, outside k=2, so should not count
        assert metrics.mrr_at_k == 0.0
        assert metrics.hit_rate_at_k == 0.0

    def test_manual_ndcg_calculation(self) -> None:
        """Test NDCG with a manually calculated example."""
        evaluator = RetrievalEvaluator(k=3)

        # Create a specific case where we can manually calculate NDCG
        qrels = {"q1": {"doc1": 1, "doc2": 1}}
        results = {
            "q1": [
                RetrievalResult(doc_id="doc1", score=1.0, text="text1"),  # rel=1
                RetrievalResult(doc_id="doc3", score=0.9, text="text3"),  # rel=0
                RetrievalResult(doc_id="doc2", score=0.8, text="text2"),  # rel=1
            ]
        }

        # Manual calculation:
        # DCG = (2^1 - 1)/log2(2) + (2^0 - 1)/log2(3) + (2^1 - 1)/log2(4)
        #     = 1/1 + 0/1.585 + 1/2
        #     = 1.0 + 0.0 + 0.5 = 1.5
        #
        # IDCG (perfect order: doc1, doc2, then others)
        # IDCG = (2^1 - 1)/log2(2) + (2^1 - 1)/log2(3) + 0
        #      = 1/1 + 1/1.585
        #      = 1.0 + 0.631 = 1.631
        #
        # NDCG = 1.5 / 1.631 ≈ 0.919

        metrics = evaluator.evaluate(qrels, results)
        expected_ndcg = 1.5 / 1.631
        assert metrics.ndcg_at_k == pytest.approx(expected_ndcg, abs=0.01)
