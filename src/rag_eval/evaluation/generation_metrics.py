"""Generation evaluation metrics using RAGAS.

Uses LLM-as-judge to evaluate Faithfulness and Answer Relevance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

from rag_eval.llm.base import BaseLLMProvider
from rag_eval.pipeline.rag_pipeline import RAGResult

logger = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """Container for generation evaluation metrics."""

    faithfulness: float
    answer_relevance: float
    num_samples: int

    # Per-sample scores for variance analysis
    per_sample_faithfulness: list[float]
    per_sample_answer_relevance: list[float]


class GenerationEvaluator:
    """Evaluate generation quality using RAGAS metrics."""

    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        """Initialize generation evaluator.

        Args:
            llm_provider: LLM provider to use for evaluation (LLM-as-judge).
        """
        self.llm_provider = llm_provider

    def evaluate(
        self,
        rag_results: list[RAGResult],
    ) -> GenerationMetrics:
        """Evaluate RAG generation quality.

        Args:
            rag_results: List of RAG pipeline results to evaluate.

        Returns:
            GenerationMetrics with faithfulness and answer relevance scores.
        """
        if not rag_results:
            logger.warning("No RAG results to evaluate")
            return GenerationMetrics(
                faithfulness=0.0,
                answer_relevance=0.0,
                num_samples=0,
                per_sample_faithfulness=[],
                per_sample_answer_relevance=[],
            )

        logger.info(f"Evaluating {len(rag_results)} RAG results with RAGAS...")

        # Convert RAG results to RAGAS dataset format
        dataset = self._create_ragas_dataset(rag_results)

        # Create RAGAS LLM wrapper
        # Note: RAGAS expects specific LLM interfaces - we may need to adapt
        # For now, we'll use the default RAGAS LLM (which uses OpenAI)
        # In a production setting, we'd create a custom RAGAS LLM wrapper
        # around our BaseLLMProvider interface

        # Evaluate using RAGAS
        try:
            results = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy],
            )

            # Extract scores
            faithfulness_scores = results["faithfulness"]
            answer_relevance_scores = results["answer_relevancy"]

            # Convert to lists if needed
            if not isinstance(faithfulness_scores, list):
                faithfulness_scores = faithfulness_scores.tolist()
            if not isinstance(answer_relevance_scores, list):
                answer_relevance_scores = answer_relevance_scores.tolist()

            avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
            avg_answer_relevance = sum(answer_relevance_scores) / len(answer_relevance_scores)

            return GenerationMetrics(
                faithfulness=avg_faithfulness,
                answer_relevance=avg_answer_relevance,
                num_samples=len(rag_results),
                per_sample_faithfulness=faithfulness_scores,
                per_sample_answer_relevance=answer_relevance_scores,
            )

        except Exception as e:
            logger.error(f"Error during RAGAS evaluation: {e}")
            raise

    def _create_ragas_dataset(self, rag_results: list[RAGResult]) -> Dataset:
        """Convert RAG results to RAGAS dataset format.

        RAGAS expects a dataset with columns:
        - question: The input query
        - answer: The generated answer
        - contexts: List of retrieved context strings
        - ground_truth: Optional reference answer (not used for faithfulness/relevance)

        Args:
            rag_results: List of RAG pipeline results.

        Returns:
            HuggingFace Dataset in RAGAS format.
        """
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
        }

        for result in rag_results:
            data["question"].append(result.query)
            data["answer"].append(result.answer)

            # Extract context strings from retrieved documents
            contexts = [doc.text for doc in result.retrieved_documents]
            data["contexts"].append(contexts)

        return Dataset.from_dict(data)


def compute_generation_metrics(
    rag_results: list[RAGResult],
    llm_provider: BaseLLMProvider,
) -> GenerationMetrics:
    """Convenience function to compute generation metrics.

    Args:
        rag_results: List of RAG results to evaluate.
        llm_provider: LLM provider for evaluation.

    Returns:
        GenerationMetrics with faithfulness and answer relevance.
    """
    evaluator = GenerationEvaluator(llm_provider=llm_provider)
    return evaluator.evaluate(rag_results)
