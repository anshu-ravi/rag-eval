"""Generation evaluation metrics using RAGAS.

Uses LLM-as-judge to evaluate Faithfulness and Answer Relevance.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
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

    def __init__(self, llm_provider: BaseLLMProvider, openai_api_key: str | None = None) -> None:
        """Initialize generation evaluator.

        Args:
            llm_provider: LLM provider used for generation (for reference).
            openai_api_key: OpenAI API key for the RAGAS judge LLM.
        """
        self.llm_provider = llm_provider
        self.openai_api_key = openai_api_key

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

        # RAGAS and LangChain read OPENAI_API_KEY from os.environ at every layer.
        # Set it explicitly from config so it's available regardless of how the process was launched.
        if not self.openai_api_key:
            raise ValueError("openai_api_key is required for RAGAS evaluation")
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        # Evaluate using RAGAS
        try:
            results = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy],
            )

            # In RAGAS 0.1.x, results is a dict subclass: results["metric"] = aggregate float.
            # results.scores is a HuggingFace Dataset with per-sample columns.
            avg_faithfulness = float(results["faithfulness"])
            avg_answer_relevance = float(results["answer_relevancy"])

            per_faithfulness = [float(x) if x is not None else 0.0 for x in results.scores["faithfulness"]]
            per_answer_relevance = [float(x) if x is not None else 0.0 for x in results.scores["answer_relevancy"]]

            return GenerationMetrics(
                faithfulness=avg_faithfulness,
                answer_relevance=avg_answer_relevance,
                num_samples=len(rag_results),
                per_sample_faithfulness=per_faithfulness,
                per_sample_answer_relevance=per_answer_relevance,
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
    openai_api_key: str | None = None,
) -> GenerationMetrics:
    """Convenience function to compute generation metrics.

    Args:
        rag_results: List of RAG results to evaluate.
        llm_provider: LLM provider used for generation.
        openai_api_key: OpenAI API key for the RAGAS judge LLM.

    Returns:
        GenerationMetrics with faithfulness and answer relevance.
    """
    evaluator = GenerationEvaluator(llm_provider=llm_provider, openai_api_key=openai_api_key)
    return evaluator.evaluate(rag_results)
