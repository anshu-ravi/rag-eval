"""Benchmark runner script.

Run benchmarks for retrieval strategies and LLM providers.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_eval.chunking import FixedChunker, RecursiveChunker
from rag_eval.config import config
from rag_eval.data.loader import BEIRSciFact
from rag_eval.evaluation.generation_metrics import compute_generation_metrics
from rag_eval.evaluation.retrieval_metrics import compute_retrieval_metrics
from rag_eval.llm import OpenAIProvider, AnthropicProvider, OllamaProvider
from rag_eval.pipeline.rag_pipeline import RAGPipeline
from rag_eval.retrieval.dense import DenseRetriever
from rag_eval.retrieval.hybrid import HybridRetriever
from rag_eval.retrieval.sparse import SparseRetriever
from rag_eval.retrieval.base import RetrievalResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dataset() -> BEIRSciFact:
    """Load BEIR SciFact dataset."""
    logger.info("Loading BEIR SciFact dataset...")
    dataset = BEIRSciFact()
    dataset.download_and_load()
    stats = dataset.get_stats()
    logger.info(f"Dataset stats: {stats}")
    return dataset


def run_retrieval_benchmark(output_path: str, selected_strategies: list[str] | None = None) -> None:
    """Run full retrieval benchmark across all strategies.

    Args:
        output_path: Path to save results JSON.
    """
    logger.info("Starting retrieval benchmark...")

    # Load dataset
    dataset = load_dataset()
    corpus_docs = dataset.get_corpus_documents()
    queries = dataset.get_queries()
    qrels = dataset.get_qrels()

    # Test different chunking + retrieval combinations
    all_strategies = [
        ("fixed_dense", FixedChunker(config.chunk_size, config.chunk_overlap)),
        ("recursive_dense", RecursiveChunker(config.chunk_size, config.chunk_overlap)),
    ]

    # Filter strategies if specific ones are selected
    if selected_strategies:
        strategies = [(name, chunker) for name, chunker in all_strategies
                     if name in selected_strategies]
    else:
        strategies = all_strategies

    results = []

    for strategy_name, chunker in strategies:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing strategy: {strategy_name}")
        logger.info(f"{'='*60}")

        # Chunk documents
        chunked_docs = chunker.chunk(corpus_docs)
        logger.info(f"Created {len(chunked_docs)} chunks")

        # Create dense retriever
        retriever = DenseRetriever(
            collection_name=f"scifact_{strategy_name}",
            embedding_model=config.embedding_model,
            qdrant_host=config.qdrant_host,
            qdrant_port=config.qdrant_port,
        )

        # Index documents
        retriever.index(chunked_docs)

        # Retrieve for all queries
        retrieval_results: dict[str, list[RetrievalResult]] = {}
        for query_id, query_text in queries.items():
            retrieval_results[query_id] = retriever.retrieve(query_text, top_k=config.top_k)

        # Evaluate
        metrics = compute_retrieval_metrics(qrels, retrieval_results, k=config.top_k)

        result = {
            "strategy": strategy_name,
            "chunker": chunker.strategy_name,
            "retriever": "dense",
            "mrr_at_10": round(metrics.mrr_at_k, 4),
            "ndcg_at_10": round(metrics.ndcg_at_k, 4),
            "hit_rate_at_10": round(metrics.hit_rate_at_k, 4),
            "n_queries": metrics.num_queries,
        }
        results.append(result)
        logger.info(f"Results: {result}")

        # Clean up collection
        retriever.clear_collection()

    # Test hybrid retrieval
    if not selected_strategies or "hybrid" in selected_strategies:
        logger.info(f"\n{'='*60}")
        logger.info("Testing hybrid retrieval")
        logger.info(f"{'='*60}")

        # Use recursive chunking for hybrid
        chunker = RecursiveChunker(config.chunk_size, config.chunk_overlap)
        chunked_docs = chunker.chunk(corpus_docs)

        dense_retriever = DenseRetriever(
            collection_name="scifact_hybrid_dense",
            embedding_model=config.embedding_model,
            qdrant_host=config.qdrant_host,
            qdrant_port=config.qdrant_port,
        )
        sparse_retriever = SparseRetriever()
        hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)

        hybrid_retriever.index(chunked_docs)

        # Retrieve for all queries
        retrieval_results = {}
        for query_id, query_text in queries.items():
            retrieval_results[query_id] = hybrid_retriever.retrieve(query_text, top_k=config.top_k)

        # Evaluate
        metrics = compute_retrieval_metrics(qrels, retrieval_results, k=config.top_k)

        result = {
            "strategy": "hybrid",
            "chunker": "recursive",
            "retriever": "hybrid_rrf",
            "mrr_at_10": round(metrics.mrr_at_k, 4),
            "ndcg_at_10": round(metrics.ndcg_at_k, 4),
            "hit_rate_at_10": round(metrics.hit_rate_at_k, 4),
            "n_queries": metrics.num_queries,
        }
        results.append(result)
        logger.info(f"Results: {result}")

        # Clean up
        dense_retriever.clear_collection()

    # Save results
    output = {
        "experiment": "retrieval_benchmark",
        "dataset": "scifact",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "embedding_model": config.embedding_model,
            "top_k": config.top_k,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
        },
        "results": results,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


def run_llm_comparison(output_path: str, best_strategy: str = "hybrid") -> None:
    """Run LLM comparison using the best retrieval strategy.

    Args:
        output_path: Path to save results JSON.
        best_strategy: Best retrieval strategy to use.
    """
    logger.info("Starting LLM comparison benchmark...")

    # Load dataset
    dataset = load_dataset()
    corpus_docs = dataset.get_corpus_documents()
    queries = dataset.get_queries()

    # Sample queries for evaluation (50 queries with seed=42)
    random.seed(42)
    sampled_query_ids = random.sample(list(queries.keys()), min(50, len(queries)))
    sampled_queries = {qid: queries[qid] for qid in sampled_query_ids}

    logger.info(f"Sampled {len(sampled_queries)} queries for LLM evaluation")

    # Setup retrieval (using best strategy)
    chunker = RecursiveChunker(config.chunk_size, config.chunk_overlap)
    chunked_docs = chunker.chunk(corpus_docs)

    if best_strategy == "hybrid":
        dense_retriever = DenseRetriever(
            collection_name="scifact_llm_eval",
            embedding_model=config.embedding_model,
            qdrant_host=config.qdrant_host,
            qdrant_port=config.qdrant_port,
        )
        sparse_retriever = SparseRetriever()
        retriever = HybridRetriever(dense_retriever, sparse_retriever)
    else:
        retriever = DenseRetriever(
            collection_name="scifact_llm_eval",
            embedding_model=config.embedding_model,
            qdrant_host=config.qdrant_host,
            qdrant_port=config.qdrant_port,
        )

    retriever.index(chunked_docs)

    # Test different LLM providers
    providers_config = [
        ("openai", "gpt-4o-mini", OpenAIProvider),
        ("anthropic", "claude-haiku-4-5", AnthropicProvider),
    ]

    # Add Ollama if configured
    if config.ollama_base_url:
        providers_config.append(("ollama", "llama3.2", OllamaProvider))

    results = []

    for provider_name, model_name, provider_class in providers_config:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing provider: {provider_name} / {model_name}")
        logger.info(f"{'='*60}")

        # Create provider instance
        if provider_name == "openai":
            if not config.openai_api_key:
                logger.warning("OpenAI API key not configured, skipping...")
                continue
            llm_provider = provider_class(api_key=config.openai_api_key, model=model_name)
        elif provider_name == "anthropic":
            if not config.anthropic_api_key:
                logger.warning("Anthropic API key not configured, skipping...")
                continue
            llm_provider = provider_class(api_key=config.anthropic_api_key, model=model_name)
        elif provider_name == "ollama":
            llm_provider = provider_class(base_url=config.ollama_base_url, model=model_name)
        else:
            continue

        # Create RAG pipeline
        rag_pipeline = RAGPipeline(retriever, llm_provider, top_k=config.top_k)

        # Run RAG for sampled queries
        rag_results = []
        for query_id, query_text in sampled_queries.items():
            try:
                result = rag_pipeline.query(query_text)
                rag_results.append(result)
            except Exception as e:
                logger.error(f"Error processing query {query_id}: {e}")
                continue

        if not rag_results:
            logger.warning(f"No results for {provider_name}, skipping evaluation")
            continue

        # Evaluate generation quality with RAGAS
        try:
            gen_metrics = compute_generation_metrics(rag_results, llm_provider)

            result = {
                "provider": provider_name,
                "model": model_name,
                "faithfulness": round(gen_metrics.faithfulness, 4),
                "answer_relevance": round(gen_metrics.answer_relevance, 4),
                "n_samples": gen_metrics.num_samples,
            }
            results.append(result)
            logger.info(f"Results: {result}")
        except Exception as e:
            logger.error(f"Error during RAGAS evaluation for {provider_name}: {e}")

    # Clean up
    if hasattr(retriever, "clear_collection"):
        retriever.clear_collection()
    elif hasattr(retriever, "dense_retriever"):
        retriever.dense_retriever.clear_collection()

    # Save results
    output = {
        "experiment": "llm_comparison",
        "dataset": "scifact",
        "retrieval_strategy": best_strategy,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "top_k": config.top_k,
            "n_sampled_queries": len(sampled_queries),
            "random_seed": 42,
        },
        "results": results,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


def run_single(strategy: str, provider: str, output_path: str) -> None:
    """Run a single strategy + provider combination.

    Args:
        strategy: Retrieval strategy name.
        provider: LLM provider name.
        output_path: Path to save results JSON.
    """
    logger.info(f"Running single benchmark: {strategy} + {provider}")
    # For now, just run a basic version
    # Can be expanded based on needs
    logger.info("Single mode not fully implemented yet. Use retrieval or llm_comparison mode.")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run RAG benchmarks")
    parser.add_argument(
        "--mode",
        choices=["retrieval", "llm_comparison", "single"],
        required=True,
        help="Benchmark mode to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/benchmark.json",
        help="Output file path for results",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy for single mode (hybrid, dense)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Provider for single mode (openai, anthropic, ollama)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        help="Comma-separated list of strategies to run (fixed_dense, recursive_dense, hybrid). If not specified, runs all.",
    )

    args = parser.parse_args()

    if args.mode == "retrieval":
        selected_strategies = args.strategies.split(",") if args.strategies else None
        run_retrieval_benchmark(args.output, selected_strategies)
    elif args.mode == "llm_comparison":
        run_llm_comparison(args.output)
    elif args.mode == "single":
        if not args.strategy or not args.provider:
            parser.error("--strategy and --provider required for single mode")
        run_single(args.strategy, args.provider, args.output)


if __name__ == "__main__":
    main()
