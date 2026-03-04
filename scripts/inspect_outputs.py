"""Spot-check RAG outputs across all three LLM providers.

Runs 5 fixed queries (indices [0, 10, 20, 30, 40] of the SciFact test set)
through every available provider using hybrid retrieval, then prints:
  - Query text
  - Top-3 retrieved chunks
  - Each provider's raw answer
  - RAGAS faithfulness and answer-relevance score for that answer
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from textwrap import indent

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

from rag_eval.chunking import RecursiveChunker
from rag_eval.config import config
from rag_eval.data.loader import BEIRSciFact
from rag_eval.llm import AnthropicProvider, OllamaProvider, OpenAIProvider
from rag_eval.llm.base import BaseLLMProvider
from rag_eval.pipeline.rag_pipeline import RAGPipeline
from rag_eval.retrieval.dense import DenseRetriever
from rag_eval.retrieval.hybrid import HybridRetriever
from rag_eval.retrieval.sparse import SparseRetriever

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

QUERY_INDICES = [0, 10, 20, 30, 40]
COLLECTION_NAME = "scifact_inspect"


def _separator(char: str = "─", width: int = 72) -> str:
    return char * width


def _build_ragas_dataset(query: str, answer: str, contexts: list[str]) -> Dataset:
    return Dataset.from_dict(
        {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }
    )


def _score_single(query: str, answer: str, contexts: list[str]) -> dict[str, float]:
    """Return RAGAS faithfulness and answer_relevancy for a single sample."""
    if not config.openai_api_key:
        return {"faithfulness": float("nan"), "answer_relevancy": float("nan")}

    os.environ["OPENAI_API_KEY"] = config.openai_api_key
    ds = _build_ragas_dataset(query, answer, contexts)
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy])
    return {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
    }


def _build_providers() -> list[tuple[str, BaseLLMProvider]]:
    providers: list[tuple[str, BaseLLMProvider]] = []

    if config.openai_api_key:
        try:
            p = OpenAIProvider(api_key=config.openai_api_key, model=config.openai_model)
            if p.health_check():
                providers.append(("openai / " + config.openai_model, p))
                print(f"  ✓ OpenAI ({config.openai_model})")
            else:
                print(f"  ✗ OpenAI health check failed")
        except Exception as e:
            print(f"  ✗ OpenAI init failed: {e}")
    else:
        print("  – OpenAI API key not set, skipping")

    if config.anthropic_api_key:
        try:
            p = AnthropicProvider(api_key=config.anthropic_api_key, model=config.anthropic_model)
            if p.health_check():
                providers.append(("anthropic / " + config.anthropic_model, p))
                print(f"  ✓ Anthropic ({config.anthropic_model})")
            else:
                print(f"  ✗ Anthropic health check failed")
        except Exception as e:
            print(f"  ✗ Anthropic init failed: {e}")
    else:
        print("  – Anthropic API key not set, skipping")

    if config.ollama_base_url:
        try:
            p = OllamaProvider(base_url=config.ollama_base_url, model=config.ollama_model)
            if p.health_check():
                providers.append(("ollama / " + config.ollama_model, p))
                print(f"  ✓ Ollama ({config.ollama_model})")
            else:
                print(f"  ✗ Ollama not accessible at {config.ollama_base_url}")
        except Exception as e:
            print(f"  ✗ Ollama init failed: {e}")

    return providers


def main() -> None:
    print(_separator("="))
    print("RAG OUTPUT INSPECTOR")
    print(_separator("="))

    # ── 1. Dataset ──────────────────────────────────────────────────────────
    print("\nLoading SciFact dataset...")
    dataset = BEIRSciFact()
    dataset.download_and_load()
    queries_all = dataset.get_queries()
    query_ids = list(queries_all.keys())

    # Select 5 queries at the requested indices (clamp to available range)
    selected: list[tuple[str, str]] = []
    for idx in QUERY_INDICES:
        if idx < len(query_ids):
            qid = query_ids[idx]
            selected.append((qid, queries_all[qid]))
        else:
            print(f"  Warning: index {idx} out of range (only {len(query_ids)} queries), skipping")

    print(f"Selected {len(selected)} queries at indices {QUERY_INDICES}.\n")

    # ── 2. Retriever ─────────────────────────────────────────────────────────
    print("Building hybrid retriever...")
    corpus_docs = dataset.get_corpus_documents()
    chunker = RecursiveChunker(config.chunk_size, config.chunk_overlap)
    chunked_docs = chunker.chunk(corpus_docs)

    dense_retriever = DenseRetriever(
        collection_name=COLLECTION_NAME,
        embedding_model=config.embedding_model,
        qdrant_host=config.qdrant_host,
        qdrant_port=config.qdrant_port,
    )
    sparse_retriever = SparseRetriever()
    retriever = HybridRetriever(dense_retriever, sparse_retriever)
    retriever.index(chunked_docs)
    print("Retriever ready.\n")

    # ── 3. Providers ─────────────────────────────────────────────────────────
    print("Checking LLM providers:")
    providers = _build_providers()
    if not providers:
        print("\nNo providers available — check your .env configuration.")
        dense_retriever.clear_collection()
        return
    print()

    # ── 4. Pre-retrieve chunks for all 5 queries (shared across providers) ──
    print("Pre-retrieving top-3 chunks for all queries...")
    retrieved: dict[str, list] = {}
    for qid, qtext in selected:
        docs = retriever.retrieve(qtext, top_k=config.top_k)
        retrieved[qid] = docs
    print("Done.\n")

    # ── 5. Run each provider and score ───────────────────────────────────────
    for qi, (qid, qtext) in enumerate(selected):
        print(_separator("="))
        print(f"QUERY {qi + 1} of {len(selected)}  [id={qid}]")
        print(_separator())
        print(f"  {qtext}")
        print()

        # Top-3 chunks
        top_chunks = retrieved[qid][:3]
        print("TOP-3 RETRIEVED CHUNKS:")
        for rank, chunk in enumerate(top_chunks, 1):
            preview = chunk.text.replace("\n", " ")[:200]
            print(f"  [{rank}] (score={chunk.score:.4f}, doc_id={chunk.doc_id})")
            print(indent(preview + ("…" if len(chunk.text) > 200 else ""), "      "))
        print()

        contexts = [c.text for c in retrieved[qid]]

        # Per-provider answers and scores
        for provider_label, llm in providers:
            print(_separator("-"))
            print(f"  PROVIDER: {provider_label}")
            try:
                pipeline = RAGPipeline(retriever, llm, top_k=config.top_k)
                rag_result = pipeline.query(qtext)
                answer = rag_result.answer

                print("  ANSWER:")
                print(indent(answer.strip(), "    "))
                print()

                print("  Scoring with RAGAS...", end=" ", flush=True)
                scores = _score_single(qtext, answer, contexts)
                print("done.")
                print(f"  faithfulness      = {scores['faithfulness']:.4f}")
                print(f"  answer_relevancy  = {scores['answer_relevancy']:.4f}")

            except Exception as e:
                print(f"  ERROR [{type(e).__name__}]: {e}")
            print()

    # ── 6. Cleanup ────────────────────────────────────────────────────────────
    print(_separator("="))
    print("Cleaning up Qdrant collection...")
    dense_retriever.clear_collection()
    print("Done.")


if __name__ == "__main__":
    main()
