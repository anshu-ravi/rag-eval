# Project Requirements Document

A production-grade RAG system with a rigorous evaluation framework, model-agnostic LLM abstraction, and benchmark results comparing retrieval strategies and LLM providers.

---

## 1. Goal

Build a clean, well-engineered Python package that:
1. Implements and compares **3 retrieval strategies** (fixed chunking, semantic chunking, hybrid BM25 + dense)
2. Evaluates them with **real IR and RAG metrics** (MRR, NDCG@10, Hit Rate, Faithfulness, Answer Relevance)
3. Runs across **multiple LLM providers** (OpenAI, Anthropic, Ollama) via a unified interface
4. Produces a **benchmark results table** with real numbers in the README

This project is intended to demonstrate research-to-production ownership: hypothesis → experiment → measure → conclude.

---

## 2. Dataset

**BEIR SciFact** (`allenai/sciSciFact` via HuggingFace Datasets or the BEIR library)

- ~5,183 scientific paper abstracts as the corpus
- 300 test queries with ground-truth relevance judgments
- Use the standard BEIR split: `corpus`, `queries`, `qrels`
- Load via: `from beir import util, LoggingHandler` and `from beir.datasets.data_loader import GenericDataLoader`
- Alternatively download directly: `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip`

Use the full corpus for indexing, and the test query set (300 queries) for evaluation.

---

## 3. Project Structure

```
production-rag-eval/
├── src/
│   └── rag_eval/
│       ├── __init__.py
│       ├── config.py              # Pydantic settings, loaded from .env
│       ├── data/
│       │   ├── __init__.py
│       │   └── loader.py          # BEIR SciFact loader and preprocessor
│       ├── chunking/
│       │   ├── __init__.py
│       │   ├── base.py            # Abstract base class for chunkers
│       │   ├── fixed.py           # Fixed-size token chunking
│       │   ├── recursive.py       # Recursive character text splitting
│       │   └── semantic.py        # Semantic chunking via embedding similarity
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── base.py            # Abstract base class for retrievers
│       │   ├── dense.py           # Dense retrieval (Qdrant + sentence-transformers)
│       │   ├── sparse.py          # BM25 (rank-bm25)
│       │   └── hybrid.py          # Hybrid: BM25 + dense with RRF fusion
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── base.py            # Abstract LLM provider interface
│       │   ├── openai_provider.py
│       │   ├── anthropic_provider.py
│       │   └── ollama_provider.py
│       ├── pipeline/
│       │   ├── __init__.py
│       │   └── rag_pipeline.py    # Orchestrates retrieval + generation
│       └── evaluation/
│           ├── __init__.py
│           ├── retrieval_metrics.py  # MRR, NDCG@10, Hit Rate
│           └── generation_metrics.py # Faithfulness, Answer Relevance (LLM-as-judge)
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   ├── test_llm_providers.py
│   └── test_metrics.py
├── scripts/
│   ├── run_benchmark.py       # Main benchmark runner, saves results to JSON
│   └── generate_report.py     # Reads JSON results, prints formatted table
├── docker-compose.yml         # Qdrant service
├── Dockerfile                 # App container
├── pyproject.toml
├── .env.example
└── README.md
```

---

## 4. Tech Stack

| Component | Library |
|---|---|
| RAG framework | LlamaIndex (`llama-index-core`, `llama-index-vector-stores-qdrant`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2` as default) |
| Vector store | Qdrant (via Docker) |
| Sparse retrieval | `rank-bm25` |
| Dataset | `beir` (pip install) |
| LLM providers | `openai`, `anthropic`, `ollama` (via HTTP) |
| Evaluation | `ragas` for Faithfulness/Answer Relevance; custom code for MRR/NDCG/HitRate |
| Config | `pydantic-settings` |
| Testing | `pytest` |
| Packaging | `pyproject.toml` with `[project]` table |

---

## 5. Core Abstractions

### 5.1 LLM Provider Interface

All LLM providers must implement the same interface. Config-driven — swapping provider = changing one env var.

```python
# src/rag_eval/llm/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int

class BaseLLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass
```

Implement for:
- **OpenAI**: `gpt-4o-mini` (default), `gpt-4o` (optional via config). Use `openai` SDK.
- **Anthropic**: `claude-haiku-4-5` (default), `claude-sonnet-4-6` (optional). Use `anthropic` SDK.
- **Ollama**: `llama3.2` (default). Call `http://localhost:11434/api/generate` via `httpx`.

Provider is selected by `LLM_PROVIDER` env var (`openai` | `anthropic` | `ollama`). Model is selected by `LLM_MODEL` env var (optional override; uses provider default if not set).

### 5.2 Chunker Interface

```python
# src/rag_eval/chunking/base.py
from abc import ABC, abstractmethod
from llama_index.core import Document

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]:
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        pass
```

Implement:
- **Fixed** (`chunk_size=512`, `chunk_overlap=50`): `SentenceSplitter` from LlamaIndex with fixed token count
- **Recursive** (`chunk_size=512`, `chunk_overlap=50`): `RecursiveCharacterTextSplitter`-style, splitting on `["\n\n", "\n", " "]` in order
- **Semantic**: Split by embedding similarity — compute cosine similarity between adjacent sentence embeddings, insert breaks where similarity drops below threshold (0.85 default)

### 5.3 Retriever Interface

```python
# src/rag_eval/retrieval/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    doc_id: str
    score: float
    text: str

class BaseRetriever(ABC):
    @abstractmethod
    def index(self, documents: list[Document]) -> None:
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        pass
```

Implement:
- **Dense**: Embed queries + docs with `all-MiniLM-L6-v2`, store in Qdrant, cosine similarity search
- **Sparse (BM25)**: Use `rank-bm25`, in-memory index, return top-k by BM25 score
- **Hybrid**: Run both dense and BM25, fuse with **Reciprocal Rank Fusion (RRF)**. Formula: `score(d) = Σ 1 / (k + rank_i(d))` where `k=60`. Normalise and re-rank.

---

## 6. Evaluation

### 6.1 Retrieval Metrics (computed against BEIR qrels ground truth)

Implement in `retrieval_metrics.py` without using BEIR's internal evaluator — write the metrics from scratch to demonstrate understanding:

```
MRR@10       = mean(1/rank of first relevant doc) across queries
NDCG@10      = mean normalised discounted cumulative gain at cutoff 10
Hit Rate@10  = % queries where at least one relevant doc is in top 10
```

All three computed per query, then averaged. Store per-query scores too (enables variance analysis).

### 6.2 Generation Metrics

Use **RAGAS** (`ragas.metrics.faithfulness`, `ragas.metrics.answer_relevancy`) for:
- **Faithfulness**: Does the answer contain only claims supported by the retrieved context?
- **Answer Relevance**: Does the answer address the question asked?

Both are computed using an LLM-as-judge. Pass the configured provider's LLM into RAGAS via its `llm` config. Score range: 0–1.

For generation evaluation, sample **50 queries** from the test set (not all 300 — too slow/expensive). Use a fixed random seed (`seed=42`) for reproducibility.

### 6.3 Benchmark Matrix

The benchmark runner must support two experimental axes:

**Axis 1: Retrieval Strategy** (hold LLM fixed to `gpt-4o-mini`)
| Strategy | MRR@10 | NDCG@10 | Hit Rate@10 | Faithfulness | Answer Relevance |
|---|---|---|---|---|---|
| Fixed Chunking + Dense | | | | | |
| Recursive Chunking + Dense | | | | | |
| Hybrid (BM25 + Dense) | | | | | |

**Axis 2: LLM Provider** (hold retrieval fixed to best strategy from Axis 1)
| Provider / Model | Faithfulness | Answer Relevance | Avg Latency (s) | Cost per query |
|---|---|---|---|---|
| OpenAI gpt-4o-mini | | | | |
| OpenAI gpt-4o | | | | |
| Anthropic claude-haiku-4-5 | | | | |
| Anthropic claude-sonnet-4-6 | | | | |
| Ollama llama3.2 (local) | | | | |

These tables with real numbers go directly into the README.

---

## 7. Benchmark Runner

`scripts/run_benchmark.py` — CLI script with the following flags:

```bash
# Run full retrieval benchmark (all 3 strategies, all 300 queries)
python scripts/run_benchmark.py --mode retrieval --output results/retrieval.json

# Run LLM comparison (50 sampled queries, best retrieval strategy, all configured providers)
python scripts/run_benchmark.py --mode llm_comparison --output results/llm_comparison.json

# Run a single strategy + provider combo (for quick testing)
python scripts/run_benchmark.py --mode single --strategy hybrid --provider openai --output results/single.json
```

Results saved as structured JSON:
```json
{
  "experiment": "retrieval_benchmark",
  "dataset": "scifact",
  "timestamp": "...",
  "config": { "embedding_model": "...", "top_k": 10 },
  "results": [
    {
      "strategy": "hybrid",
      "mrr_at_10": 0.423,
      "ndcg_at_10": 0.481,
      "hit_rate_at_10": 0.712,
      "n_queries": 300
    }
  ]
}
```

`scripts/generate_report.py` reads the JSON and prints formatted markdown tables to stdout, which can be pasted into the README.

---

## 8. Configuration

All secrets and settings via `.env` (loaded with `pydantic-settings`):

```bash
# .env.example
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434

LLM_PROVIDER=openai           # openai | anthropic | ollama
LLM_MODEL=gpt-4o-mini         # optional override

QDRANT_HOST=localhost
QDRANT_PORT=6333

EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K=10
CHUNK_SIZE=512
CHUNK_OVERLAP=50
SEMANTIC_SIMILARITY_THRESHOLD=0.85
```

`config.py` uses `pydantic-settings` `BaseSettings` to load and validate all values. Pass the config object via dependency injection — no `os.getenv()` calls scattered through the codebase.


Use poetry to manage all dependency management. Create separate virtual env locally and maintain everything properly

---

## 9. Docker Setup
`
`docker-compose.yml` — two services:
1. **qdrant**: `qdrant/qdrant:latest`, ports `6333:6333`, volume `qdrant_storage:/qdrant/storage`
2. **app**: builds from `Dockerfile`, mounts `.env`, depends on `qdrant`

`Dockerfile`:
- Base: `python:3.11-slim`
- Install system deps: `build-essential`
- Copy `pyproject.toml` + `src/` + `scripts/`
- `pip install -e .`
- Entrypoint: `python scripts/run_benchmark.py`

---

## 10. Tests

Minimum test coverage required — focus on correctness, not mocking everything:

**`test_chunking.py`**: Given a sample document, assert each chunker returns non-empty chunks, chunks cover the full text, overlap works correctly for fixed chunking.

**`test_retrieval.py`**: Index 10 sample documents, query for something present, assert BM25/dense/hybrid all return the correct document in top-3. Test RRF fusion produces scores in valid range.

**`test_llm_providers.py`**: Mock the HTTP calls (don't call real APIs in tests). Assert the provider interface returns a correctly structured `LLMResponse` with all fields populated.

**`test_metrics.py`**: Hand-craft a small qrels dict and retrieval result, compute MRR/NDCG/HitRate manually, assert `retrieval_metrics.py` produces matching values. This is the most important test — it validates your eval logic.

Run with: `pytest tests/ -v`

---

## 11. README Structure

The README should read like a mini research report. Structure:

1. **Problem Statement** — Why does RAG evaluation matter? What gap does this project address?
2. **Approach** — Brief description of retrieval strategies, evaluation methodology, model-agnostic design
3. **Quick Start** — `docker-compose up -d`, `pip install -e .`, `cp .env.example .env`, `python scripts/run_benchmark.py`
4. **Benchmark Results** — Two tables with real numbers (see Section 6.3)
5. **Key Findings** — 3–5 bullet observations from the results (e.g. "Hybrid retrieval outperforms dense-only by X% on NDCG@10", "Haiku achieves 94% of GPT-4o Faithfulness at 10x lower cost")
6. **Architecture** — Brief explanation of the provider abstraction and why it was designed this way
7. **Repo Structure** — File tree with one-line descriptions

---

## 12. Build Order

Implement in this order to unblock testing at each step. **After completing each step, make a git commit with the message shown.** Do not batch commits — each step should be its own commit so the history tells a clear story.

1. `pyproject.toml` + `config.py` + `data/loader.py` — get data loading working first
   `git commit -m "feat: add project structure, config, and data loader"`

2. `chunking/` — all three chunkers with tests
   `git commit -m "feat: implement fixed, recursive, and semantic chunking strategies"`

3. `retrieval/dense.py` + Qdrant docker setup — verify basic indexing and retrieval
   `git commit -m "feat: dense retrieval with Qdrant and sentence-transformers"`

4. `retrieval/sparse.py` + `retrieval/hybrid.py` — BM25 and RRF fusion
   `git commit -m "feat: sparse BM25 and hybrid RRF retrieval"`

5. `evaluation/retrieval_metrics.py` — MRR, NDCG, Hit Rate with tests
   `git commit -m "feat: retrieval metrics — MRR, NDCG@10, Hit Rate"`

6. `llm/` — all three providers behind the abstract interface
   `git commit -m "feat: model-agnostic LLM interface with OpenAI, Anthropic, and Ollama providers"`

7. `pipeline/rag_pipeline.py` — wire retrieval + generation together
   `git commit -m "feat: RAG pipeline orchestrating retrieval and generation"`

8. `evaluation/generation_metrics.py` — RAGAS integration
   `git commit -m "feat: generation metrics — Faithfulness and Answer Relevance via RAGAS"`

9. `scripts/run_benchmark.py` + `scripts/generate_report.py`
   `git commit -m "feat: benchmark runner and report generator"`

10. `Dockerfile` + `docker-compose.yml`
    `git commit -m "feat: Docker setup for Qdrant and app container"`

11. README with real benchmark numbers filled in after runs
    `git commit -m "docs: README with benchmark results and key findings"`

Before starting, initialise the repo:
```bash
git init
git add .gitignore .env.example
git commit -m "chore: initial repo setup"
```

Make sure `.env` (with real keys) is in `.gitignore` and never committed.

---

## 13. Quality Standards

- No Jupyter notebooks as primary artifacts — all logic in `.py` files
- Type hints throughout (`from __future__ import annotations` at top of each file)
- Docstrings on all public classes and methods
- No hardcoded API keys or paths — everything via config
- Retrieval metrics implemented from scratch (not delegated entirely to a library) — this demonstrates understanding
- RRF fusion implemented manually — not a library call
- All results reproducible with `seed=42` where randomness is involved