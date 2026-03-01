# production-rag-eval

## Project
Benchmarking RAG retrieval strategies and LLM providers on BEIR SciFact.
Stack: Python, LlamaIndex, Qdrant, sentence-transformers, RAGAS.

## Commands
- Run tests: `pytest tests/ -v`
- Start Qdrant: `docker-compose up -d qdrant`
- Run benchmark: `python scripts/run_benchmark.py --mode retrieval`
- Install: `pip install -e .`

## Code Standards
- Type hints on all functions (`from __future__ import annotations`)
- Docstrings on all public classes and methods
- No hardcoded secrets — everything via pydantic-settings from .env
- No Jupyter notebooks — all logic in .py files

## Git Conventions
- Commit after each section of the build order (see PRD Section 12)
- Use exact commit messages from the PRD
- Never commit .env

## Architecture Decisions
- LLM provider is config-driven via LLM_PROVIDER env var — never hardcode a provider
- Retrieval metrics (MRR, NDCG, Hit Rate) implemented from scratch, not delegated to BEIR's evaluator
- RRF fusion implemented manually in hybrid.py

## Gotchas
- RAGAS import: `from ragas.metrics import faithfulness` not `from ragas import faithfulness`
- Qdrant must be running before any indexing/retrieval tests
- Generation eval uses seed=42 on 50 sampled queries — keep this consistent

## Spec
Full project spec: see `docs/PRD.md`