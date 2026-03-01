"""BEIR SciFact dataset loader and preprocessor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from llama_index.core import Document

logger = logging.getLogger(__name__)


class BEIRSciFact:
    """Load and preprocess the BEIR SciFact dataset."""

    def __init__(self, data_dir: str = "datasets") -> None:
        """Initialize the dataset loader.

        Args:
            data_dir: Directory to store downloaded datasets.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = "scifact"
        self.dataset_path = self.data_dir / self.dataset_name

        self.corpus: dict[str, dict[str, str]] | None = None
        self.queries: dict[str, str] | None = None
        self.qrels: dict[str, dict[str, int]] | None = None

    def download_and_load(self) -> None:
        """Download SciFact dataset if not present and load it into memory."""
        # Download if not exists
        if not self.dataset_path.exists():
            logger.info(f"Downloading {self.dataset_name} dataset...")
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
            util.download_and_unzip(url, str(self.data_dir))
            logger.info("Download complete.")

        # Load corpus, queries, and qrels
        logger.info(f"Loading {self.dataset_name} dataset from {self.dataset_path}...")
        self.corpus, self.queries, self.qrels = GenericDataLoader(
            data_folder=str(self.dataset_path)
        ).load(split="test")

        logger.info(
            f"Loaded {len(self.corpus)} documents, {len(self.queries)} queries, "
            f"{len(self.qrels)} query-document relevance judgments."
        )

    def get_corpus_documents(self) -> list[Document]:
        """Convert BEIR corpus to LlamaIndex Document objects.

        Returns:
            List of Document objects with text and metadata.
        """
        if self.corpus is None:
            raise ValueError("Dataset not loaded. Call download_and_load() first.")

        documents = []
        for doc_id, doc_data in self.corpus.items():
            # BEIR corpus format: {"_id": str, "title": str, "text": str}
            title = doc_data.get("title", "")
            text = doc_data.get("text", "")

            # Combine title and text
            full_text = f"{title}\n\n{text}" if title else text

            doc = Document(
                text=full_text,
                metadata={"doc_id": doc_id, "title": title},
                doc_id=doc_id,
            )
            documents.append(doc)

        return documents

    def get_queries(self) -> dict[str, str]:
        """Get the query set.

        Returns:
            Dictionary mapping query_id to query text.
        """
        if self.queries is None:
            raise ValueError("Dataset not loaded. Call download_and_load() first.")
        return self.queries

    def get_qrels(self) -> dict[str, dict[str, int]]:
        """Get the query relevance judgments (ground truth).

        Returns:
            Nested dict: {query_id: {doc_id: relevance_score}}.
        """
        if self.qrels is None:
            raise ValueError("Dataset not loaded. Call download_and_load() first.")
        return self.qrels

    def get_stats(self) -> dict[str, Any]:
        """Get dataset statistics.

        Returns:
            Dictionary with corpus size, query count, and avg docs per query.
        """
        if not all([self.corpus, self.queries, self.qrels]):
            raise ValueError("Dataset not loaded. Call download_and_load() first.")

        total_relevant_docs = sum(len(docs) for docs in self.qrels.values())
        avg_docs_per_query = total_relevant_docs / len(self.qrels) if self.qrels else 0

        return {
            "corpus_size": len(self.corpus),
            "num_queries": len(self.queries),
            "num_qrels": len(self.qrels),
            "avg_relevant_docs_per_query": round(avg_docs_per_query, 2),
        }
