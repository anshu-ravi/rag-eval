"""Dense retrieval using Qdrant vector store and sentence-transformers."""

from __future__ import annotations

import logging
from typing import Any

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import PrivateAttr
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from rag_eval.retrieval.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedding(BaseEmbedding):
    """Wrapper for sentence-transformers to work with LlamaIndex."""

    _model: Any = PrivateAttr()

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model.
        """
        super().__init__()
        # use a pydantic PrivateAttr to store non-field attribute
        self._model = SentenceTransformer(model_name)

    @classmethod
    def class_name(cls) -> str:
        """Return class name."""
        return "SentenceTransformerEmbedding"

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Async get query embedding."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Async get text embedding."""
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a query.

        Args:
            query: Query text.

        Returns:
            Query embedding as list of floats.
        """
        embedding = self._model.encode(query, convert_to_numpy=True)
        return embedding.tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a text.

        Args:
            text: Text to embed.

        Returns:
            Text embedding as list of floats.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embeddings.
        """
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]


class DenseRetriever(BaseRetriever):
    """Dense retrieval using Qdrant and sentence-transformers embeddings."""

    def __init__(
        self,
        collection_name: str = "scifact",
        embedding_model: str = "all-MiniLM-L6-v2",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
    ) -> None:
        """Initialize dense retriever.

        Args:
            collection_name: Name of the Qdrant collection.
            embedding_model: Name of sentence-transformers model.
            qdrant_host: Qdrant server host.
            qdrant_port: Qdrant server port.
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port

        # Initialize embedding model
        self.embed_model = SentenceTransformerEmbedding(model_name=embedding_model)

        # Initialize Qdrant client
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Vector store and index will be created during indexing
        self.vector_store: QdrantVectorStore | None = None
        self.vector_index: VectorStoreIndex | None = None

        # Store documents for later retrieval
        self.documents: dict[str, Document] = {}

    def index(self, documents: list[Document]) -> None:
        """Index documents in Qdrant.

        Args:
            documents: List of documents to index.
        """
        logger.info(f"Indexing {len(documents)} documents into Qdrant collection '{self.collection_name}'...")

        # Store documents by doc_id for retrieval
        for doc in documents:
            self.documents[doc.doc_id] = doc

        # Create vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
        )

        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Create index with custom embedding model
        self.vector_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=True,
        )

        logger.info("Indexing complete.")

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve documents using dense vector search.

        Args:
            query: Search query.
            top_k: Number of documents to retrieve.

        Returns:
            List of retrieval results.
        """
        if self.vector_index is None:
            raise ValueError("Index not created. Call index() first.")

        # Create a retriever from the index
        retriever = self.vector_index.as_retriever(similarity_top_k=top_k)

        # Retrieve nodes
        nodes = retriever.retrieve(query)

        # Convert to RetrievalResult
        results = []
        for node in nodes:
            # Use source_doc_id from metadata for metrics matching
            doc_id = node.metadata.get("source_doc_id", node.node_id)
            result = RetrievalResult(
                doc_id=doc_id,
                score=node.score if node.score is not None else 0.0,
                text=node.get_content(),
            )
            results.append(result)

        return results

    @property
    def strategy_name(self) -> str:
        """Return strategy name."""
        return "dense"

    def clear_collection(self) -> None:
        """Delete the Qdrant collection (useful for testing)."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")
