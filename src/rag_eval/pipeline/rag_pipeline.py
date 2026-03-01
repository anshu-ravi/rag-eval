"""RAG pipeline orchestrating retrieval and generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rag_eval.llm.base import BaseLLMProvider, LLMResponse
from rag_eval.retrieval.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Result from a RAG pipeline execution."""

    query: str
    answer: str
    retrieved_documents: list[RetrievalResult]
    llm_response: LLMResponse


class RAGPipeline:
    """RAG pipeline combining retrieval and generation."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm_provider: BaseLLMProvider,
        top_k: int = 10,
    ) -> None:
        """Initialize RAG pipeline.

        Args:
            retriever: Retrieval strategy to use.
            llm_provider: LLM provider for generation.
            top_k: Number of documents to retrieve.
        """
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.top_k = top_k

    def query(self, question: str, system_prompt: str | None = None) -> RAGResult:
        """Execute RAG pipeline: retrieve relevant docs and generate answer.

        Args:
            question: User question.
            system_prompt: Optional system prompt for the LLM.

        Returns:
            RAGResult with answer and retrieved documents.
        """
        logger.debug(f"Processing query: {question}")

        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question, top_k=self.top_k)
        logger.debug(f"Retrieved {len(retrieved_docs)} documents")

        # Step 2: Format context from retrieved documents
        context = self._format_context(retrieved_docs)

        # Step 3: Create prompt with context
        prompt = self._create_prompt(question, context)

        # Step 4: Generate answer using LLM
        llm_response = self.llm_provider.complete(prompt, system_prompt=system_prompt)

        logger.debug(f"Generated answer: {llm_response.content[:100]}...")

        return RAGResult(
            query=question,
            answer=llm_response.content,
            retrieved_documents=retrieved_docs,
            llm_response=llm_response,
        )

    def _format_context(self, documents: list[RetrievalResult]) -> str:
        """Format retrieved documents into a context string.

        Args:
            documents: Retrieved documents.

        Returns:
            Formatted context string.
        """
        if not documents:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(documents, start=1):
            context_parts.append(f"Document {i}:\n{doc.text}\n")

        return "\n".join(context_parts)

    def _create_prompt(self, question: str, context: str) -> str:
        """Create a prompt combining question and context.

        Args:
            question: User question.
            context: Retrieved context.

        Returns:
            Formatted prompt for the LLM.
        """
        prompt = f"""Answer the following question based on the provided context. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {question}

Answer:"""

        return prompt

    def batch_query(
        self,
        questions: list[str],
        system_prompt: str | None = None,
    ) -> list[RAGResult]:
        """Execute RAG pipeline for multiple questions.

        Args:
            questions: List of user questions.
            system_prompt: Optional system prompt for the LLM.

        Returns:
            List of RAGResults.
        """
        results = []
        for question in questions:
            result = self.query(question, system_prompt=system_prompt)
            results.append(result)

        return results
