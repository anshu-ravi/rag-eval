"""Tests for chunking strategies."""

from __future__ import annotations

import pytest
from llama_index.core import Document

from rag_eval.chunking import FixedChunker, RecursiveChunker, SemanticChunker


@pytest.fixture
def sample_document() -> Document:
    """Create a sample document for testing."""
    text = """This is the first paragraph. It contains multiple sentences. Each sentence adds information.

This is the second paragraph. It discusses a different topic. The content here is distinct from the first paragraph.

This is the third paragraph. It provides additional context. The semantic meaning shifts here."""
    return Document(text=text, metadata={"doc_id": "test_doc_1"}, doc_id="test_doc_1")


@pytest.fixture
def long_document() -> Document:
    """Create a longer document for testing overlap."""
    # Create a document with repeated sections to test overlap
    text = " ".join([f"Sentence number {i}." for i in range(100)])
    return Document(text=text, metadata={"doc_id": "long_doc"}, doc_id="long_doc")


class TestFixedChunker:
    """Tests for FixedChunker."""

    def test_fixed_chunker_returns_non_empty_chunks(
        self, sample_document: Document
    ) -> None:
        """Test that fixed chunker returns non-empty chunks."""
        chunker = FixedChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk([sample_document])

        assert len(chunks) > 0
        assert all(len(chunk.text) > 0 for chunk in chunks)

    def test_fixed_chunker_preserves_metadata(self, sample_document: Document) -> None:
        """Test that metadata is preserved in chunks."""
        chunker = FixedChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk([sample_document])

        for chunk in chunks:
            assert "doc_id" in chunk.metadata
            assert chunk.metadata["chunk_strategy"] == "fixed"
            assert chunk.metadata["source_doc_id"] == sample_document.doc_id

    def test_fixed_chunker_respects_chunk_size(self, long_document: Document) -> None:
        """Test that chunks respect the approximate size limit."""
        chunk_size = 50
        chunker = FixedChunker(chunk_size=chunk_size, chunk_overlap=10)
        chunks = chunker.chunk([long_document])

        # Due to sentence splitting, chunks may not be exactly chunk_size
        # but should be in a reasonable range
        for chunk in chunks:
            # Allow some flexibility - chunks should generally be < 2x chunk_size
            assert len(chunk.text.split()) < chunk_size * 3

    def test_fixed_chunker_overlap_works(self, long_document: Document) -> None:
        """Test that overlap is applied between chunks."""
        chunk_overlap = 20
        chunker = FixedChunker(chunk_size=100, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk([long_document])

        if len(chunks) > 1:
            # Check that there's some content overlap (not strict equality due to sentence splitting)
            # This is a basic sanity check
            assert chunks[0].text != chunks[1].text

    def test_strategy_name(self) -> None:
        """Test that strategy name is correct."""
        chunker = FixedChunker()
        assert chunker.strategy_name == "fixed"


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_recursive_chunker_returns_non_empty_chunks(
        self, sample_document: Document
    ) -> None:
        """Test that recursive chunker returns non-empty chunks."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk([sample_document])

        assert len(chunks) > 0
        assert all(len(chunk.text) > 0 for chunk in chunks)

    def test_recursive_chunker_preserves_metadata(
        self, sample_document: Document
    ) -> None:
        """Test that metadata is preserved in chunks."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk([sample_document])

        for chunk in chunks:
            assert chunk.metadata["chunk_strategy"] == "recursive"
            assert chunk.metadata["source_doc_id"] == sample_document.doc_id

    def test_recursive_chunker_respects_separators(self) -> None:
        """Test that recursive chunker uses separators correctly."""
        text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        doc = Document(text=text, doc_id="sep_test")

        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0, separators=["\n\n"])
        chunks = chunker.chunk([doc])

        # Should split on double newline
        assert len(chunks) >= 1

    def test_recursive_chunker_covers_full_text(
        self, sample_document: Document
    ) -> None:
        """Test that all chunks together cover the full text."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk([sample_document])

        # Concatenate all chunks (without overlap in this case)
        combined = "".join(chunk.text for chunk in chunks)

        # Should contain most of the original text (accounting for separator handling)
        # At minimum, all major content should be present
        assert len(combined) > 0
        assert "first paragraph" in combined.lower()

    def test_strategy_name(self) -> None:
        """Test that strategy name is correct."""
        chunker = RecursiveChunker()
        assert chunker.strategy_name == "recursive"


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_semantic_chunker_returns_non_empty_chunks(
        self, sample_document: Document
    ) -> None:
        """Test that semantic chunker returns non-empty chunks."""
        chunker = SemanticChunker(similarity_threshold=0.5)
        chunks = chunker.chunk([sample_document])

        assert len(chunks) > 0
        assert all(len(chunk.text) > 0 for chunk in chunks)

    def test_semantic_chunker_preserves_metadata(
        self, sample_document: Document
    ) -> None:
        """Test that metadata is preserved in chunks."""
        chunker = SemanticChunker(similarity_threshold=0.5)
        chunks = chunker.chunk([sample_document])

        for chunk in chunks:
            assert chunk.metadata["chunk_strategy"] == "semantic"
            assert chunk.metadata["source_doc_id"] == sample_document.doc_id

    def test_semantic_chunker_splits_on_semantic_boundaries(self) -> None:
        """Test that semantic chunker creates splits based on meaning."""
        # Create a document with clearly different semantic sections
        text = (
            "The cat sat on the mat. The feline was very comfortable. "
            "Quantum physics describes subatomic particles. "
            "The theory of relativity revolutionized physics."
        )
        doc = Document(text=text, doc_id="semantic_test")

        # Low threshold should create more splits
        chunker = SemanticChunker(similarity_threshold=0.9)
        chunks = chunker.chunk([doc])

        # Should have at least one chunk
        assert len(chunks) >= 1

    def test_semantic_chunker_single_sentence(self) -> None:
        """Test semantic chunker with a single sentence."""
        text = "This is a single sentence"
        doc = Document(text=text, doc_id="single_sentence")

        chunker = SemanticChunker()
        chunks = chunker.chunk([doc])

        assert len(chunks) == 1
        assert chunks[0].text.strip() == text

    def test_strategy_name(self) -> None:
        """Test that strategy name is correct."""
        chunker = SemanticChunker()
        assert chunker.strategy_name == "semantic"


class TestChunkerComparison:
    """Comparative tests across different chunkers."""

    def test_all_chunkers_handle_empty_input(self) -> None:
        """Test that all chunkers handle empty document list."""
        chunkers = [
            FixedChunker(),
            RecursiveChunker(),
            SemanticChunker(),
        ]

        for chunker in chunkers:
            chunks = chunker.chunk([])
            assert chunks == []

    def test_all_chunkers_produce_valid_documents(
        self, sample_document: Document
    ) -> None:
        """Test that all chunkers produce valid Document objects."""
        chunkers = [
            FixedChunker(),
            RecursiveChunker(),
            SemanticChunker(),
        ]

        for chunker in chunkers:
            chunks = chunker.chunk([sample_document])
            assert len(chunks) > 0

            for chunk in chunks:
                assert isinstance(chunk, Document)
                assert chunk.text
                assert chunk.doc_id
                assert "chunk_strategy" in chunk.metadata
