# RAG Evaluation System Diagrams

## Architecture Diagram

This diagram shows the overall system architecture and component relationships.

```mermaid
graph TB
    subgraph "Data Layer"
        DS[BEIR SciFact Dataset]
        DS --> CORPUS[5,183 Documents]
        DS --> QUERIES[300 Queries]
        DS --> QRELS[Relevance Judgments]
    end

    subgraph "Chunking Layer"
        CORPUS --> FC[FixedChunker -512 tokens]
        CORPUS --> RC[RecursiveChunker -512 tokens]
        FC --> CHUNKS1[Chunked Documents]
        RC --> CHUNKS2[Chunked Documents]
    end

    subgraph "Embedding Layer"
        EMB[SentenceTransformer -all-MiniLM-L6-v2]
        CHUNKS1 --> EMB
        CHUNKS2 --> EMB
    end

    subgraph "Storage Layer"
        EMB --> QDRANT[(Qdrant -Vector Store)]
        QDRANT --> COLL1[Collection: -fixed_dense]
        QDRANT --> COLL2[Collection: -recursive_dense]
        QDRANT --> COLL3[Collection: -hybrid_dense]
    end

    subgraph "Retrieval Layer"
        QUERIES --> DR[DenseRetriever -Vector Search]
        QUERIES --> SR[SparseRetriever -BM25]
        QUERIES --> HR[HybridRetriever -RRF Fusion]

        COLL1 --> DR
        COLL2 --> DR
        COLL3 --> DR
        CHUNKS1 --> SR
        CHUNKS2 --> SR

        DR --> HR
        SR --> HR
    end

    subgraph "Evaluation Layer"
        DR --> METRICS[Retrieval Metrics]
        SR --> METRICS
        HR --> METRICS

        QRELS --> METRICS

        METRICS --> MRR["MRR@10"]
        METRICS --> NDCG["NDCG@10"]
        METRICS --> HR_RATE["Hit Rate@10"]
    end

    subgraph "Generation Layer"
        HR --> RAG[RAG Pipeline]
        RAG --> LLM_PROVIDERS

        LLM_PROVIDERS[LLM Providers]
        LLM_PROVIDERS --> OPENAI[OpenAI -GPT-4]
        LLM_PROVIDERS --> ANTHROPIC[Anthropic -Claude]
        LLM_PROVIDERS --> OLLAMA[Ollama -Local LLMs]

        OPENAI --> GEN_METRICS[Generation Metrics]
        ANTHROPIC --> GEN_METRICS
        OLLAMA --> GEN_METRICS

        GEN_METRICS --> FAITH[Faithfulness]
        GEN_METRICS --> REL[Relevance]
        GEN_METRICS --> CTX[Context Precision]
    end

    subgraph "Output"
        METRICS --> RESULTS1[retrieval.json]
        GEN_METRICS --> RESULTS2[llm_comparison.json]
    end

    style DS fill:#e1f5ff
    style QDRANT fill:#fff3e0
    style METRICS fill:#f3e5f5
    style GEN_METRICS fill:#f3e5f5
    style RESULTS1 fill:#e8f5e9
    style RESULTS2 fill:#e8f5e9
```

## Execution Flow Diagram

This diagram shows the step-by-step execution flow during a retrieval benchmark run.

```mermaid
sequenceDiagram
    autonumber

    participant User
    participant Script as run_benchmark.py
    participant Loader as BEIRSciFact
    participant Chunker as Chunker
    participant Embedder as SentenceTransformer
    participant Qdrant as QdrantClient
    participant Retriever as DenseRetriever
    participant Evaluator as RetrievalEvaluator
    participant Output as JSON File

    User->>Script: Run benchmark with fixed_dense strategy

    rect rgb(230, 240, 255)
        Note over Script,Loader: Phase 1: Data Loading
        Script->>Loader: load_dataset()
        Loader->>Loader: Download BEIR SciFact
        Loader-->>Script: corpus, queries, qrels
    end

    rect rgb(255, 240, 230)
        Note over Script,Chunker: Phase 2: Document Chunking
        Script->>Chunker: chunk(corpus_docs)
        loop For each document
            Chunker->>Chunker: Split into 512-token chunks with overlap
            Chunker->>Chunker: Set metadata - source_doc_id
        end
        Chunker-->>Script: chunked_docs
    end

    rect rgb(240, 255, 240)
        Note over Script,Qdrant: Phase 3: Indexing
        Script->>Retriever: DenseRetriever.index(chunks)
        Retriever->>Embedder: Initialize all-MiniLM-L6-v2

        loop For each chunk
            Retriever->>Embedder: encode(chunk.text)
            Embedder-->>Retriever: embedding vector
        end

        Retriever->>Qdrant: Create collection scifact_fixed_dense
        Retriever->>Qdrant: Upload vectors and metadata
        Qdrant-->>Retriever: Indexing complete
    end

    rect rgb(255, 245, 230)
        Note over Script,Retriever: Phase 4: Retrieval
        loop For each query
            Script->>Retriever: retrieve(query_text, top_k=10)
            Retriever->>Embedder: encode(query_text)
            Embedder-->>Retriever: query_embedding
            Retriever->>Qdrant: query_points(query_embedding)
            Qdrant-->>Retriever: top-10 similar chunks
            Retriever->>Retriever: Extract source_doc_id from metadata
            Retriever-->>Script: RetrievalResult with doc_id, score, text
        end
    end

    rect rgb(245, 235, 255)
        Note over Script,Evaluator: Phase 5: Evaluation
        Script->>Evaluator: compute_metrics(qrels, results)

        loop For each query
            Evaluator->>Evaluator: Check retrieved doc_ids against qrels
            Evaluator->>Evaluator: compute_mrr()
            Evaluator->>Evaluator: compute_ndcg()
            Evaluator->>Evaluator: compute_hit_rate()
        end

        Evaluator->>Evaluator: Average across queries
        Evaluator-->>Script: RetrievalMetrics
    end

    rect rgb(240, 255, 245)
        Note over Script,Output: Phase 6: Save Results
        Script->>Retriever: clear_collection()
        Retriever->>Qdrant: delete_collection()

        Script->>Output: Write JSON
        Output-->>Script: Saved to retrieval.json
    end

    Script-->>User: Benchmark complete with results
```

## Key Components Explained

### Data Flow

1. **BEIR SciFact Dataset** → Scientific fact-checking dataset with 5,183 documents and 300 queries
2. **Chunking** → Documents split into 512-token chunks with 50-token overlap for better retrieval
3. **Embedding** → Each chunk encoded into 384-dimensional vectors using sentence-transformers
4. **Vector Store** → Qdrant stores embeddings and enables fast similarity search
5. **Retrieval** → Queries converted to embeddings and matched against stored vectors
6. **Evaluation** → Retrieved document IDs matched against ground truth relevance judgments

### Critical Design Decisions

- **source_doc_id in metadata**: Chunks get new IDs, but original doc ID preserved for metrics matching
- **Multiple strategies**: Test fixed vs recursive chunking, dense vs sparse vs hybrid retrieval
- **From-scratch metrics**: MRR, NDCG, Hit Rate implemented manually (not using BEIR's evaluator)
- **RRF fusion**: Hybrid retrieval combines dense + sparse using Reciprocal Rank Fusion

### Metrics Explained

- **MRR@10** (Mean Reciprocal Rank): 1 / rank of first relevant document
- **NDCG@10** (Normalized Discounted Cumulative Gain): Quality-weighted ranking score
- **Hit Rate@10**: Percentage of queries with at least one relevant doc in top-10
