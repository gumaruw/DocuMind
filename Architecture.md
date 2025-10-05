# DocuMind Architecture

## System Overview

DocuMind is a RAG-based document analysis system with three implementation variants. Each version processes PDF documents, extracts structured content, and answers natural language queries.

## Implementation Variants

### V2 (LocalRagV2)
Production-grade implementation using external LLM and robust vector search.

**Components:**
- PDF Processing: pdfplumber + Docling library
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector Store: FAISS (IndexFlatL2)
- LLM: Ollama client (Phi4 model, local)
- Config: YAML-based configuration

**Data Flow:**
1. PDF → Docling converter → structured content (text/table/image)
2. Content → markdown tagging → text chunks
3. Chunks → SentenceTransformer → embeddings
4. Embeddings → FAISS index
5. Query → FAISS search → relevant chunks
6. Chunks + Query → Ollama/Phi4 → answer

**Key Characteristics:**
- Chunk size: 500 tokens
- Uses external LLM for generation
- Language-agnostic embeddings
- Minimal preprocessing

### V3 (LocalRagV3)
Turkish-optimized implementation without external LLM dependency.

**Components:**
- PDF Processing: PyMuPDF (fitz) with custom table detection
- Embeddings: dbmdz/bert-base-turkish-cased
- Vector Store: ChromaDB (persistent, cosine similarity)
- NLP: Transformers fill-mask pipeline (no LLM)
- UI: Streamlit web interface
- Chunk size: 512 tokens, 50-token overlap

**Data Flow:**
1. PDF → PyMuPDF → block-level extraction
2. Blocks → table detection algorithm → text/table classification
3. Content → VectorStore.add_documents → ChromaDB
4. Query → question type analysis → search strategy
5. ChromaDB → semantic search → top-k documents
6. Documents → answer strategy (table/comparison/numerical/general)
7. Strategy → template-based response → answer

**Table Detection Logic:**
- Checks span alignment across lines (±5px tolerance)
- Requires consistent column structure
- Minimum 2 rows, 2 columns
- Adjacent table merging based on vertical distance

**Answer Strategies:**
- Table: extracts first/last N rows, formats markdown
- Comparison: combines text summaries with table data
- Numerical: regex extraction of numbers with context
- General: keyword-based filtering + text summarization

**Key Characteristics:**
- No external LLM (template-based generation)
- Turkish stopwords filtering
- Context-aware chunking
- Question classification system

### BetterTextHandling
Optimized variant focusing on text chunking quality.

**Components:**
- PDF Processing: PyMuPDF with advanced chunking
- Embeddings: dbmdz/bert-base-turkish-cased
- Vector Store: ChromaDB
- NLP: Transformers fill-mask pipeline
- Chunk size: 300 tokens, 100-token overlap

**Data Flow:**
Same as V3 but with enhanced text preprocessing:
1. Structural splitting (paragraphs)
2. Sentence-boundary aware chunking
3. Overlap preservation for context
4. Word-level fallback for oversized sentences

**Chunking Algorithm:**
```
sections = split_by_paragraphs(text)
for section in sections:
    if len(section) < chunk_size:
        add_to_chunks(section)
    else:
        sentences = split_by_sentence_boundaries(section)
        for sentence in sentences:
            if fits_in_current_chunk(sentence):
                add_to_chunk(sentence)
            else:
                save_chunk_with_overlap()
                start_new_chunk(sentence)
```

**Key Characteristics:**
- Smaller chunks (300 vs 512)
- Larger overlap (100 vs 50)
- Sentence-boundary preservation
- Simplified answer generation

## Core Architecture Patterns

### Document Processing Pipeline

**V2 Flow:**
```
PDF → Docling → [text, table, image] → markdown tags → chunks
```

**V3/BetterTextHandling Flow:**
```
PDF → PyMuPDF blocks → table detection → [text, table] → chunks
```

### Embedding Strategy

All versions use SentenceTransformers but differ in models:
- V2: all-MiniLM-L6-v2 (multilingual, 384-dim)
- V3/Better: bert-base-turkish-cased (Turkish-specific, 768-dim)

### Vector Search

**V2:**
- Library: FAISS
- Index: IndexFlatL2 (L2 distance)
- Search: k=3-4 nearest neighbors

**V3/BetterTextHandling:**
- Library: ChromaDB
- Index: HNSW (cosine similarity)
- Search: k=3-5 text-based query
- Persistent storage

### Answer Generation

**V2 (LLM-based):**
```
context = search_results
prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
answer = ollama.generate(prompt)
```

**V3/BetterTextHandling (Template-based):**
```
question_type = analyze_question(question)
contents = classify_contents(search_results)
answer = strategy_map[question_type](contents, question)
```

## Technical Stack

### Dependencies

**Core:**
- torch (2.1.0+)
- transformers (4.37.0+)
- sentence-transformers (2.3.0+)
- numpy (1.26.0+)
- pandas (2.1.0+)

**Vector Stores:**
- faiss-cpu (1.7.4+) - V2 only
- chromadb - V3/BetterTextHandling only

**PDF Processing:**
- pypdf (4.0.0+) - V2
- PyMuPDF (fitz) - V3/BetterTextHandling
- pdfplumber - V2
- docling - V2

**LLM:**
- ollama (0.1.0+) - V2 only

**UI:**
- streamlit - V3 only

**Optional:**
- langchain (0.1.0+)
- python-docx (0.8.11+)

### Resource Requirements

**V2:**
- Model size: ~100MB (MiniLM)
- Memory: ~2-4GB
- Requires: Ollama service running

**V3:**
- Model size: ~500MB (BERT-base)
- Memory: ~4-8GB
- GPU optional (CUDA support)

**BetterTextHandling:**
- Model size: ~500MB
- Memory: ~3-6GB (smaller chunks)
- GPU optional

## Deployment Considerations

### V2
Best for: Production environments with LLM infrastructure
- External LLM dependency
- Language-agnostic
- Better answer quality
- Requires Ollama setup

### V3
Best for: Turkish-language applications without LLM access
- Self-contained (no external services)
- Streamlit UI included
- Rule-based generation
- Lower operational cost

### BetterTextHandling
Best for: Memory-constrained environments
- Optimized chunking
- Reduced memory footprint
- Turkish-specific
- No UI

## Configuration

### V2 (config.yaml)
```yaml
ollama:
  model: "phi4"
  host: "http://localhost:11434"
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
chunk_size: 500
max_tokens: 2048
language: "tr"
```

### V3/BetterTextHandling (code-based)
- Model: hardcoded "dbmdz/bert-base-turkish-cased"
- Chunk size: 512 (V3) / 300 (Better)
- Overlap: 50 (V3) / 100 (Better)
- Device: auto-detect CUDA/CPU
- Cache: ~/.cache/huggingface

## Performance Characteristics

### Accuracy
- V2: Highest (LLM-based generation)
- V3: Moderate (template-based, question-type aware)
- BetterTextHandling: Moderate (simplified templates)

### Speed
- V2: Slowest (LLM inference)
- V3: Fast (no LLM, complex logic)
- BetterTextHandling: Fastest (no LLM, simple logic)

### Resource Usage
- V2: Medium (small embeddings, external LLM)
- V3: High (large embeddings, complex processing)
- BetterTextHandling: Medium (large embeddings, simple processing)

## Data Flow Comparison

```
┌─────────────────────────────────────────────────────┐
│                    PDF Document                      │
└────────────┬────────────────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼────┐      ┌────▼─────┐
│   V2   │      │ V3/Better│
│Docling │      │ PyMuPDF  │
└───┬────┘      └────┬─────┘
    │                │
    │ markdown       │ blocks + table detection
    │                │
┌───▼────┐      ┌────▼─────┐
│ Chunks │      │ Chunks   │
│  500   │      │ 300-512  │
└───┬────┘      └────┬─────┘
    │                │
┌───▼────┐      ┌────▼─────┐
│MiniLM  │      │BERT-TR   │
│ 384-d  │      │ 768-d    │
└───┬────┘      └────┬─────┘
    │                │
┌───▼────┐      ┌────▼─────┐
│ FAISS  │      │ChromaDB  │
│  L2    │      │ Cosine   │
└───┬────┘      └────┬─────┘
    │                │
    │ Query          │ Query + Type Analysis
    │                │
┌───▼────┐      ┌────▼─────┐
│Retrieve│      │Retrieve  │
│ k=3-4  │      │ k=3-5    │
└───┬────┘      └────┬─────┘
    │                │
┌───▼────┐      ┌────▼─────┐
│Ollama  │      │Template  │
│ Phi4   │      │Strategy  │
└───┬────┘      └────┬─────┘
    │                │
┌───▼────────────────▼─────┐
│        Answer            │
└──────────────────────────┘
```

## Limitations

### All Versions
- PDF-only input (no Word/Excel in production code)
- No authentication/authorization
- No multi-user support
- In-memory session state

### V2
- Requires Ollama service availability
- No Turkish language optimization
- No question type handling

### V3/BetterTextHandling
- No generative LLM (template-only responses)
- Turkish language only
- Limited answer flexibility
- No image processing beyond detection

## Security Considerations

- All versions run locally (on-premise)
- No data sent to external APIs (except V2 to local Ollama)
- Temporary file cleanup after processing
- No persistent user data storage