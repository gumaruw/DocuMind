# DocuMind | RAG-Based Document Analysis System

## About

A RAG-based document analysis system for financial document processing, built during an internship at ÜNLÜ & Co. Three implementation variants explore different trade-offs between answer quality, resource usage, and operational simplicity.

**Status:** This was a demo / proof-of-concept built during an internship — it was not deployed to production or used on live company data.

## Key Features

### Document Processing
- **Multi-format Support:** PDF processing with text, table, and image extraction
- **Intelligent Table Detection:** Automated identification and structured extraction of tabular data
- **Advanced Chunking:** Context-aware text segmentation with configurable overlap
- **Turkish Language Optimization:** Native support for Turkish documents (V3/BetterTextHandling)

### Retrieval System
- **Vector Search:** FAISS (V2) or ChromaDB (V3/Better) for semantic similarity
- **Multiple Embedding Models:** Language-agnostic (V2) or Turkish-specific (V3/BetterTextHandling)
- **Persistent Storage:** ChromaDB persistence for indexed documents (V3/BetterTextHandling)
- **Batch Processing:** Efficient handling of multiple documents

### Answer Generation
- **LLM Integration:** Ollama/Phi4 for natural language generation (V2)
- **Template-based Strategies:** Question-type routing for structured responses (V3/BetterTextHandling)
- **Query Classification:** Automatic detection of table, comparison, numerical, and general questions
- **Context-aware Responses:** Relevant document excerpts with source attribution

### User Interface
- **Web Interface:** Streamlit-based UI with document upload and chat (V3)
- **CLI Interface:** Command-line operation for batch processing (V2/BetterTextHandling)
- **Session Management:** Conversation history and document state tracking (V3)

---

## Tech Stack

### Core Framework
- **Python 3.9+**
- **PyTorch 2.1+:** deep learning framework for model inference
- **Transformers 4.37+:** HuggingFace library for NLP models
- **Sentence-Transformers 2.3+:** text embedding generation

### Embedding Models
- **V2:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, multilingual)
- **V3/BetterTextHandling:** `dbmdz/bert-base-turkish-cased` (768-dim, Turkish-optimized)

### Vector Databases
- **FAISS 1.7.4+** (V2): IndexFlatL2 for L2 distance
- **ChromaDB 0.4+** (V3/BetterTextHandling): HNSW indexing, cosine similarity

### LLM Integration
- **Ollama 0.1+** with **Phi4** (V2 only) — local LLM inference server

### PDF Processing
- **V2:** pdfplumber (text/table extraction), Docling (document structure)
- **V3/BetterTextHandling:** PyMuPDF (fitz), with custom table detection algorithms

### Web Framework
- **Streamlit 1.28+** (V3 UI)

### Data Processing
- **NumPy 1.26+**, **Pandas 2.1+**

---

## Implementation Variants

### V2 (LocalRagV2) — LLM-Based

Uses an external LLM (via Ollama) for answer generation.

**Architecture:**
- PDF Processing: Docling + pdfplumber
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- Vector Store: FAISS (IndexFlatL2)
- LLM: Ollama (Phi4) — local inference server
- Answer Generation: context-based prompting

**Requirements:** Ollama service running locally, with the `phi4` model pulled.

### V3 (LocalRagV3) — Turkish-Optimized

Self-contained implementation with question-type routing and no external LLM dependency.

**Architecture:**
- PDF Processing: PyMuPDF with table detection
- Embeddings: dbmdz/bert-base-turkish-cased (768-dim)
- Vector Store: ChromaDB (persistent, cosine similarity)
- Answer Generation: template-based strategies (table/comparison/numerical/general)
- UI: Streamlit web interface
- Chunking: 512 tokens with 50-token overlap

### BetterTextHandling — Memory-Optimized

A variant focused on chunking quality and a smaller memory footprint.

**Architecture:**
- PDF Processing: PyMuPDF with a refined chunking strategy
- Embeddings: dbmdz/bert-base-turkish-cased (768-dim)
- Vector Store: ChromaDB (persistent, cosine similarity)
- Answer Generation: simplified template matching
- Chunking: 300 tokens with 100-token overlap (sentence-boundary aware)

---

## Quick Start

### V2 Setup
```bash
pip install -r LocalRagV2/requirements.txt

# Start Ollama service (in separate terminal)
ollama serve
ollama run phi4

cd LocalRagV2
python app.py
```

### V3 Setup
```bash
pip install -r LocalRagV3/requirements.txt
cd LocalRagV3
streamlit run app.py
```

### BetterTextHandling Setup
```bash
pip install -r BetterTextHandling/requirements.txt
cd BetterTextHandling
python main.py
```

---

## Architecture Comparison

| Feature               | V2            | V3                   | BetterTextHandling   |
| ---------------------- | ------------- | -------------------- | --------------------- |
| **LLM**                | Ollama (Phi4) | None                  | None                   |
| **Answer Generation**  | LLM-based     | Template strategies   | Simple templates       |
| **Embeddings**         | MiniLM (384d) | BERT-Turkish (768d)   | BERT-Turkish (768d)    |
| **Vector Store**       | FAISS         | ChromaDB               | ChromaDB                |
| **Chunk Size**         | 500 tokens    | 512 tokens             | 300 tokens               |
| **Overlap**            | None          | 50 tokens               | 100 tokens                |
| **Language**           | Agnostic      | Turkish                 | Turkish                    |
| **UI**                 | CLI           | Streamlit                | CLI                          |
| **External Services**  | Yes (Ollama)  | No                        | No                             |
| **Relative Accuracy*** | Highest       | Moderate                  | Moderate                        |
| **Relative Speed***    | Slowest       | Fast                       | Fastest                          |

\* Qualitative comparison between the three variants — no benchmark numbers were measured.

---

## V3 Question Type Strategies

**Table Questions:** detects table-related keywords, extracts first/last N rows, formats as markdown, highlights specific columns.

**Comparison Questions:** identifies comparison intent, combines text and table data, presents side-by-side information.

**Numerical Questions:** regex-based number extraction, context window preservation, percentage/currency handling.

**General Questions:** keyword-based filtering, text summarization, template-based formatting.

## BetterTextHandling Chunking Algorithm

```
1. Split by paragraph boundaries
2. For each paragraph:
   a. If < chunk_size: add to chunks
   b. Else: split by sentence boundaries
3. For each sentence:
   a. If fits in current chunk: append
   b. Else: save chunk with overlap, start new
4. Preserve last sentence as overlap
5. Handle oversized sentences with word-level split
```

---

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
```python
model_name = "dbmdz/bert-base-turkish-cased"
chunk_size = 512  # V3: 512, Better: 300
overlap = 50      # V3: 50, Better: 100
collection_name = "documents"
similarity_metric = "cosine"
```

---

## Limitations

### All Versions
- PDF-only input format
- No authentication/authorization
- Single-user operation
- No persistent chat history
- In-memory session state
- **No automated tests**
- **No deployment tooling** (no Docker, Kubernetes, or CI/CD configuration in this repo)
- Never deployed to production — built as an internship demo/proof-of-concept

### V2 Specific
- Requires Ollama service availability
- No Turkish language optimization
- Higher latency due to LLM inference

### V3/BetterTextHandling Specific
- No generative LLM (template responses only)
- Turkish language only
- Rule-based response generation, limited answer flexibility

---

## Development

### Project Structure
```
DocuMind/
├── LocalRagV2/
│   ├── app.py
│   ├── rag_engine.py
│   ├── pdf_processor.py
│   ├── config.yaml
│   └── requirements.txt
├── LocalRagV3/
│   ├── app.py              # Streamlit UI
│   ├── main.py              # CLI
│   ├── rag_system.py
│   ├── vector_store.py
│   ├── pdf_extractor.py
│   └── requirements.txt
├── BetterTextHandling/
│   ├── rag_system.py
│   ├── vector_store.py
│   ├── pdf_extractor.py
│   └── requirements.txt
├── LICENSE
├── Architecture.md
└── README.md
```

---

## Architecture

For a detailed technical breakdown — data flow diagrams, the table-detection algorithm, chunking logic, and per-variant resource requirements — see [Architecture.md](https://github.com/gumaruw/DocuMind/blob/main/Architecture.md).

## License

See [LICENSE](https://github.com/gumaruw/DocuMind/blob/main/LICENSE) for details.
