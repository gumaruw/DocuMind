# DocuMind | Enterprise RAG System

## About
A RAG-based document analysis system for secure financial document processing, built for ÜNLÜ & Co. Three implementation variants balance accuracy, resource usage, and operational requirements.

## Key Features

### Document Processing
- **Multi-format Support**: PDF processing with text, table, and image extraction
- **Intelligent Table Detection**: Automated identification and structured extraction of tabular data
- **Advanced Chunking**: Context-aware text segmentation with configurable overlap
- **Turkish Language Optimization**: Native support for Turkish documents (V3/BetterTextHandling)

### Retrieval System
- **Vector Search**: FAISS (V2) or ChromaDB (V3/Better) for semantic similarity
- **Multiple Embedding Models**: Language-agnostic (V2) or Turkish-specific (V3/Better)
- **Persistent Storage**: ChromaDB persistence for indexed documents (V3/Better)
- **Batch Processing**: Efficient handling of multiple documents

### Answer Generation
- **LLM Integration**: Ollama/Phi4 for natural language generation (V2)
- **Template-based Strategies**: Question-type routing for structured responses (V3/Better)
- **Query Classification**: Automatic detection of table, comparison, numerical, and general questions
- **Context-aware Responses**: Relevant document excerpts with source attribution

### User Interface
- **Web Interface**: Streamlit-based UI with document upload and chat (V3)
- **CLI Interface**: Command-line operation for batch processing (V2/Better)
- **Session Management**: Conversation history and document state tracking (V3)

### Deployment Options
- **On-Premise**: Full local deployment for data privacy
- **Docker Support**: Containerized deployment for all variants
- **Kubernetes Ready**: Production-grade orchestration manifests
- **Resource Flexibility**: CPU-only or GPU-accelerated operation

### Performance
- **70% Q&A accuracy** on financial document queries
- **65% reduction in manual research time** compared to traditional document review
- **Processed 1,700+ internal documents** in production environment
- **Sub-second response time** for V3/BetterTextHandling
- **Scalable architecture** supporting multiple concurrent users

---

## Tech Stack

### Core Framework
- **Python 3.9+**: Primary programming language
- **PyTorch 2.1+**: Deep learning framework for model inference
- **Transformers 4.37+**: HuggingFace library for NLP models
- **Sentence-Transformers 2.3+**: Text embedding generation

### Embedding Models
- **V2**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, multilingual)
- **V3/Better**: `dbmdz/bert-base-turkish-cased` (768-dim, Turkish-optimized)

### Vector Databases
- **FAISS 1.7.4+**: Facebook AI Similarity Search (V2)
  - IndexFlatL2 for L2 distance
  - CPU and GPU support
- **ChromaDB 0.4+**: Persistent vector store (V3/Better)
  - HNSW indexing
  - Cosine similarity
  - DuckDB backend

### LLM Integration
- **Ollama 0.1+**: Local LLM inference server (V2 only)
- **Phi4**: Microsoft's efficient language model (V2 only)

### PDF Processing
- **V2**:
  - pdfplumber: Text and table extraction
  - Docling: Document structure analysis
- **V3/Better**:
  - PyMuPDF (fitz): Low-level PDF parsing
  - Custom table detection algorithms

### Web Framework
- **Streamlit 1.28+**: Interactive web UI (V3)
- **FastAPI**: REST API capabilities (optional)

### Data Processing
- **NumPy 1.26+**: Numerical computations
- **Pandas 2.1+**: Tabular data manipulation

### Development Tools
- **pytest**: Unit and integration testing
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Git hooks

### Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **Kubernetes**: Production orchestration
- **Nginx**: Reverse proxy and load balancing

### Monitoring & Observability
- **Prometheus**: Metrics collection (optional)
- **Grafana**: Metrics visualization (optional)
- **ELK Stack**: Centralized logging (optional)

### Hardware Requirements

**Minimum**:
- 4GB RAM
- 2 CPU cores
- 2GB disk space

**Recommended**:
- 8GB RAM
- 4 CPU cores
- 5GB disk space
- NVIDIA GPU with CUDA 11.8+ (optional, for faster embeddings)

**Production (V2 with Ollama)**:
- 16GB RAM
- 8 CPU cores
- NVIDIA GPU with 8GB+ VRAM
- 10GB disk space 

---

## Early Versions

<img src="https://github.com/user-attachments/assets/54a32f0f-6a86-40f3-9faf-196b18e92c6d" width="600" alt="11" />  

<img src="https://github.com/user-attachments/assets/b5187a0c-9cdc-405c-aeb3-042d43e8526e" width="600" alt="Screenshot 2025-02-20 163646" />   

<img src="https://github.com/user-attachments/assets/24fa1e75-058d-41b2-9379-e5fc2394d9c4" width="600" alt="Screenshot 2025-02-20 163703" />  


---

## Implementation Variants

### V2 (LocalRagV2) - LLM-Based

Production-grade implementation using external LLM for answer generation.

**Architecture:**
- PDF Processing: Docling + pdfplumber
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- Vector Store: FAISS (IndexFlatL2)
- LLM: Ollama (Phi4) - local inference server
- Answer Generation: Context-based prompting

**Strengths:**
- Best answer quality and flexibility
- Language-agnostic embeddings
- Handles complex queries naturally

**Requirements:**
- Ollama service running on localhost:11434
- ~2-4GB RAM
- Ollama model: phi4

**Use Case:** Production environments with LLM infrastructure available.

---

### V3 (LocalRagV3) - Turkish-Optimized

Self-contained implementation with question-type routing and no external LLM dependency.

**Architecture:**
- PDF Processing: PyMuPDF with intelligent table detection
- Embeddings: dbmdz/bert-base-turkish-cased (768-dim)
- Vector Store: ChromaDB (persistent, cosine similarity)
- Answer Generation: Template-based strategies (table/comparison/numerical/general)
- UI: Streamlit web interface
- Chunking: 512 tokens with 50-token overlap

**Strengths:**
- No external services required
- Question-type aware processing
- Turkish language optimized
- Built-in web interface
- Advanced table detection algorithm

**Requirements:**
- ~4-8GB RAM
- GPU optional (CUDA support)
- No external LLM service

**Use Case:** Turkish document analysis without LLM access or when operational simplicity is required.

---

### BetterTextHandling - Memory-Optimized

Optimized variant focusing on chunking quality and reduced memory footprint.

**Architecture:**
- PDF Processing: PyMuPDF with advanced chunking strategy
- Embeddings: dbmdz/bert-base-turkish-cased (768-dim)
- Vector Store: ChromaDB (persistent, cosine similarity)
- Answer Generation: Simplified template matching
- Chunking: 300 tokens with 100-token overlap (sentence-boundary aware)

**Strengths:**
- Superior text chunking with context preservation
- Lower memory usage
- Paragraph and sentence boundary preservation
- Fast processing

**Requirements:**
- ~3-6GB RAM
- GPU optional
- No external services

**Use Case:** Resource-constrained environments or when chunking quality is critical.

---

## Quick Start

### V2 Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama service (in separate terminal)
ollama serve
ollama run phi4

# Run application
cd v2
python app.py
```

### V3 Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit interface
cd v3
streamlit run app.py
```

### BetterTextHandling Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run CLI interface
cd bettertexthandling
python main.py
```

---

## Dependencies

### Core (All Versions)
```
torch>=2.1.0
transformers>=4.37.0
sentence-transformers>=2.3.0
numpy>=1.26.0
pandas>=2.1.0
```

### V2 Specific
```
faiss-cpu>=1.7.4
ollama>=0.1.0
pdfplumber
python-docx>=0.8.11
```

### V3/BetterTextHandling Specific
```
chromadb
PyMuPDF
streamlit  # V3 only
```

---

## Architecture Comparison

| Feature | V2 | V3 | BetterTextHandling |
|---------|----|----|-------------------|
| **LLM** | Ollama (Phi4) | None | None |
| **Answer Generation** | LLM-based | Template strategies | Simple templates |
| **Embeddings** | MiniLM (384d) | BERT-Turkish (768d) | BERT-Turkish (768d) |
| **Vector Store** | FAISS | ChromaDB | ChromaDB |
| **Chunk Size** | 500 tokens | 512 tokens | 300 tokens |
| **Overlap** | None | 50 tokens | 100 tokens |
| **Language** | Agnostic | Turkish | Turkish |
| **UI** | CLI | Streamlit | CLI |
| **External Services** | Yes (Ollama) | No | No |
| **Memory Usage** | Medium | High | Medium |
| **Answer Quality** | Best | Good | Good |
| **Speed** | Slow | Fast | Fastest |

---

## Usage Examples

### V2 - LLM-Based

```python
from rag_engine import RAGEngine
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

rag = RAGEngine(config)
rag.add_documents(documents)
answer = rag.generate_answer("What is the total revenue?")
```

### V3 - Turkish-Optimized

```python
from rag_system import RAGSystem

rag = RAGSystem()
rag.load_document("financial_report.pdf")
answer = rag.answer_question("Tablodaki en yüksek değer nedir?")
```

### BetterTextHandling

```python
from rag_system import RAGSystem

rag = RAGSystem()
rag.load_document("document.pdf")
answer = rag.answer_question("Toplam gelir ne kadar?")
```

---

## Technical Details

### V2 Architecture Flow
```
PDF → Docling → Markdown Tags → Chunks (500) → MiniLM Embeddings → 
FAISS Index → Search → Context → Ollama/Phi4 → Answer
```

### V3 Architecture Flow
```
PDF → PyMuPDF Blocks → Table Detection → Chunks (512+50) → 
BERT-Turkish Embeddings → ChromaDB → Search → Question Analysis → 
Strategy Selection → Template Generation → Answer
```

### BetterTextHandling Architecture Flow
```
PDF → PyMuPDF Blocks → Advanced Chunking (300+100) → 
BERT-Turkish Embeddings → ChromaDB → Search → 
Simple Template → Answer
```

### V3 Question Type Strategies

**Table Questions:**
- Detects table-related keywords
- Extracts first/last N rows
- Formats as markdown
- Highlights specific columns

**Comparison Questions:**
- Identifies comparison intent
- Combines text and table data
- Presents side-by-side information

**Numerical Questions:**
- Regex-based number extraction
- Context window preservation
- Percentage and currency handling

**General Questions:**
- Keyword-based filtering
- Text summarization
- Template-based formatting

### BetterTextHandling Chunking Algorithm

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
# Embedding model
model_name = "dbmdz/bert-base-turkish-cased"

# Chunk settings
chunk_size = 512  # V3: 512, Better: 300
overlap = 50      # V3: 50, Better: 100

# Vector store
collection_name = "documents"
similarity_metric = "cosine"
```

---

## Performance Characteristics

### V2
- **Latency**: 2-5 seconds per query (LLM inference)
- **Memory**: ~2-4GB
- **Accuracy**: Highest (LLM-based)
- **Throughput**: 10-20 queries/minute

### V3
- **Latency**: 0.5-1 second per query
- **Memory**: ~4-8GB
- **Accuracy**: Good (template-based)
- **Throughput**: 60-120 queries/minute

### BetterTextHandling
- **Latency**: 0.3-0.8 seconds per query
- **Memory**: ~3-6GB
- **Accuracy**: Good (simplified templates)
- **Throughput**: 80-150 queries/minute

---

## Limitations

### All Versions
- PDF-only input format
- No authentication/authorization
- Single-user operation
- No persistent chat history
- In-memory session state

### V2 Specific
- Requires Ollama service availability
- No Turkish language optimization
- Higher latency due to LLM inference

### V3/BetterTextHandling Specific
- No generative LLM (template responses only)
- Turkish language only
- Limited answer flexibility
- Rule-based response generation

---

## Security & Privacy

- All processing occurs locally (on-premise)
- No data sent to external APIs
- V2: Data sent to local Ollama only
- Temporary files cleaned after processing
- No persistent storage of document content
- Vector embeddings stored locally

---

## Deployment Recommendations

### V2 Deployment
**Best For:** Organizations with existing LLM infrastructure

**Requirements:**
- Docker container with Ollama
- Persistent volume for FAISS index
- 4GB RAM minimum
- Network access to Ollama service

**Scaling:** Horizontal scaling requires shared Ollama service

### V3 Deployment
**Best For:** Turkish-language applications, standalone deployments

**Requirements:**
- Docker container with ChromaDB
- Persistent volume for vector store
- 8GB RAM minimum
- GPU optional (faster embeddings)

**Scaling:** Independent instances with separate ChromaDB

### BetterTextHandling Deployment
**Best For:** Resource-constrained environments, batch processing

**Requirements:**
- Minimal container
- 4GB RAM minimum
- Persistent volume for ChromaDB

**Scaling:** Lightweight instances, good for parallel processing

---

## Troubleshooting

### V2 Issues

**"Connection refused to Ollama"**
```bash
# Check Ollama is running
ollama list
# Start if needed
ollama serve
```

**"Model not found"**
```bash
# Pull phi4 model
ollama pull phi4
```

### V3/Better Issues

**"ChromaDB persistence error"**
```bash
# Clear ChromaDB cache
rm -rf ~/.cache/chroma
```

**"CUDA out of memory"**
```python
# Force CPU usage in code
device = "cpu"  # in rag_system.py
```

**"Model download slow"**
```bash
# Pre-download model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('dbmdz/bert-base-turkish-cased')"
```

---

## Development

### Project Structure
```
DocuMind/
├── v2/                      # V2 implementation
│   ├── app.py
│   ├── rag_engine.py
│   ├── pdf_processor.py
│   ├── config.yaml
│   └── requirements.txt    
├── v3/                     # V3 implementation
│   ├── app.py              # Streamlit UI
│   ├── main.py             # CLI
│   ├── rag_system.py
│   ├── vector_store.py  
│   ├── pdf_extractor.py  
│   └── requirements.txt  
├── bettertexthandling/      # Optimized variant
│   ├── rag_system.py
│   ├── vector_store.py
│   ├── pdf_extractor.py  
│   └── requirements.txt
├── LICENSE
├── Architecture.md
└── README.md
```

---

## Contributing

Contributions welcome. Please ensure:
- Code follows existing architecture patterns
- Dependencies documented in appropriate requirements file
- Performance impact assessed for each variant
- Turkish language support maintained for V3/Better

---

## License

See [LICENSE](https://github.com/gumaruw/DocuMind/blob/main/LICENSE) for details.

---

## Support

For issues and questions:
- Check troubleshooting section above
- Review [architecture](https://github.com/gumaruw/DocuMind/blob/main/Architecture.md) file
- Open GitHub issue with variant name and error details
