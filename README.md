# DocuMind | Enterprise RAG System

## About
DocuMind is an enterprise-grade **Retrieval-Augmented Generation (RAG)** platform for secure financial document analysis.  
Built for **ÜNLÜ & Co** (Jan–Apr 2025), it focuses on **scalability**, **data privacy**, and **regulatory compliance** to streamline financial research and internal knowledge retrieval.

## Key Features
- **Document Pipeline**: Custom parsing for PDFs, Word files, and tabular data.  
- **Semantic Search**: Vector database infrastructure for fast contextual retrieval.  
- **Transformer Models**: Context-aware response generation using advanced NLP.  
- **On-Prem Deployment**: Full data privacy with local hosting to meet financial regulations.  
- **Performance**:  
  - 70% Q&A accuracy  
  - 65% reduction in manual research time  
  - Processed 1,700+ internal documents  
  - Scales to 500–1,000 employees

## Model Versions
The repo provides **3 model variants** with different trade-offs in speed, accuracy, and resource usage.

## Tech Stack
- **Core**: Python, LangChain, Transformers, PyTorch  
- **Vector DB**: FAISS / Milvus (depending on version)  
- **NLP Models**: BERT, custom transformer-based RAG  
- **Hardware**: CUDA acceleration for large-scale embeddings  
- **Clients**: Ollama for local LLM integration

## Setup
Clone repo:
```bash
git clone https://github.com/gumaruw/DocuMind.git
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Run the system:
```bash
python run_app.py
```

## Usage
1. Upload documents (PDF/Word/Tables).
2. Index content into the vector database.
3. Query with natural language to get context-aware answers.
4. Explore different model versions for performance/accuracy needs.
