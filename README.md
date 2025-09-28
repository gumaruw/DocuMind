# DocuMind | Enterprise RAG System

## About
DocuMind is an enterprise-grade **Retrieval-Augmented Generation (RAG)** platform for secure financial document analysis.  
Built for **ÜNLÜ & Co**, it focuses on **scalability**, **data privacy**, and **regulatory compliance** to streamline financial research and internal knowledge retrieval.

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

## Early Demo

<img width="1040" height="265" alt="11" src="https://github.com/user-attachments/assets/54a32f0f-6a86-40f3-9faf-196b18e92c6d" />  

<img width="1916" height="923" alt="Screenshot 2025-02-20 163646" src="https://github.com/user-attachments/assets/b5187a0c-9cdc-405c-aeb3-042d43e8526e" />   

<img width="1917" height="896" alt="Screenshot 2025-02-20 163703" src="https://github.com/user-attachments/assets/24fa1e75-058d-41b2-9379-e5fc2394d9c4" />  

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
