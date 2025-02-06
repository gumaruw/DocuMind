# vector_store.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.chroma_client = chromadb.Client(Settings(is_persistent=True))
        self.collection = self.chroma_client.get_or_create_collection(name="documents")

    def add_documents(self, documents: List[Dict]):
        """Dökümanları vektör veritabanına ekler"""
        for i, doc in enumerate(documents):
            content = doc['content']
            metadata = {
                'type': doc['type'],
                'page': str(doc['page'])
            }
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[f"doc_{i}"]
            )

    def search(self, query: str, k: int = 3) -> List[str]:
        """Sorguya en yakın dökümanları bulur"""
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        return results['documents'][0]