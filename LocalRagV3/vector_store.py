# vector_store.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import chromadb
from chromadb.config import Settings
import uuid

class VectorStore:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.chroma_client = chromadb.Client(Settings(is_persistent=True))
        # Koleksiyonu her seferinde yeniden oluştur
        try:
            self.chroma_client.delete_collection("documents")
        except:
            pass
        self.collection = self.chroma_client.create_collection(name="documents")

    def add_documents(self, documents: List[Dict]):
        """Dökümanları vektör veritabanına ekler"""
        for i, doc in enumerate(documents):
            content = doc['content']
            metadata = {
                'type': doc['type'],
                'page': str(doc['page'])
            }
            # Benzersiz ID oluştur
            unique_id = str(uuid.uuid4())
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[unique_id]
            )

    def search(self, query: str, k: int = 3) -> List[str]:
        """
        Sorguya en yakın dökümanları bulur
        """
        try:
            embedding = self.embedding_model.encode(query)
            results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=k
            )
            if results and 'documents' in results and len(results['documents']) > 0:
                return results['documents'][0]
            return ["İlgili döküman bulunamadı."]
        except Exception as e:
            print(f"Arama sırasında hata: {str(e)}")
            return ["Arama sırasında bir hata oluştu."]
