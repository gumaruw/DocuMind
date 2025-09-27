from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import chromadb
from chromadb.config import Settings
import uuid

class VectorStore:
    def __init__(self, model_name: str = 'dbmdz/bert-base-turkish-cased'):
        # Aynı modeli kullan
        self.embedding_model = SentenceTransformer(model_name, device="cpu")  # CPU'da çalıştır
        self.chroma_client = chromadb.Client(Settings(
            is_persistent=True,
            anonymized_telemetry=False
        ))
        try:
            self.chroma_client.delete_collection("documents")
        except:
            pass
        self.collection = self.chroma_client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}  # Cosine similarity kullan
        )

    def add_documents(self, documents: List[Dict]):
        """Dökümanları vektör veritabanına ekler"""
        batch_size = 5  # Bellek için batch işleme
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            contents = [doc['content'] for doc in batch]
            metadatas = [{'type': doc['type'], 'page': str(doc['page'])} for doc in batch]
            ids = [str(uuid.uuid4()) for _ in batch]
            
            self.collection.add(
                documents=contents,
                metadatas=metadatas,
                ids=ids
            )

    def search(self, query: str, k: int = 3) -> List[str]:
        """
        Sorguya en yakın dökümanları bulur
        """
        try:
            # Direkt metin tabanlı arama kullan
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            if results and 'documents' in results and len(results['documents']) > 0:
                return results['documents'][0]
            return ["İlgili döküman bulunamadı."]
        except Exception as e:
            print(f"Arama sırasında hata: {str(e)}")
            return ["Arama sırasında bir hata oluştu."]
