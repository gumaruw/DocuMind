import faiss
import torch
import nltk
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import yaml
from ollama import Client
from typing import List, Dict

def clean_text(text):
    """Metindeki boşlukları, satır başlarını ve diğer gereksiz karakterleri temizler."""
    text = re.sub(r'\s+', ' ', text)  # Çoklu boşlukları tek boşluğa indir
    text = text.strip()  # Baş ve sondaki boşlukları sil
    return text

def create_embeddings(text, model_name='all-MiniLM-L6-v2'):
    """Metinleri vektörlere dönüştürür."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text)
    if isinstance(embeddings, np.ndarray):
        return embeddings.astype('float32')
    return np.array(embeddings, dtype=np.float32)

def generate_response(prompt, model='phi4', host="http://localhost:11434"):
    """Ollama ile belirtilen model kullanarak cevap üretir."""
    ollama_client = Client(host=host)
    try:
        response = ollama_client.generate(model=model, prompt=prompt)
        return response['response']
    except Exception as e:
        print(f"Ollama Hata: {e}")
        return "Cevap üretilemedi"

def create_faiss_index(embeddings, dimension):
    """FAISS indexi oluşturur."""
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_faiss_index(index, query_embedding, k=4):
    """FAISS indexinde benzerlik araması yapar."""
    D, I = index.search(query_embedding.reshape(1, -1).astype('float32'), k)
    return I

class RAGEngine:
    def __init__(self, config: dict):
        self.config = config
        self.embedding_model = SentenceTransformer(config['embedding_model'])
        self.ollama_client = Client(host=config['ollama']['host'])
        self.document_chunks = []
        self.index = None
        
    def add_documents(self, documents: List[Dict]):
        """Dökümanları işler ve vector veritabanına ekler"""
        chunks = self._prepare_chunks(documents)
        self.document_chunks.extend(chunks)
        
        embeddings = self._create_embeddings(chunks)
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def generate_answer(self, question: str) -> str:
        """Soruyu cevaplar"""
        question_embedding = self._create_embeddings([question])[0]
        
        D, I = self.index.search(question_embedding.reshape(1, -1), k=3)
        relevant_chunks = [self.document_chunks[i] for i in I[0]]
        
        prompt = self._create_prompt(question, relevant_chunks)
        
        try:
            response = self.ollama_client.generate(
                model=self.config['ollama']['model'],
                prompt=prompt
            )
            return response['response']
        except Exception as e:
            return f"Cevap üretilirken hata oluştu: {e}"

    def _prepare_chunks(self, documents: List[Dict]) -> List[str]:
        """Dökümanları uygun boyutlu chunk'lara ayırır"""
        chunks = []
        for doc in documents:
            content = doc['content']
            if doc['type'] == 'text':
                sentences = content.split('.')
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    if current_length + len(sentence) > 500:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_length = len(sentence)
                    else:
                        current_chunk.append(sentence)
                        current_length += len(sentence)
                
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
            else:
                chunks.append(f"{doc['type']}: {content}")
        
        return chunks

    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Metinler için embedding vektörleri oluşturur"""
        return self.embedding_model.encode(texts, convert_to_numpy=True)

    def _create_prompt(self, question: str, relevant_chunks: List[str]) -> str:
        """Phi4 için optimize edilmiş prompt oluşturur"""
        context = "\n".join(relevant_chunks)
        return f"""Aşağıdaki bağlam bilgisini kullanarak soruyu yanıtla. 
        Sadece verilen bağlam bilgisine dayanarak cevap ver.
        Eğer cevap bağlamda yoksa, "Bu sorunun cevabı dokümanlarda bulunamadı" de.
        
        Bağlam:
        {context}
        
        Soru: {question}
        
        Cevap:"""
