# rag_system.py
from vector_store import VectorStore
from pdf_extractor import PDFExtractor
import ollama
from typing import List, Dict

class RAGSystem:
    def __init__(self, model_name: str = "phi4:14b"): # Model ismi değiştirilebilir.
        self.model_name = model_name
        self.vector_store = VectorStore()
        self.pdf_extractor = PDFExtractor()

    def load_document(self, pdf_path: str) -> bool:
        """
        PDF dokümanını yükler ve işler
        """
        try:
            contents = self.pdf_extractor.extract_content(pdf_path)
            if contents:
                self.vector_store.add_documents(contents)
                return True
            return False
        except Exception as e:
            print(f"Döküman yükleme hatası: {e}")
            return False

    def answer_question(self, question: str) -> str:
        """
        Kullanıcı sorusunu cevaplar
        """
        try:
            relevant_docs = self.vector_store.search(question)
            context = "\n".join(relevant_docs)
            
            prompt = f"""
            Aşağıdaki Türkçe bağlam bilgisini kullanarak soruyu yanıtla.
            Sadece verilen bağlam bilgisine dayanarak cevap ver.
            Eğer cevap bağlamda yoksa, "Bu sorunun cevabı dokümanlarda bulunamadı" de.

            Bağlam:
            {context}

            Soru: {question}

            Cevap:
            """
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt
            )
            return response['response']
        except Exception as e:
            return f"Soru cevaplanırken hata oluştu: {str(e)}"
