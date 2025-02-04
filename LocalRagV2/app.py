import os
import yaml
from typing import List, Dict
from rag_engine import RAGEngine
from pdf_processor import PDFProcessor

def load_config() -> dict:
    """Yapılandırma dosyasını yükler"""
    with open('config.yaml', 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

class DocumentQA:
    def __init__(self, config: dict):
        self.config = config
        self.pdf_processor = PDFProcessor()
        self.rag_engine = RAGEngine(config)
        self.processed_docs = []

    def load_document(self, pdf_path: str) -> bool:
        """PDF dokümanını yükler ve işler"""
        try:
            processed_content = self.pdf_processor.process_document(pdf_path)
            if processed_content:
                self.processed_docs.extend(processed_content)
                # RAG engine'e dökümanları ekle
                self.rag_engine.add_documents(processed_content)
                return True
            return False
        except Exception as e:
            print(f"Döküman yükleme hatası: {e}")
            return False

    def answer_question(self, question: str) -> str:
        """Kullanıcı sorusunu cevaplar"""
        try:
            return self.rag_engine.generate_answer(question)
        except Exception as e:
            return f"Soru cevaplanırken hata oluştu: {e}"

def main():
    config = load_config()
    qa_system = DocumentQA(config)
    
    # PDF dosyasını yükle
    pdf_path = input("PDF dosyasının yolunu girin: ").strip().strip('"')  # Boşlukları ve çift tırnakları temizle
    if not qa_system.load_document(pdf_path):
        print("PDF yüklenemedi!")
        return

    print("\nDöküman yüklendi. Sorularınızı sorabilirsiniz.")
    print("Çıkmak için 'exit' yazın.\n")

    while True:
        question = input("\nSorunuz: ")
        if question.lower() == 'exit':
            break
        
        answer = qa_system.answer_question(question)
        print("\nCevap:", answer)

if __name__ == "__main__":
    main()
