import re
from vector_store import VectorStore
from pdf_extractor import PDFExtractor
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
import torch
from typing import List, Dict, Optional
import os
import gc

class RAGSystem:
    def __init__(self, model_name: str = "dbmdz/bert-base-turkish-cased"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Cihaz: {self.device}")
        
        # Bellek optimizasyonu
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Cache dizinini ayarla
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Model ve tokenizer yükleme
        try:
            print("Tokenizer yükleniyor...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=False
            )
            
            print("Model yükleniyor...")
            model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # Fill-mask pipeline oluştur
            self.nlp = pipeline(
                "fill-mask",
                model=model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            print("Model yüklendi!")
            
        except Exception as e:
            print(f"Model yükleme hatası: {str(e)}")
            print("Alternatif model deneniyor...")
            try:
                # Alternatif model
                model_name = "yavuzKomecoglu/electra-base-turkish-cased-discriminator"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                model = AutoModelForMaskedLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                self.nlp = pipeline(
                    "fill-mask",
                    model=model,
                    tokenizer=self.tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
                print("Alternatif model başarıyla yüklendi!")
            except Exception as e2:
                print(f"Alternatif model yüklemesi de başarısız: {str(e2)}")
                raise

        self.vector_store = VectorStore(model_name='dbmdz/bert-base-turkish-cased')
        self.pdf_extractor = PDFExtractor(chunk_size=300)

    def load_document(self, pdf_path: str) -> bool:
        """PDF dokümanını yükler ve işler"""
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
        """Soruyu yanıtlar"""
        try:
            # Sorguyu analiz et
            question_type = self._analyze_question(question)
            relevant_docs = self.vector_store.search(question, k=5)
            
            if not relevant_docs:
                return "Üzgünüm, bu soru için dokümanlarda ilgili bir bilgi bulunamadı."
            
            # İçerik sınıflandırma
            contents = self._classify_contents(relevant_docs)
            
            # Soru tipine göre yanıt oluştur
            if question_type == "table":
                return self._create_table_answer(contents, question)
            elif question_type == "comparison":
                return self._create_comparison_answer(contents, question)
            elif question_type == "numerical":
                return self._create_numerical_answer(contents, question)
            else:
                return self._create_general_answer(contents, question)

        except Exception as e:
            print(f"Hata: {str(e)}")
            return self._create_fallback_answer(relevant_docs if 'relevant_docs' in locals() else None)

    def _analyze_question(self, question: str) -> str:
        """Soru tipini belirler"""
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in ["tablo", "liste", "çizelge"]):
            return "table"
        elif any(keyword in question_lower for keyword in ["karşılaştır", "kıyasla", "fark", "arasında"]):
            return "comparison"
        elif any(keyword in question_lower for keyword in ["toplam", "ortalama", "yüzde", "oran", "sayı", "miktar"]):
            return "numerical"
        return "general"

    def _classify_contents(self, docs: List[str]) -> Dict[str, List[str]]:
        """İçerikleri sınıflandırır"""
        return {
            "tables": [doc for doc in docs if "| " in doc and "-" in doc],
            "texts": [doc for doc in docs if "| " not in doc or "-" not in doc]
        }

    def _create_table_answer(self, contents: Dict[str, List[str]], question: str) -> str:
        """Tablo yanıtı oluşturur"""
        if not contents["tables"]:
            return "Dokümanda ilgili tablo bulunamadı."
        
        response = "Tablodaki bilgilere göre:\n\n"
        response += contents["tables"][0]
        
        if len(contents["tables"]) > 1:
            response += "\n\nNot: Dokümanda başka ilgili tablolar da bulunmaktadır."
            
        return response

    def _create_comparison_answer(self, contents: Dict[str, List[str]], question: str) -> str:
        """Karşılaştırma yanıtı oluşturur"""
        response_parts = []
        
        if contents["texts"]:
            text_info = self._summarize_texts(contents["texts"][:2])
            response_parts.append(f"Metin bilgilerine göre:\n{text_info}")
        
        if contents["tables"]:
            response_parts.append(f"Tablo verilerine göre:\n{contents['tables'][0]}")
        
        return "\n\n".join(response_parts) if response_parts else "Karşılaştırma için yeterli bilgi bulunamadı."

    def _create_numerical_answer(self, contents: Dict[str, List[str]], question: str) -> str:
        """Sayısal yanıt oluşturur"""
        response_parts = []
        
        if contents["tables"]:
            response_parts.append(f"Sayısal veriler (tablodan):\n{contents['tables'][0]}")
        
        if contents["texts"]:
            numerical_info = self._extract_numerical_info(contents["texts"][0])
            if numerical_info:
                response_parts.append(f"Metin içindeki sayısal bilgiler:\n{numerical_info}")
        
        return "\n\n".join(response_parts) if response_parts else "Sayısal veri bulunamadı."

    def _create_general_answer(self, contents: Dict[str, List[str]], question: str) -> str:
        """Genel yanıt oluşturur"""
        response_parts = []
        
        if contents["texts"]:
            response_parts.append(self._summarize_texts(contents["texts"][:2]))
        
        if contents["tables"]:
            response_parts.append("\nİlgili tablo verisi:\n" + contents["tables"][0])
        
        return "\n\n".join(response_parts) if response_parts else "İlgili bilgi bulunamadı."

    def _create_fallback_answer(self, docs: Optional[List[str]]) -> str:
        """Hata durumunda yedek yanıt oluşturur"""
        if not docs:
            return "Üzgünüm, bir hata oluştu ve ilgili bilgi bulunamadı."
        return f"En ilgili bilgi:\n\n{docs[0]}"

    def _summarize_texts(self, texts: List[str]) -> str:
        """Metinleri özetler"""
        return "\n".join(text for text in texts)

    def _extract_numerical_info(self, text: str) -> str:
        """Metinden sayısal bilgileri çıkarır"""
        number_pattern = r'\b\d+(?:[.,]\d+)?(?:\s*(?:%|percent|yüzde|milyon|bin|TL))?\b'
        matches = re.finditer(number_pattern, text)
        
        found_numbers = []
        for match in matches:
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].strip()
            found_numbers.append(context)
        
        return "\n".join(found_numbers) if found_numbers else ""
