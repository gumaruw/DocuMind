"""
Metotlar:
- __init__(self, model_name: str): RAGSystem sınıfının yapıcı metodu, model ve tokenizer yükler.
- load_document(self, pdf_path: str) -> bool: PDF dokümanını yükler ve işler.
- answer_question(self, question: str) -> str: Soruyu yanıtlar.
- _analyze_question(self, question: str) -> str: Soru tipini belirler.
- _classify_contents(self, docs: List[str]) -> Dict[str, List[str]]: İçerikleri sınıflandırır.
- _create_table_answer(self, contents: Dict[str, List[str]], question: str) -> str: Tablo yanıtı oluşturur.
- _extract_specific_table_info(self, table: str, question: str) -> str: Spesifik tablo bilgisini çıkarır.
- _extract_keywords(self, text: str) -> List[str]: Metinden anahtar kelimeleri çıkarır.
- _create_comparison_answer(self, contents: Dict[str, List[str]], question: str) -> str: Karşılaştırma yanıtı oluşturur.
- _create_numerical_answer(self, contents: Dict[str, List[str]], question: str) -> str: Sayısal yanıt oluşturur.
- _create_general_answer(self, contents: Dict[str, List[str]], question: str) -> str: Genel yanıt oluşturur.
- _create_fallback_answer(self, docs: Optional[List[str]]) -> str: Hata durumunda yedek yanıt oluşturur.
- _summarize_texts(self, texts: List[str]) -> str: Metinleri özetler.
- _extract_numerical_info(self, text: str) -> str: Metinden sayısal bilgileri çıkarır.
"""

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
        self.pdf_extractor = PDFExtractor(chunk_size=768)

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
            contents = {
            "tables": [doc for doc in relevant_docs if "| " in doc and "-" in doc],
            "texts": [doc for doc in relevant_docs if "| " not in doc or "-" not in doc]
        }
            
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
            # return "Üzgünüm, bu soru için dokümanlarda ilgili bir bilgi bulunamadı."

    def _analyze_question(self, question: str) -> str:
        """Soru tipini belirler"""
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in ["tablo", "liste", "çizelge", "sıralama", "sırala", 
                     "göster", "ilk", "son", "en az", "en çok", "en büyük", "en küçük"]):
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
            return "İlgili tablo bilgisi bulunamadı."
        
        try:
            # En ilgili tabloyu seç
            most_relevant_table = contents["tables"][0]
            
            # Tabloyu satırlara böl
            table_lines = most_relevant_table.split('\n')
            if len(table_lines) < 3:  # Başlık + ayraç + en az bir veri satırı
                return "Tablo verisi okunamadı."
            
            # Başlıkları al
            headers = [h.strip() for h in table_lines[0].split('|')[1:-1]]
            
            # Veri satırlarını al (ayraç satırını atla)
            data_rows = []
            for row in table_lines[2:]:
                cells = [cell.strip() for cell in row.split('|')[1:-1]]
                if len(cells) == len(headers):
                    data_rows.append(dict(zip(headers, cells)))
            
            # Soru analizi
            question_lower = question.lower()
            
            # İlk/son N satır sorgusu
            if any(word in question_lower for word in ["ilk", "son"]):
                n = 3  # Varsayılan olarak 3 satır
                for number_word in ["bir", "iki", "üç", "dört", "beş", "1", "2", "3", "4", "5"]:
                    if number_word in question_lower:
                        n = {"bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5,
                            "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}[number_word]
                        break
                
                # İlk veya son N satırı al
                target_rows = data_rows[:n] if "ilk" in question_lower else data_rows[-n:]
                
                # Belirli bir sütun sorgulanıyorsa
                column = None
                for header in headers:
                    if header.lower() in question_lower:
                        column = header
                        break
                
                response = f"Tablodaki {'ilk' if 'ilk' in question_lower else 'son'} {n} satır "
                response += f"için {column} değerleri:\n" if column else "bilgileri:\n"
                
                for i, row in enumerate(target_rows, 1):
                    if column:
                        response += f"{i}. {row[column]}\n"
                    else:
                        response += f"{i}. satır: " + " | ".join(f"{k}: {v}" for k, v in row.items()) + "\n"
                
                return response
            
            # Genel tablo gösterimi
            return most_relevant_table
            
        except Exception as e:
            print(f"Tablo yanıt hatası: {str(e)}")
            return "Tablo bilgisi işlenirken bir hata oluştu."
    
    def _extract_specific_table_info(self, table: str, question: str) -> str:
        # Tabloyu satırlara böl
        rows = table.split('\n')
        if len(rows) < 3:  # Başlık + ayraç + en az bir veri satırı
            return ""
    
        # Başlıkları al
        headers = [h.strip() for h in rows[0].split('|')[1:-1]]
    
        # Soru içindeki anahtar kelimeleri bul
        keywords = self._extract_keywords(question)
        
        # İlgili sütunları ve satırları bul
        relevant_info = []
        for i, header in enumerate(headers):
            if any(keyword.lower() in header.lower() for keyword in keywords):
                # İlgili sütundaki değerleri topla
                values = []
                for row in rows[2:]:  # İlk iki satır başlık ve ayraç
                    cells = [cell.strip() for cell in row.split('|')[1:-1]]
                    if i < len(cells):
                        values.append(f"{header}: {cells[i]}")
                relevant_info.extend(values)
        
        return " | ".join(relevant_info) if relevant_info else ""

    def _extract_keywords(self, text: str) -> List[str]:
        # Stopwords ve özel karakterleri temizle
        stopwords = {"ve", "veya", "ile", "bu", "şu", "o", "bir", "için"}
        words = text.lower().split()
        return [word for word in words if word not in stopwords and len(word) > 2]


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
