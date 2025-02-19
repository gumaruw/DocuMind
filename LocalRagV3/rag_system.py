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

from vector_store import VectorStore
from pdf_extractor import PDFExtractor
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
import torch
from typing import List, Dict, Optional
import os
import gc
import re

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
        
        # Soru analizi için anahtar kelimeler
        self.question_keywords = {
        "table": [
            "tablo", "liste", "çizelge", "sırala", "göster", "kaç", "hangi", 
            "ne kadar", "en çok", "en az", "miktar", "oran", "değer", "veri"
        ],
        "comparison": [
            "karşılaştır", "kıyasla", "fark", "arasında", "benzer", "farklı", 
            "daha", "göre", "avantaj", "dezavantaj", "artı", "eksi", "olumlu", 
            "olumsuz", "güçlü", "zayıf"
        ],
        "numerical": [
            "toplam", "ortalama", "yüzde", "oran", "sayı", "miktar", "kaç", 
            "ne kadar", "tutar", "değer", "adet", "rakam"
        ],
        "location": [
            "nerede", "hangi", "konum", "yer", "pozisyon", "alan", "bölge", 
            "kısım", "nokta"
        ],
        "temporal": [
            "ne zaman", "tarih", "süre", "zaman", "dönem", "ay", "yıl", "gün",
            "periyot", "dönemsel", "süreç"
        ]
        }
        
        # Çıktı formatları
        self.output_templates = {
            "table": "Tabloda gösterilen bilgilere göre:\n{}\n\nÖnemli noktalar:\n{}",
            "comparison": "Karşılaştırma sonucu:\n\n{}\n\nÖnemli farklar:\n{}",
            "numerical": "Sayısal analiz:\n\n{}\n\nÖzet:\n{}",
            "location": "Konum bilgisi:\n\n{}",
            "temporal": "Zaman bilgisi:\n\n{}",
            "general": "{}\n\nÖzet:\n{}"
        }
        
        # Türkçe stop words
        self.stopwords = {
            "ve", "veya", "ile", "bu", "şu", "o", "bir", "için", "gibi", "de", "da",
            "ki", "mi", "ne", "ya", "hem", "ama", "fakat", "ancak", "lakin", "ise",
            "değil", "olan", "üzere", "yoksa", "oysa", "yani", "nasıl", "neden", 
            "çünkü", "ise", "ama", "fakat", "lakin", "ancak", "yalnız", "oysa",
            "oysaki", "halbu", "halbuki", "gelince", "değin", "dair", "göre", "kadar",
            "karşın", "rağmen", "dolayı", "diye", "üzere", "içinde", "hakkında",
            "tarafından", "nedeniyle", "sebebiyle", "vasıtasıyla", "açısından"
        }

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
            
            # İçerikleri sınıflandır
            contents = self._classify_contents(relevant_docs)
            
            # Sorunun ana konusunu çıkar
            main_topic = self._extract_main_topic(question)
            
            # Ana konuya göre filtrelenmiş içerik oluştur
            filtered_contents = self._filter_by_topic(contents, main_topic)
            
            # Soru tipine göre yanıt oluştur
            if filtered_contents:
                if question_type == "table":
                    response = self._create_table_answer(filtered_contents, question)
                elif question_type == "comparison":
                    response = self._create_comparison_answer(filtered_contents, question)
                elif question_type == "numerical":
                    response = self._create_numerical_answer(filtered_contents, question)
                else:
                    response = self._create_general_answer(filtered_contents, question)
            else:
                # Filtrelenmiş içerik boşsa tüm içeriği kullan
                response = self._create_general_answer(contents, question)

            # Yanıtı kontrol et
            if not response or len(response.strip()) < 20:  # Minimum yanıt uzunluğu
                return "Üzgünüm, bu soru için yeterli bilgi bulunamadı."

            return response
                
        except Exception as e:
            print(f"Hata: {str(e)}")
            return self._create_fallback_answer(relevant_docs if 'relevant_docs' in locals() else None)
        
    def _extract_main_topic(self, question: str) -> str:
        """Sorudan ana konuyu çıkarır"""
        # Soru kelimelerini temizle
        question_words = [
            "nedir", "nasıl", "ne", "hangi", "kaç", "neden", "niçin", 
            "kim", "kime", "nerede", "ne zaman"
        ]
        
        words = question.lower().split()
        # Soru kelimelerini ve stop words'leri çıkar
        main_words = [
            word for word in words 
            if word not in question_words and word not in self.stopwords
        ]
        
        return " ".join(main_words)

    def _filter_by_topic(self, contents: Dict[str, List[str]], topic: str) -> Dict[str, List[str]]:
        """İçerikleri ana konuya göre filtreler"""
        topic_words = set(self._extract_keywords(topic))
        
        filtered = {
            "texts": [],
            "tables": []
        }
        
        # Metinleri filtrele
        for text in contents["texts"]:
            text_words = set(self._extract_keywords(text))
            if topic_words.intersection(text_words):
                filtered["texts"].append(text)
        
        # Tabloları filtrele
        for table in contents["tables"]:
            table_words = set(self._extract_keywords(table))
            if topic_words.intersection(table_words):
                filtered["tables"].append(table)
        
        return filtered
    
    def _analyze_question(self, question: str) -> str:
        """
        Sorunun tiplerini belirler (birden fazla tip olabilir)
        """
        question_lower = question.lower()
        question_types = []
        
        for q_type, keywords in self.question_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                question_types.append(q_type)
                
        return question_types[0] if question_types else "general"

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
        """Metinden anahtar kelimeleri çıkarır"""
        # Türkçe stop words
        stopwords = {
            "ve", "veya", "ile", "bu", "şu", "o", "bir", "için", "gibi", "de", "da",
            "ki", "mi", "ne", "ya", "hem", "ama", "fakat", "ancak", "lakin", "ise",
            "değil", "olan", "üzere", "yoksa", "oysa", "yani", "nasıl", "neden", "çünkü"
        }
        
        # Kelimelere ayır ve temizle
        words = text.lower().split()
        words = [word.strip('.,!?()[]{}":;') for word in words]
        
        # Stop words'leri ve kısa kelimeleri çıkar
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        return keywords

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
        # Soru tipine göre yanıt şablonları
        templates = {
            "avantaj": "Avantajları şunlardır:\n\n",
            "dezavantaj": "Dezavantajları şunlardır:\n\n",
            "özellik": "Özellikleri şunlardır:\n\n",
            "nasıl": "Şu şekilde çalışır:\n\n",
            "nedir": "Şu şekilde tanımlanabilir:\n\n"
        }
        
        # Sorunun tipini belirle
        question_lower = question.lower()
        template_start = ""
        for key, template in templates.items():
            if key in question_lower:
                template_start = template
                break
        
        response_parts = []
        
        # Metinleri işle
        if contents["texts"]:
            relevant_text = ""
            for text in contents["texts"]:
                # Soruyla ilgili cümleleri seç
                sentences = text.split('. ')
                relevant_sentences = []
                
                question_keywords = self._extract_keywords(question)
                for sentence in sentences:
                    sentence_keywords = self._extract_keywords(sentence)
                    # Eğer cümlede soruyla ilgili anahtar kelimeler varsa
                    if any(keyword in sentence_lower for keyword in question_keywords):
                        relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    relevant_text += ". ".join(relevant_sentences) + "\n\n"
            
            if relevant_text:
                response_parts.append(relevant_text)
        
        # Tabloları işle
        if contents["tables"]:
            table_info = self._extract_specific_table_info(contents["tables"][0], question)
            if table_info:
                response_parts.append("\nTablo verilerine göre:\n" + table_info)
        
        # Yanıt oluştur
        if response_parts:
            full_response = template_start + "\n".join(response_parts)
            # Gereksiz boşlukları temizle
            full_response = re.sub(r'\n\s*\n', '\n\n', full_response)
            return full_response.strip()
        
        # Yanıt oluşturulamazsa
        return self._summarize_texts(contents["texts"][:2])

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
