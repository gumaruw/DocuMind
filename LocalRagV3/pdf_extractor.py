"""
Metotlar:
- __init__(self, chunk_size: int): PDFExtractor sınıfının yapıcı metodu, chunk boyutunu ayarlar.
- extract_content(self, pdf_path: str) -> List[Dict]: PDF içeriğini çıkarır.
- _is_potential_table(self, text: str) -> bool: Metin bloğunun tablo olup olmadığını kontrol eder.
- _analyze_line_structure(self, line: str) -> str: Satır yapısını analiz eder.
- _extract_tables(self, page) -> List[str]: Sayfadaki tablo bölgelerini tespit eder ve çıkarır.
- _detect_and_extract_tables(self, page) -> List[str]: PyMuPDF kullanarak tabloları tespit eder ve çıkarır.
- _fallback_table_detection(self, page) -> List[str]: Yedek tablo algılama yöntemi.
- _extract_table_region(self, text: str, start_pos: int) -> str: Metin içinden tablo bölgesini çıkarır.
- _convert_to_table_format(self, table_text: str) -> str: Tablo metnini standart formata çevirir.
- _format_table(self, df: pd.DataFrame) -> str: DataFrame'i okunabilir tablo formatına çevirir.
- _split_into_chunks(self, text: str) -> List[str]: Metni chunklara böler.
- _extract_table_data(self, text: str) -> List[List[str]]: Tabloyu satır ve sütunlara ayırır.
- _normalize_table_data(self, table_data: List[List[str]]) -> List[List[str]]: Tablo verilerini normalize eder.
- _split_into_blocks(self, text: str) -> List[str]: Metni anlamlı bloklara ayırır.
- _format_markdown_table(self, table_data: List[List[str]]) -> str: Tablo verilerini Markdown formatına dönüştürür.
"""
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import re
import logging

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('table_detection.log'),
        logging.StreamHandler()
    ]
)

class PDFExtractor:
    def __init__(self, chunk_size: int = 512):  
        # BERT-base için optimal boyut 768 ama belleği daha az kullanmam gerek
        self.chunk_size = chunk_size
        self.overlap = 50  # Bağlam kaybını önlemek için overlap ekledim
        self.logger = logging.getLogger(__name__)

    def extract_content(self, pdf_path: str) -> List[Dict]:
        try:
            doc = fitz.open(pdf_path)
            contents = []
            total_tables = 0

            self.logger.info(f"PDF dosyası açıldı: {pdf_path}")
            self.logger.info(f"Toplam sayfa sayısı: {len(doc)}")

            for page_num in range(len(doc)):
                page = doc[page_num]
                self.logger.info(f"Sayfa {page_num + 1} işleniyor...")
                
                # Tabloları tespit et
                tables = self._detect_and_extract_tables(page)
                page_table_count = len(tables)
                total_tables += page_table_count
                
                self.logger.info(f"Sayfa {page_num + 1}'de {page_table_count} tablo bulundu")
                
                for table in tables:
                    if table.strip():
                        self.logger.debug(f"Tablo içeriği:\n{table[:200]}...")  # İlk 200 karakter
                        contents.append({
                            'type': 'table',
                            'content': table,
                            'page': page_num + 1
                        })

                # Metin içeriğini çıkar
                text = page.get_text()
                text_chunks = self._split_into_chunks(text)
                
                for chunk in text_chunks:
                    if chunk.strip():
                        contents.append({
                            'type': 'text',
                            'content': chunk,
                            'page': page_num + 1
                        })

            self.logger.info(f"Toplam {total_tables} tablo tespit edildi")
            self.logger.info(f"Toplam {len(contents)} içerik parçası çıkarıldı")
            
            doc.close()
            return contents
        except Exception as e:
            self.logger.error(f"PDF işleme hatası: {str(e)}", exc_info=True)
            return []

    def _is_potential_table(self, text: str) -> bool:
        """Metin bloğunun tablo olup olmadığını kontrol et"""
        lines = text.strip().split('\n')
        if len(lines) < 2:  # En az 2 satır olmalı
            return False

        # Tablo göstergelerini kontrol et
        indicators = {
            'separators': False,  # | veya tab karakterleri
            'aligned_spaces': False,  # Hizalanmış boşluklar
            'consistent_structure': False  # Tutarlı yapı
        }

        # İlk iki satırın yapısını analiz et
        first_line_structure = self._analyze_line_structure(lines[0])
        second_line_structure = self._analyze_line_structure(lines[1])

        # Ayraç karakterleri kontrolü
        if any('|' in line for line in lines) or any('\t' in line for line in lines):
            indicators['separators'] = True

        # Hizalanmış boşluklar kontrolü
        space_positions = [i for i, char in enumerate(lines[0]) if char == ' ']
        if any(all(i < len(line) and line[i] == ' ' for line in lines[1:]) for i in space_positions):
            indicators['aligned_spaces'] = True

        # Yapı tutarlılığı kontrolü
        if first_line_structure and first_line_structure == second_line_structure:
            indicators['consistent_structure'] = True

        # En az iki gösterge varsa tablo olarak kabul et
        return sum(indicators.values()) >= 2
    
    def _analyze_line_structure(self, line: str) -> str:
        """Satır yapısını analiz et"""
        # Boşluk gruplarını ve kelime gruplarını tespit et
        structure = ''
        current_type = None
        count = 0

        for char in line:
            char_type = 'S' if char.isspace() else 'W'  # Space veya Word
            
            if char_type != current_type:
                if current_type:
                    structure += f"{current_type}{count}"
                current_type = char_type
                count = 1
            else:
                count += 1

        if current_type:
            structure += f"{current_type}{count}"

        return structure

    def _extract_tables(self, page) -> List[str]:
        # Tablo bölgelerini tespit et
        table_regions = self._detect_table_regions(page)
        tables = []
        
        for region in table_regions:
            table_text = page.get_text("text", clip=region)
            if table_text.strip():
                df = self._parse_table_text(table_text)
                if not df.empty and len(df.columns) > 1:
                    tables.append(self._format_table(df))
        
        return tables

    def _detect_and_extract_tables(self, page) -> List[str]:
        tables = []
        try:
            # PyMuPDF'in tablo algılama özelliğini kullan
            tab = page.find_tables()
            if tab.tables:
                self.logger.info(f"PyMuPDF ile {len(tab.tables)} tablo bulundu")
                for table in tab.tables:
                    try:
                        # Tablo verilerini al
                        rows = []
                        header = None
                        
                        # Tüm hücreleri kontrol et
                        for i, row in enumerate(table.cells):
                            row_data = [cell.text.strip() for cell in row if hasattr(cell, 'text')]
                            
                            # Boş satırları atla
                            if not any(cell for cell in row_data):
                                continue
                                
                            # İlk anlamlı satırı başlık olarak al
                            if header is None:
                                header = row_data
                                continue
                                
                            rows.append(row_data)
                        
                        # Başlık ve en az bir satır veri varsa
                        if header and rows:
                            self.logger.debug(f"Tablo başlıkları: {header}")
                            self.logger.debug(f"Tablo satır sayısı: {len(rows)}")
                            
                            # Tüm satırların başlık sayısı kadar sütunu olduğunu kontrol et
                            if all(len(row) == len(header) for row in rows):
                                df = pd.DataFrame(rows, columns=header)
                                formatted_table = self._format_table(df)
                                if formatted_table:
                                    tables.append(formatted_table)
                                    self.logger.info(f"Tablo başarıyla formatlandı ve eklendi")
                            else:
                                self.logger.warning("Sütun sayıları uyumsuz, tablo atlandı")
                        
                    except Exception as table_error:
                        self.logger.error(f"Tekil tablo işleme hatası: {str(table_error)}")
                        continue

        except Exception as e:
            self.logger.error(f"Tablo algılama hatası: {str(e)}")
            # Yedek tablo algılama yöntemi
            self.logger.info("Yedek tablo algılama yöntemi deneniyor...")
            backup_tables = self._fallback_table_detection(page)
            if backup_tables:
                self.logger.info(f"Yedek yöntem ile {len(backup_tables)} tablo bulundu")
                tables.extend(backup_tables)
        
        return tables

    def _fallback_table_detection(self, page) -> List[str]:
        """Yedek tablo algılama yöntemi"""
        tables = []
        text = page.get_text()
        
        # Olası tablo başlangıçlarını bul
        table_patterns = [
            (r'\n([\w\s]+\|[\w\s]+\|[\w\s]+\n)', 'pipe'),
            (r'\n([\w\s]+\t[\w\s]+\t[\w\s]+\n)', 'tab'),
            (r'\n([\w\s]+\s{3,}[\w\s]+\s{3,}[\w\s]+\n)', 'space')
        ]
        
        for pattern, pattern_type in table_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                self.logger.debug(f"{pattern_type} tipinde olası tablo bulundu")
                table_text = self._extract_table_region(text, match.start())
                if table_text:
                    formatted_table = self._convert_to_table_format(table_text)
                    if formatted_table:
                        self.logger.info(f"{pattern_type} tipinde tablo başarıyla çıkarıldı")
                        tables.append(formatted_table)
        
        return tables

    def _extract_table_region(self, text: str, start_pos: int) -> str:
        """Metin içinden tablo bölgesini çıkar"""
        lines = text[start_pos:].split('\n')
        table_lines = []
        
        # İlk satırı al
        header_pattern = re.compile(r'[\w\s]+[\|\t\s{3,}][\w\s]+')
        if not header_pattern.match(lines[0]):
            return ""
        
        table_lines.append(lines[0])
        
        # Takip eden satırları kontrol et
        for line in lines[1:]:
            if not line.strip() or not header_pattern.match(line):
                break
            table_lines.append(line)
        
        return '\n'.join(table_lines)

    def _convert_to_table_format(self, table_text: str) -> str:
        """Tablo metnini standart formata çevir"""
        lines = table_text.strip().split('\n')
        if len(lines) < 2:  # En az başlık ve bir veri satırı olmalı
            return ""
        
        # Ayırıcıyı belirle (|, \t veya çoklu boşluk)
        if '|' in lines[0]:
            separator = '|'
        elif '\t' in lines[0]:
            separator = '\t'
        else:
            separator = None  # Çoklu boşluk durumu
        
        # Satırları böl
        if separator:
            rows = [row.split(separator) for row in lines]
        else:
            rows = [re.split(r'\s{3,}', row) for row in lines]
        
        # DataFrame oluştur
        try:
            df = pd.DataFrame(rows[1:], columns=rows[0])
            return self._format_table(df)
        except:
            return ""

    def _format_table(self, df: pd.DataFrame) -> str:
        """DataFrame'i okunabilir tablo formatına çevir"""
        if df.empty or len(df.columns) < 2:
            self.logger.warning("Boş veya yetersiz sütuna sahip tablo")
            return ""
        
        try:
            # NaN değerleri boş string ile değiştir
            df = df.fillna('')
            
            # Sütun genişliklerini hesapla
            col_widths = {}
            for col in df.columns:
                # Başlık ve içerikteki en uzun stringi bul
                max_content_length = df[col].astype(str).map(len).max()
                col_widths[col] = max(len(str(col)), max_content_length) + 2
            
            # Başlık satırı
            header = "| " + " | ".join(
                str(col).ljust(col_widths[col]) 
                for col in df.columns
            ) + " |"
            
            # Ayraç satırı
            separator = "|" + "|".join(
                "-" * (col_widths[col] + 2)
                for col in df.columns
            ) + "|"
            
            # Veri satırları
            rows = []
            for _, row in df.iterrows():
                formatted_row = "| " + " | ".join(
                    str(val).ljust(col_widths[col]) 
                    for col, val in row.items()
                ) + " |"
                rows.append(formatted_row)
            
            formatted_table = "\n".join([header, separator] + rows)
            if len(formatted_table.strip()) > 0:
                self.logger.info("Tablo başarıyla formatlandı")
                return formatted_table
            return ""
            
        except Exception as e:
            self.logger.error(f"Tablo formatlama hatası: {str(e)}")
            return ""

    def _split_into_chunks(self, text: str) -> List[str]:
        """Metni chunklara böl"""
        # Escape sequence hatası düzeltildi
        sentences = re.split(r'([.!?।]\s+)', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
            
            if current_length + len(sentence) <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += len(sentence)
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def _analyze_line_structure(self, line: str) -> str:
        """Satır yapısını analiz et"""
        # Boşluk gruplarını ve kelime gruplarını tespit et
        structure = ''
        current_type = None
        count = 0

        for char in line:
            char_type = 'S' if char.isspace() else 'W'  # Space veya Word
            
            if char_type != current_type:
                if current_type:
                    structure += f"{current_type}{count}"
                current_type = char_type
                count = 1
            else:
                count += 1

        if current_type:
            structure += f"{current_type}{count}"

        return structure

    def _extract_table_data(self, text: str) -> List[List[str]]:
        """Tabloyu satır ve sütunlara ayır"""
        lines = text.strip().split('\n')
        table_data = []

        for line in lines:
            # Boş satırları atla
            if not line.strip():
                continue

            # Ayraç karakterine göre böl
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|')]
                cells = [cell for cell in cells if cell]  # Boş hücreleri temizle
            
            # Tab karakterine göre böl
            elif '\t' in line:
                cells = [cell.strip() for cell in line.split('\t')]
            
            # Hizalanmış boşluklara göre böl
            else:
                cells = [cell.strip() for cell in re.split(r'\s{2,}', line.strip())]

            if cells:  # Boş olmayan satırları ekle
                table_data.append(cells)

        return self._normalize_table_data(table_data)

    def _normalize_table_data(self, table_data: List[List[str]]) -> List[List[str]]:
        """Tablo verilerini normalize et"""
        if not table_data:
            return []

        # En çok sütuna sahip satırı bul
        max_columns = max(len(row) for row in table_data)

        # Tüm satırları aynı sütun sayısına getir
        normalized_data = []
        for row in table_data:
            # Eksik sütunları boş string ile doldur
            normalized_row = row + [''] * (max_columns - len(row))
            normalized_data.append(normalized_row)

        return normalized_data

    def extract_content(self, pdf_path: str) -> List[Dict]:
        try:
            doc = fitz.open(pdf_path)
            contents = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                # Metin bloklarını ayır
                blocks = self._split_into_blocks(text)

                for block in blocks:
                    if self._is_potential_table(block):
                        table_data = self._extract_table_data(block)
                        if len(table_data) >= 2:  # En az başlık ve bir veri satırı
                            formatted_table = self._format_markdown_table(table_data)
                            if formatted_table:
                                contents.append({
                                    'type': 'table',
                                    'content': formatted_table,
                                    'page': page_num + 1
                                })
                    else:
                        # Normal metin bloklarını işle
                        if block.strip():
                            chunks = self._split_into_chunks(block)
                            for chunk in chunks:
                                if chunk.strip():
                                    contents.append({
                                        'type': 'text',
                                        'content': chunk,
                                        'page': page_num + 1
                                    })

            doc.close()
            return contents

        except Exception as e:
            self.logger.error(f"PDF işleme hatası: {str(e)}")
            return []

    def _split_into_blocks(self, text: str) -> List[str]:
        """Metni anlamlı bloklara ayır"""
        # Boş satırları temel alarak bloklara ayır
        initial_blocks = text.split('\n\n')
        
        final_blocks = []
        current_block = []
        
        for block in initial_blocks:
            if not block.strip():
                continue
                
            # Eğer mevcut blok potansiyel bir tablo ise
            if current_block and (self._is_potential_table('\n'.join(current_block)) != 
                                self._is_potential_table(block)):
                final_blocks.append('\n'.join(current_block))
                current_block = []
            
            current_block.append(block)
            
        if current_block:
            final_blocks.append('\n'.join(current_block))
            
        return final_blocks

    def _format_markdown_table(self, table_data: List[List[str]]) -> str:
        """Tablo verilerini Markdown formatına dönüştür"""
        if not table_data or len(table_data) < 2:
            return ""

        # Sütun genişliklerini hesapla
        col_widths = []
        for col in range(len(table_data[0])):
            width = max(len(str(row[col])) for row in table_data if col < len(row))
            col_widths.append(width + 2)  # 2 karakter padding

        # Başlık satırı
        header = "| " + " | ".join(
            str(cell).ljust(width) for cell, width in zip(table_data[0], col_widths)
        ) + " |"

        # Ayraç satırı
        separator = "|" + "|".join("-" * width for width in col_widths) + "|"

        # Veri satırları
        rows = []
        for row in table_data[1:]:
            formatted_row = "| " + " | ".join(
                str(cell).ljust(width) for cell, width in zip(row, col_widths)
            ) + " |"
            rows.append(formatted_row)

        return "\n".join([header, separator] + rows)
