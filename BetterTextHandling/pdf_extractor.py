import fitz  # PyMuPDF
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any

class PDFExtractor:
    def __init__(self, chunk_size: int = 768):  # BERT-base için optimal boyut
        self.chunk_size = chunk_size
        self.overlap = 100  # Bağlam kaybını önlemek için overlap ekledik

    def extract_content(self, pdf_path: str) -> List[Dict]:
        doc = fitz.open(pdf_path)
        contents = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Metin çıkarma
            text = page.get_text()
            
            # Tablo çıkarma
            tables = self._extract_advanced_tables(page)
            
            # Text chunkları
            text_chunks = self._split_into_chunks(text)
            
            for chunk in text_chunks:
                contents.append({
                    'type': 'text',
                    'content': chunk,
                    'page': page_num + 1
                })
            
            # Tabloları ekle
            for table in tables:
                contents.append({
                    'type': 'table',
                    'content': table,
                    'page': page_num + 1
                })

        doc.close()
        return contents

    def _format_table_to_string(self, df: pd.DataFrame) -> str:
        """DataFrame'i okunabilir tablo formatına çevir"""
        # Sütun genişliklerini hesapla
        col_widths = {}
        for col in df.columns:
            col_widths[col] = max(
                len(str(col)),
                df[col].astype(str).map(len).max()
            )
        
        # Başlık satırı
        header = " | ".join(
            str(col).ljust(col_widths[col]) 
            for col in df.columns
        )
        rows = [header]
        
        # Ayraç satırı
        separator = "-" * len(header)
        rows.append(separator)
        
        # Veri satırları
        for _, row in df.iterrows():
            formatted_row = " | ".join(
                str(val).ljust(col_widths[col]) 
                for col, val in row.items()
            )
            rows.append(formatted_row)
        
        return "\n".join(rows)

    def _split_into_chunks(self, text: str) -> List[str]:
        """Metni chunk'lara böl"""
        # Önce yapısal bölme
        sections = text.split('\n\n')
        chunks = []
        
        for section in sections:
            # Bölüm zaten uygun boyuttaysa
            if len(section.strip()) < self.chunk_size:
                if section.strip():
                    chunks.append(section.strip())
                continue
            
            # Uzun bölümleri anlamlı şekilde böl
            current_chunk = []
            current_length = 0
            
            # Paragraf ve cümle sınırlarına dikkat ederek böl
            sentences = re.split('([.!?।]\s+)', section)
            
            for i in range(0, len(sentences) - 1, 2):
                sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
                
                # Cümle tek başına chunk_size'dan büyükse
                if len(sentence) > self.chunk_size:
                    # Mevcut chunk'ı kaydet
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        # Overlap için son cümleyi koru
                        current_chunk = [current_chunk[-1]] if current_chunk else []
                        current_length = len(current_chunk[-1]) if current_chunk else 0
                    
                    # Uzun cümleyi kelime bazında böl
                    words = sentence.split()
                    temp_chunk = []
                    temp_length = 0
                    
                    for word in words:
                        if temp_length + len(word) + 1 <= self.chunk_size:
                            temp_chunk.append(word)
                            temp_length += len(word) + 1
                        else:
                            if temp_chunk:
                                chunks.append(' '.join(temp_chunk))
                            temp_chunk = [word]
                            temp_length = len(word)
                    
                    if temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                        
                # Normal cümle işleme
                elif current_length + len(sentence) <= self.chunk_size:
                    current_chunk.append(sentence)
                    current_length += len(sentence)
                else:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        # Overlap için son cümleyi koru
                        current_chunk = [current_chunk[-1]] if len(current_chunk) > 0 else []
                        current_length = len(current_chunk[-1]) if current_chunk else 0
                    current_chunk.append(sentence)
                    current_length = len(sentence)
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        # Son işlemler
        final_chunks = []
        for chunk in chunks:
            # Gereksiz boşlukları temizle
            cleaned = ' '.join(chunk.split())
            if cleaned:
                final_chunks.append(cleaned)
        
        return final_chunks
    
    def _extract_advanced_tables(self, page) -> List[str]:
        """Tabloları akıllıca tespit et ve işle"""
        tables = []
        table_blocks = []
        current_block = []
        
        blocks = page.get_text("dict")["blocks"]
        
        # Gelişmiş tablo tespit kriterleri
        for block in blocks:
            if block.get("type") == 1:  # text block
                lines = block.get("lines", [])
                
                # Tablo özelliklerini kontrol et
                if len(lines) > 2:  # En az 3 satır olmalı
                    x_positions = []
                    consistent_spans = True
                    
                    # Sütun hizalama analizi
                    for line in lines:
                        spans = line.get("spans", [])
                        if spans:
                            current_positions = [span.get("x0") for span in spans]
                            
                            # İlk satırın x pozisyonlarını referans al
                            if not x_positions:
                                x_positions = current_positions
                            else:
                                # Pozisyon tutarlılığını kontrol et (±5 pixel tolerans)
                                if not all(any(abs(pos - ref) < 5 for ref in x_positions) 
                                         for pos in current_positions):
                                    consistent_spans = False
                                    break
                    
                    if consistent_spans and len(x_positions) > 1:
                        current_block.append(block)
                    else:
                        if current_block:
                            table_blocks.append(current_block)
                            current_block = []
        
        if current_block:
            table_blocks.append(current_block)
        
        # Tablo verilerini işle
        for blocks in table_blocks:
            table_data = []
            header_processed = False
            
            for block in blocks:
                for line in block.get("lines", []):
                    row_data = []
                    last_x = 0
                    
                    # Satır verilerini topla
                    spans = sorted(line.get("spans", []), key=lambda x: x.get("x0", 0))
                    for span in spans:
                        # Boşluk analizi
                        current_x = span.get("x0", 0)
                        if current_x - last_x > 20:  # Sütun ayrımı için boşluk eşiği
                            row_data.append("")
                        
                        text = span.get("text", "").strip()
                        if text:
                            row_data.append(text)
                        last_x = span.get("x1", 0)
                    
                    if row_data:
                        table_data.append(row_data)
            
            if table_data:
                # Tablo verilerini normalize et
                max_cols = max(len(row) for row in table_data)
                normalized_table = []
                
                # Başlık satırını işle
                if table_data:
                    headers = table_data[0]
                    headers.extend([""] * (max_cols - len(headers)))
                    normalized_table.append(headers)
                
                # Veri satırlarını işle
                for row in table_data[1:]:
                    normalized_row = row + [""] * (max_cols - len(row))
                    normalized_table.append(normalized_row)
                
                # Tabloyu string formatına çevir
                table_str = self._format_table_to_string(
                    pd.DataFrame(normalized_table[1:], columns=normalized_table[0])
                )
                tables.append(table_str)
        
        return tables
