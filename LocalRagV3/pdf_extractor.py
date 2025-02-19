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
        logging.FileHandler('pdf_extraction.log'),
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
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Sayfadaki tüm blokları al
                blocks = page.get_text("dict")["blocks"]
                
                # Her bloğu analiz et
                for block in blocks:
                    block_type = block["type"]
                    
                    if block_type == 0:  # Text block
                        if self._is_table_block(block):
                            table_content = self._process_table_block(block)
                            if table_content:
                                contents.append({
                                    'type': 'table',
                                    'content': table_content,
                                    'page': page_num + 1,
                                    'bbox': block["bbox"]
                                })
                        else:
                            text_content = self._process_text_block(block)
                            if text_content:
                                contents.append({
                                    'type': 'text',
                                    'content': text_content,
                                    'page': page_num + 1,
                                    'bbox': block["bbox"]
                                })
                    
                    elif block_type == 1:  # Image block
                        contents.append({
                            'type': 'image',
                            'content': f"[Görsel - Sayfa {page_num + 1}]",
                            'page': page_num + 1,
                            'bbox': block["bbox"]
                        })
                
                # Sayfa bittikten sonra tabloları birleştir
                contents = self._merge_adjacent_tables(contents)
            
            doc.close()
            return contents
            
        except Exception as e:
            self.logger.error(f"PDF işleme hatası: {str(e)}")
            return []

    def _is_table_block(self, block: Dict) -> bool:
        """Bir bloğun tablo olup olmadığını kontrol et"""
        if "lines" not in block:
            return False
            
        lines = block["lines"]
        if len(lines) < 2:  # En az 2 satır olmalı
            return False
            
        # İlk iki satırın yapısını karşılaştır
        first_line_structure = self._get_line_structure(lines[0])
        second_line_structure = self._get_line_structure(lines[1])
        
        # Yapısal özellikleri kontrol et
        has_aligned_spans = self._check_span_alignment(lines)
        has_consistent_structure = first_line_structure == second_line_structure
        has_multiple_columns = len(first_line_structure) > 1
        
        return has_aligned_spans and has_consistent_structure and has_multiple_columns

    def _get_line_structure(self, line: Dict) -> List[Tuple[float, float]]:
        """Satırdaki metin parçalarının x-koordinatlarını çıkar"""
        spans = line["spans"]
        return [(span["bbox"][0], span["bbox"][2]) for span in spans]

    def _check_span_alignment(self, lines: List[Dict]) -> bool:
        """Satırlardaki metin parçalarının hizalı olup olmadığını kontrol et"""
        if not lines:
            return False
            
        # İlk satırın x-koordinatlarını referans al
        reference_spans = self._get_line_structure(lines[0])
        
        # Diğer satırların benzer hizalamaya sahip olup olmadığını kontrol et
        for line in lines[1:]:
            current_spans = self._get_line_structure(line)
            
            # Span sayıları çok farklıysa tablo değil
            if abs(len(reference_spans) - len(current_spans)) > 1:
                return False
                
            # X-koordinatları benzer mi kontrol et
            for ref_span, curr_span in zip(reference_spans, current_spans):
                if abs(ref_span[0] - curr_span[0]) > 5:  # 5 piksel tolerans
                    return False
                    
        return True

    def _process_table_block(self, block: Dict) -> str:
        """Tablo bloğunu işle ve markdown formatına çevir"""
        try:
            # Satırları ve sütunları çıkar
            rows = []
            for line in block["lines"]:
                row = []
                for span in line["spans"]:
                    text = span["text"].strip()
                    row.append(text)
                if any(cell.strip() for cell in row):  # Boş satırları atla
                    rows.append(row)
            
            if len(rows) < 2:  # En az başlık ve bir veri satırı
                return ""
                
            # En çok sütuna sahip satırı bul
            max_cols = max(len(row) for row in rows)
            
            # Tüm satırları aynı sütun sayısına getir
            normalized_rows = []
            for row in rows:
                normalized_row = row + [""] * (max_cols - len(row))
                normalized_rows.append(normalized_row)
            
            # DataFrame oluştur
            df = pd.DataFrame(normalized_rows[1:], columns=normalized_rows[0])
            
            # Markdown formatına çevir
            return self._format_table(df)
            
        except Exception as e:
            self.logger.error(f"Tablo işleme hatası: {str(e)}")
            return ""

    def _process_text_block(self, block: Dict) -> str:
        """Metin bloğunu işle"""
        try:
            text_parts = []
            for line in block["lines"]:
                line_text = " ".join(span["text"].strip() for span in line["spans"])
                if line_text.strip():
                    text_parts.append(line_text)
            
            return " ".join(text_parts)
            
        except Exception as e:
            self.logger.error(f"Metin işleme hatası: {str(e)}")
            return ""

    def _format_table(self, df: pd.DataFrame) -> str:
        """DataFrame'i markdown tablo formatına çevir"""
        if df.empty or len(df.columns) < 2:
            return ""
            
        try:
            # NaN değerleri temizle
            df = df.fillna("")
            
            # Sütun genişliklerini hesapla
            col_widths = {}
            for col in df.columns:
                max_content_length = max(
                    len(str(col)),
                    df[col].astype(str).map(len).max()
                ) + 2
                col_widths[col] = max_content_length
            
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
                    for val in row
                ) + " |"
                rows.append(formatted_row)
            
            return "\n".join([header, separator] + rows)
            
        except Exception as e:
            self.logger.error(f"Tablo formatlama hatası: {str(e)}")
            return ""

    def _merge_adjacent_tables(self, contents: List[Dict]) -> List[Dict]:
        """Birbirine yakın tabloları birleştir"""
        if len(contents) < 2:
            return contents
            
        merged_contents = []
        i = 0
        
        while i < len(contents):
            if i == len(contents) - 1:
                merged_contents.append(contents[i])
                break
                
            current = contents[i]
            next_content = contents[i + 1]
            
            # İki ardışık içerik de tablo mu ve yakın mı kontrol et
            if (current['type'] == 'table' and 
                next_content['type'] == 'table' and 
                self._are_blocks_adjacent(current['bbox'], next_content['bbox'])):
                
                # Tabloları birleştir
                merged_table = self._merge_table_contents(
                    current['content'], 
                    next_content['content']
                )
                
                merged_contents.append({
                    'type': 'table',
                    'content': merged_table,
                    'page': current['page'],
                    'bbox': self._merge_bboxes(current['bbox'], next_content['bbox'])
                })
                
                i += 2  # İki tabloyu da işledik
                
            else:
                merged_contents.append(current)
                i += 1
                
        return merged_contents

    def _are_blocks_adjacent(self, bbox1: List[float], bbox2: List[float], 
                           threshold: float = 20.0) -> bool:
        """İki bloğun birbirine yakın olup olmadığını kontrol et"""
        _, y1_bottom = bbox1[1], bbox1[3]
        _, y2_top = bbox2[1], bbox2[3]
        
        return abs(y1_bottom - y2_top) < threshold

    def _merge_bboxes(self, bbox1: List[float], bbox2: List[float]) -> List[float]:
        """İki sınırlayıcı kutuyu birleştir"""
        return [
            min(bbox1[0], bbox2[0]),  # x_min
            min(bbox1[1], bbox2[1]),  # y_min
            max(bbox1[2], bbox2[2]),  # x_max
            max(bbox1[3], bbox2[3])   # y_max
        ]

    def _merge_table_contents(self, table1: str, table2: str) -> str:
        """İki tablo içeriğini birleştir"""
        try:
            # Tabloları satırlara böl
            rows1 = table1.split('\n')
            rows2 = table2.split('\n')
            
            # Başlık satırları aynıysa birleştir
            if rows1[0] == rows2[0]:
                return "\n".join(rows1 + rows2[2:])  # Ayraç satırını tekrarlama
            else:
                return table1 + "\n" + table2
                
        except Exception as e:
            self.logger.error(f"Tablo birleştirme hatası: {str(e)}")
            return table1
