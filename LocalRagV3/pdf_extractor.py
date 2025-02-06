# pdf_extractor.py
import fitz  # PyMuPDF
from typing import List, Dict
import re

class PDFExtractor:
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size

    def extract_content(self, pdf_path: str) -> List[Dict]:
        """PDF'den içerik çıkarır ve chunk'lara böler"""
        doc = fitz.open(pdf_path)
        contents = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Metin çıkarma
            text = page.get_text()
            if text.strip():
                # Metni chunklara böl
                chunks = self._split_into_chunks(text)
                for chunk in chunks:
                    contents.append({
                        'type': 'text',
                        'content': chunk,
                        'page': page_num + 1
                    })

            # Tablo çıkarma
            tables = self._extract_tables(page)
            for table in tables:
                contents.append({
                    'type': 'table',
                    'content': table,
                    'page': page_num + 1
                })

        doc.close()
        return contents

    def _split_into_chunks(self, text: str) -> List[str]:
        """Metni belirli boyutta chunk'lara böler"""
        chunks = []
        sentences = re.split('([.!?।])', text)
        current_chunk = []
        current_length = 0

        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip()
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]

            if current_length + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _extract_tables(self, page) -> List[str]:
        """Sayfadaki tabloları çıkarır"""
        tables = []
        # PyMuPDF ile tablo benzeri yapıları tespit et
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") == 1:  # Tablo benzeri yapı
                lines = block.get("lines", [])
                if lines:
                    table_text = []
                    for line in lines:
                        spans = line.get("spans", [])
                        row_text = " | ".join(span.get("text", "").strip() for span in spans if span.get("text", "").strip())
                        if row_text:
                            table_text.append(row_text)
                    if table_text:
                        tables.append("\n".join(table_text))
        return tables
