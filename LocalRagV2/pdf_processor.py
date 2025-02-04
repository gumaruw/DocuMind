import pdfplumber
import docling
import markitdown
from PIL import Image
import io
import pytesseract
from typing import List, Dict

class PDFProcessor:
    def __init__(self):
        self.supported_types = ['text', 'table', 'image']

    def process_document(self, pdf_path: str) -> List[Dict]:
        """PDF dokümanını işler ve yapılandırılmış veri döndürür"""
        processed_content = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Metin çıkarma
                    text = page.extract_text()
                    if text:
                        processed_content.append({
                            'type': 'text',
                            'content': text,
                            'page': page_num
                        })
                    
                    # Tablo çıkarma
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            # Tabloyu metin formatına çevir
                            table_text = self._format_table(table)
                            processed_content.append({
                                'type': 'table',
                                'content': table_text,
                                'page': page_num
                            })
                    
                    # Görsel çıkarma
                    for image in page.images:
                        try:
                            img = Image.open(io.BytesIO(image['stream'].get_data()))
                            text = pytesseract.image_to_string(img, lang='tur')
                            if text.strip():
                                processed_content.append({
                                    'type': 'image',
                                    'content': text,
                                    'page': page_num
                                })
                        except Exception as e:
                            print(f"Görsel işleme hatası (sayfa {page_num}): {e}")
                            
        except Exception as e:
            print(f"PDF işleme hatası: {e}")
            return []
            
        return processed_content

    def _format_table(self, table: List[List]) -> str:
        """Tabloyu okunabilir metin formatına çevirir"""
        formatted_rows = []
        for row in table:
            # Boş hücreleri temizle
            cleaned_row = [str(cell).strip() if cell else '' for cell in row]
            # Boş olmayan hücreleri birleştir
            formatted_row = ' | '.join(cell for cell in cleaned_row if cell)
            if formatted_row:
                formatted_rows.append(formatted_row)
        
        return '\n'.join(formatted_rows)

def tag_data_with_markitdown(data):
    """Çıkarılan veriyi markitdown ile etiketler"""
    tagged_data = []
    for item in data:
       if item["type"] == "text":
           tagged_data.append(f"<text page='{item['page']}'>{item['content']}</text>")
       elif item["type"] == "table":
           tagged_data.append(f"<table page='{item['page']}'>{item['content']}</table>")
       elif item["type"] == "image":
           tagged_data.append(f"<image page='{item['page']}'>{item['content']}</image>")
    return "\n".join(tagged_data)


def read_pdf(pdf_path, config):
    """PDF'yi okur, veriyi çıkarır ve etiketler."""
    extracted_data = PDFProcessor().process_document(pdf_path) #OCR kapalı, tüm kaynaklar çıkar
    if not extracted_data:
        return None
    tagged_data = tag_data_with_markitdown(extracted_data) # markdown etiketlerini ekle

    return tagged_data
