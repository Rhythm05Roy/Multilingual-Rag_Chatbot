import pytesseract
from pdf2image import convert_from_path
import re
import nltk
from nltk.tokenize import sent_tokenize

class PDFProcessor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using OCR with Bengali language support"""
        # Convert PDF pages to images
        images = convert_from_path(pdf_path)
        
        # OCR each image using Bengali language
        full_text = ""
        for img in images:
            text = pytesseract.image_to_string(img, lang='ben')
            full_text += text + "\n"
        
        # Clean the OCR text
        cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
        
        return cleaned_text
    
    def create_chunks(self, text, max_words=200):
        """Create sentence-aware chunks from text"""
        # Sentence-aware chunking
        sentences = sent_tokenize(text)  
        chunks = []
        chunk = ""
        
        for sent in sentences:
            if len(chunk.split()) + len(sent.split()) <= max_words:
                chunk += " " + sent
            else:
                chunks.append(chunk.strip())
                chunk = sent
        
        if chunk:  
            chunks.append(chunk.strip())
        
        return chunks
    
    def process_pdf(self, pdf_path, max_words=200):
        """Complete PDF processing pipeline"""
        print("🔄 Extracting text from PDF...")
        text = self.extract_text_from_pdf(pdf_path)
        print(f"✅ Extracted {len(text)} characters from PDF")
        
        print("🔄 Creating chunks...")
        chunks = self.create_chunks(text, max_words)
        print(f"✅ Generated {len(chunks)} sentence-aware chunks")
        
        return text, chunks