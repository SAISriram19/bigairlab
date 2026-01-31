import fitz
import pdfplumber
from PIL import Image
import pytesseract
import io
import os
import logging
from modules import config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Multi-modal document processor for extracting text, tables, and images from PDFs.
    
    Handles:
    - Text extraction with semantic chunking
    - Table extraction with markdown formatting
    - Image extraction with OCR processing
    """
    
    def __init__(self, pdf_path):
        """
        Initialize document processor.
        
        Args:
            pdf_path: Path to the PDF file to process
        """
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    def extract_text_chunks(self, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP):
        """
        Extract and chunk text from PDF using semantic-aware splitting.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between consecutive chunks
            
        Returns:
            List of text chunks with metadata
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()

            if text.strip():
                page_chunks = text_splitter.split_text(text)
                
                for i, chunk_text in enumerate(page_chunks):
                    chunks.append({
                        'type': 'text',
                        'content': chunk_text.strip(),
                        'page': page_num + 1,
                        'source': f'Page {page_num + 1}, Chunk {i + 1}',
                        'chunk_index': i
                    })

        return chunks

    def extract_tables(self):
        """
        Extract tables from PDF and convert to structured format.
        
        Returns:
            List of table chunks with metadata
        """
        tables = []

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    extracted_tables = page.extract_tables()

                    for i, table in enumerate(extracted_tables):
                        if table and len(table) > 1:  # Ensure table has header + data
                            # Convert table to markdown format for better structure
                            table_text = self._format_table_as_markdown(table)

                            if table_text.strip():
                                tables.append({
                                    'type': 'table',
                                    'content': table_text,
                                    'page': page_num + 1,
                                    'source': f'Table {i+1} on Page {page_num + 1}',
                                    'table_index': i,
                                    'row_count': len(table),
                                    'col_count': len(table[0]) if table else 0
                                })
        except Exception as e:
            logger.error(f"Error extracting tables with pdfplumber: {e}")

        return tables
    
    def _format_table_as_markdown(self, table):
        """Convert table to markdown format for better structure preservation."""
        if not table:
            return ""
            
        markdown_lines = []
        
        # Process header
        if table[0]:
            header = " | ".join([str(cell) if cell is not None else "" for cell in table[0]])
            markdown_lines.append(header)
            # Add separator
            separator = " | ".join(["---"] * len(table[0]))
            markdown_lines.append(separator)
        
        # Process data rows
        for row in table[1:]:
            if row:
                row_text = " | ".join([str(cell) if cell is not None else "" for cell in row])
                markdown_lines.append(row_text)
        
        return "\n".join(markdown_lines)

    def extract_images_with_ocr(self, output_folder=None):
        """
        Extract images from PDF and perform OCR to extract text.
        
        Args:
            output_folder: Directory to save extracted images
            
        Returns:
            List of image chunks with OCR text and metadata
        """
        if output_folder is None:
            output_folder = config.IMAGES_DIR

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        images_data = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = self.doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    image_filename = f"{output_folder}/page{page_num+1}_img{img_index+1}.png"
                    
                    # Save image file
                    with open(image_filename, "wb") as image_file:
                        image_file.write(image_bytes)

                    # Perform OCR
                    img_pil = Image.open(io.BytesIO(image_bytes))
                    
                    # Enhance image for better OCR if needed
                    if img_pil.mode != 'RGB':
                        img_pil = img_pil.convert('RGB')
                    
                    ocr_text = pytesseract.image_to_string(img_pil, config='--psm 6')

                    if ocr_text.strip():
                        images_data.append({
                            'type': 'image',
                            'content': ocr_text.strip(),
                            'page': page_num + 1,
                            'image_path': image_filename,
                            'source': f'Image {img_index + 1} on Page {page_num + 1}',
                            'image_index': img_index,
                            'image_size': len(image_bytes)
                        })
                    else:
                        logger.debug(f"No text found in image on page {page_num + 1}")
                        
                except Exception as e:
                    logger.warning(f"Failed to process image {img_index + 1} on page {page_num + 1}: {e}")

        return images_data

    def process_document(self):
        logger.info(f"Processing document: {self.pdf_path}")

        text_chunks = self.extract_text_chunks()
        logger.info(f"Extracted {len(text_chunks)} text chunks")

        tables = self.extract_tables()
        logger.info(f"Extracted {len(tables)} tables")

        images = self.extract_images_with_ocr()
        logger.info(f"Extracted {len(images)} images with OCR")

        all_chunks = text_chunks + tables + images
        logger.info(f"Total chunks: {len(all_chunks)}")

        return all_chunks

    def close(self):
        self.doc.close()

if __name__ == "__main__":
    processor = DocumentProcessor("qatar_test_doc.pdf")
    chunks = processor.process_document()
    print(f"\nSample chunk: {chunks[0]}")
    processor.close()