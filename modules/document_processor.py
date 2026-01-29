import fitz
import pdfplumber
from PIL import Image
import pytesseract
import io
import os

class DocumentProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    def extract_text_chunks(self, chunk_size=1000, chunk_overlap=200):
        # Improved chunking using method from implementation plan
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
                
                for chunk_text in page_chunks:
                    chunks.append({
                        'type': 'text',
                        'content': chunk_text,
                        'page': page_num + 1,
                        'source': f'Page {page_num + 1}'
                    })

        return chunks

    def extract_tables(self):
        tables = []

        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    extracted_tables = page.extract_tables()

                    for i, table in enumerate(extracted_tables):
                        if table:
                            # Convert table to string format
                            table_text = ""
                            for row in table:
                                # Filter out None values and join
                                row_text = " | ".join([str(cell) if cell is not None else "" for cell in row])
                                table_text += row_text + "\n"

                            if table_text.strip():
                                tables.append({
                                    'type': 'table',
                                    'content': table_text,
                                    'page': page_num + 1,
                                    'source': f'Table {i+1} on Page {page_num + 1}'
                                })
        except Exception as e:
            print(f"Error extracting tables with pdfplumber: {e}")

        return tables

    def extract_images_with_ocr(self, output_folder=None):
        if output_folder is None:
            try:
                import config
                output_folder = config.IMAGES_DIR
            except:
                output_folder = 'extracted_images'

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        images_data = []

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]

                image_filename = f"{output_folder}/page{page_num+1}_img{img_index+1}.png"
                with open(image_filename, "wb") as image_file:
                    image_file.write(image_bytes)

                try:
                    img_pil = Image.open(io.BytesIO(image_bytes))
                    ocr_text = pytesseract.image_to_string(img_pil)

                    if ocr_text.strip():
                        images_data.append({
                            'type': 'image',
                            'content': ocr_text,
                            'page': page_num + 1,
                            'image_path': image_filename,
                            'source': f'Image on Page {page_num + 1}'
                        })
                except Exception as e:
                    print(f"OCR failed on page {page_num + 1}: {e}")

        return images_data

    def process_document(self):
        print(f"Processing document: {self.pdf_path}")

        text_chunks = self.extract_text_chunks()
        print(f"Extracted {len(text_chunks)} text chunks")

        tables = self.extract_tables()
        print(f"Extracted {len(tables)} tables")

        images = self.extract_images_with_ocr()
        print(f"Extracted {len(images)} images with OCR")

        all_chunks = text_chunks + tables + images
        print(f" Total chunks: {len(all_chunks)}")

        return all_chunks

    def close(self):
        self.doc.close()

if __name__ == "__main__":
    processor = DocumentProcessor("qatar_test_doc.pdf")
    chunks = processor.process_document()
    print(f"\nSample chunk: {chunks[0]}")
    processor.close()