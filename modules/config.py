import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
VECTOR_STORE_DIR = os.path.join(DATA_DIR, 'vector_store')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')

PDF_PATH = os.path.join(BASE_DIR, 'multi-modal_rag_qa_assignment (1).pdf')
CHUNKS_PATH = os.path.join(PROCESSED_DATA_DIR, 'extracted_chunks.json')
VECTOR_STORE_PATH = os.path.join(VECTOR_STORE_DIR, 'faiss_index')

EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = 'google/flan-t5-base'

# Magic Numbers / Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3
REPETITION_PENALTY = 1.15
MAX_FILE_SIZE_MB = 10

def create_directories():
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        VECTOR_STORE_DIR,
        IMAGES_DIR
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)