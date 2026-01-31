import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
VECTOR_STORE_DIR = os.path.join(DATA_DIR, 'vector_store')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')

PDF_PATH = os.path.join(BASE_DIR, 'multi-modal_rag_qa_assignment (1).pdf')
CHUNKS_PATH = os.path.join(PROCESSED_DATA_DIR, 'extracted_chunks.json')
VECTOR_STORE_PATH = os.path.join(VECTOR_STORE_DIR, 'faiss_index')

# Model configurations - can be overridden by environment variables
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
LLM_MODEL = os.getenv('LLM_MODEL', 'google/flan-t5-base')

# Processing parameters
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))
MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', '512'))
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.3'))
REPETITION_PENALTY = float(os.getenv('REPETITION_PENALTY', '1.15'))
MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '10'))

# Search parameters
SEARCH_K = int(os.getenv('SEARCH_K', '5'))
RERANK_TOP_K = int(os.getenv('RERANK_TOP_K', '3'))

# Performance optimizations
MAX_CONTEXT_LENGTH = int(os.getenv('MAX_CONTEXT_LENGTH', '1500'))

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