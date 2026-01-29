# Multi-Modal RAG-Based QA System

This project implements a Retrieval-Augmented Generation (RAG) system capable of processing text, tables, and images from PDF documents.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Features

- **Multi-Modal Ingestion**: Handles text, tables, and images (OCR).
- **Advanced Chunking**: Uses `RecursiveCharacterTextSplitter` for semantic context preservation.
- **Hybrid Retrieval**: Combines Vector Search (FAISS) + Keyword Search (BM25) with Reciprocal Rank Fusion (RRF).
- **Interactive UI**: Streamlit app with **File Upload** and dedicated **Evaluation Dashboard**.
- **QA Bot**: Powered by `google/flan-t5-base` with citation support and repetition penalties.
- **Evaluation Suite**: Built-in benchmarking for latency and retrieval accuracy.

## Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   *Note: First run will download necessary models.*

3. **Upload & Chat**:
   - Upload any PDF in the sidebar.
   - Click "Process Document".
   - Ask questions or check the "Evaluation" tab.

## Components

- `document_processor.py`: PDF parsing logic.
- `vector_store.py`: FAISS index management.
- `llm_qa.py`: Answer generation logic.
- `app.py`: Streamlit user interface.
- `Technical_Report.md`: Detailed architecture documentation.
