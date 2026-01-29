# Multi-Modal RAG-Based QA System

A specialized Retrieval-Augmented Generation (RAG) system designed to handle complex financial documents. This project moves beyond simple text extraction by handling Table, Images (OCR), and Structured Text as distinct modalities.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)

## Key Features

*   **Multi-Modal Ingestion**:
    *   **Text**: Recursive Character Splitting for semantic coherence.
    *   **Tables**: extracted via pdfplumber and formatted as Markdown to preserve row/column relationships.
    *   **Images**: OCR via pytesseract to unlock data in charts and scanned pages.
*   **Hybrid Search Engine**:
    *   Combines **Dense Vector Search** (FAISS + all-MiniLM-L6-v2) for semantic understanding.
    *   Combines **Sparse Keyword Search** (BM25) for precise term matching.
    *   Merges results using **Reciprocal Rank Fusion (RRF)**.
*   **Advanced Re-ranking**: Uses a Cross-Encoder to re-score citations for maximum relevance.
*   **Local Privacy**: Runs entirely locally using HuggingFace pipelines (no external API keys required).
*   **Interactive UI**: Streamlit dashboard with citation transparency and performance evaluation.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/SAISriram19/bigairlab.git
cd bigairlab
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR
This project uses pytesseract for image OCR. You must have Tesseract installed on your system:
*   **Windows**: Download Installer and add it to your PATH.
*   **Linux**: sudo apt install tesseract-ocr
*   **Mac**: brew install tesseract

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

1.  **Upload**: Use the sidebar to upload a PDF (e.g., a financial report).
2.  **Process**: Click "Process Document" to start the ingestion pipeline.
3.  **Chat**: Ask questions like "What are the key fiscal indicators for 2024?".
4.  **Verify**: Expand the "References" citations to see the exact text/table rows used for the answer.

## Architecture

For a deep dive into the system design, please refer to:
*   [Technical Report](docs/Technical_Report.md)
*   [Architecture Diagrams](docs/Architecture_Diagrams.md)

## Project Structure

*   `app.py`: Main Streamlit application.
*   `modules/`: Core logic files.
    *   `document_processor.py`: Handles PDF parsing.
    *   `vector_store.py`: Manages indices.
    *   `llm_qa.py`: RAG logic.
*   `data/`: Stores processed chunks and vector indices.
*   `docs/`: Documentation and diagrams.
