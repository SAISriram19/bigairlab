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

### 3. Install Missing Package
```bash
pip install rank_bm25
```

### 4. Install Tesseract OCR
This project uses pytesseract for image OCR. You must have Tesseract installed on your system:
*   **Windows**: Download Installer and add it to your PATH.
*   **Linux**: sudo apt install tesseract-ocr
*   **Mac**: brew install tesseract

### 5. Configuration (Optional)
Copy `.env.example` to `.env` and adjust settings:
```bash
cp .env.example .env
```

## Performance Notes

- **CPU vs GPU**: The system runs on CPU by default. Generation times are 15-20s on CPU, 2-3s with GPU.
- **Optimization**: Reduce `MAX_NEW_TOKENS` in `.env` for faster responses.
- **Memory**: Requires ~4GB RAM for full operation with all models loaded.

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

1.  **Upload**: Use the sidebar to upload a PDF (e.g., a financial report).
2.  **Process**: Click "Process Document" to start the ingestion pipeline.
3.  **Chat**: Ask questions like "What are the key fiscal indicators for 2024?".
4.  **Verify**: Expand the "References" citations to see the exact text/table rows used for the answer.
5.  **Evaluate**: Use the "Evaluation" tab to benchmark system performance.

## Test Questions for Qatar Document

Try these questions with the included Qatar IMF report:
- "What is Qatar's projected GDP growth rate for 2024-25?"
- "What were Qatar's fiscal and current account surpluses in 2023?"
- "How is Qatar's banking sector performing in terms of capitalization?"
- "What are the main risks to Qatar's economic outlook?"

## Architecture

For a deep dive into the system design, please refer to:
*   [Technical Report](docs/Technical_Report.md)
*   [Architecture Diagrams](docs/Architecture_Diagrams.md)

## Development

### Running Tests
To run the automated test suite:
```bash
python -m pytest tests/
```

## Project Structure

*   `app.py`: Main Streamlit application.
*   `modules/`: Core logic files (Config, Processor, VectorStore, LLMQA).
*   `data/`: Stores processed chunks and vector indices.
*   `docs/`: Documentation and diagrams.
*   `tests/`: Unit tests.
