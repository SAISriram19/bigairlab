# Multi-Modal RAG System Architecture

## 1. High-Level Component Interaction
This diagram shows how the Python files interact with each other and the external libraries.

```mermaid
graph TD
    User((User)) <--> UI[app.py<br/>Streamlit Interface]
    
    subgraph "Core Modules"
        UI --> DP[document_processor.py<br/>Data Extraction]
        UI --> VS[vector_store.py<br/>Retrieval Engine]
        UI --> QA[llm_qa.py<br/>Generation Engine]
        EV[evaluate.py<br/>Benchmarking] -.-> VS
        EV -.-> QA
    end

    subgraph "Storage & Models"
        FAISS[(FAISS Index<br/>Vector DB)]
        BM25[(BM25 Index<br/>Keyword DB)]
        Models[HuggingFace Models<br/>Embeddings + LLM]
    end

    DP -->|Raw Chunks| VS
    VS <-->|Read/Write| FAISS
    VS <-->|Read/Write| BM25
    VS -->|Embeddings| Models
    QA -->|Inference| Models
```

## 2. Data Ingestion Pipeline
How a PDF document is processed into searchable vectors.

```mermaid
sequenceDiagram
    participant U as User
    participant App as app.py
    participant DP as document_processor.py
    participant VS as vector_store.py
    participant DB as Disk Storage

    U->>App: Upload PDF
    App->>DP: process_document(pdf_path)
    
    rect rgb(240, 248, 255)
        note right of DP: Extraction Phase
        DP->>DP: PyMuPDF (Text & Images)
        DP->>DP: pdfplumber (Tables)
        DP->>DP: pytesseract (OCR)
    end
    
    rect rgb(255, 240, 245)
        note right of DP: Chunking Phase
        DP->>DP: RecursiveCharacterTextSplitter
        DP-->>App: List[Chunks]
    end

    App->>VS: create_embeddings(chunks)
    
    rect rgb(240, 255, 240)
        note right of VS: Indexing Phase
        VS->>VS: Generate Embeddings (MiniLM)
        VS->>VS: Build FAISS Index (Dense)
        VS->>VS: Build BM25 Index (Sparse)
    end
    
    VS->>DB: Save Index & Metadata
    VS-->>App: Success
    App-->>U: "System Ready"
```

## 3. Query & Retrieval Flow (RAG)
How the system answers a user question.

```mermaid
flowchart TD
    Q[User Query] -->|Input| App[app.py]
    App -->|Pass Query| QA[llm_qa.py]
    
    subgraph "Retrieval (Hybrid)"
        QA -->|Search| VS[vector_store.py]
        VS -->|Dense Search| FAISS[FAISS]
        VS -->|Keyword Search| BM25[BM25]
        FAISS -->|Top K| Results
        BM25 -->|Top K| Results
        Results -->|RRF Fusion| Ranked[Ranked Candidates]
    end
    
    subgraph "Reranking & Context"
        Ranked -->|Top Candidates| Cross[Cross-Encoder]
        Cross -->|Re-score| Final[Top 3 Context Chunks]
    end
    
    subgraph "Generation"
        Final -->|Context + Prompt| LLM[Flan-T5 Model]
        LLM -->|Generate| Ans[Answer]
    end
    
    Ans -->|Display| App
```
