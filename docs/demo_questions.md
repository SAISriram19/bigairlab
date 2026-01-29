# Demo Questions for Multi-Modal RAG (Qatar IMF Report)

Use these specific questions to show off different capabilities of your architecture during the video.

## 1. Shows "Smart Chunking" & Reasoning
**Question:** *"Describe the fiscal policy recommendations."*
*   **Why:** This requires the model to read a large block of text. Standard chunking might cut the recommendations in half. Your system will retrieve the full context and summarize it effectively.

## 2. Shows "Hybrid Search" (Exact Match)
**Question:** *"What contact information is provided?"*
*   **Why:** This tests exact extraction of entities (Phone numbers, Address). Because of your prompt engineering, it will output a formatted list instead of a blob of text.
*   **Alternative:** *"What does the report say about the Third National Development Strategy?"* (Tests specific Entity Search).

## 3. Shows "Table Understanding" (Crucial)
**Question:** *"What is the projected economic growth for 2024?"*
*   **Why:** This answer lives inside a numerical table. Most RAG systems fail here. Yours will (hopefully) pull the specific percentage because `pdfplumber` preserved the table structure.

## 4. Shows "Synthesis"
**Question:** *"What are the key risks to the economic outlook?"*
*   **Why:** The report discusses risks in multiple sections (Global risks, Oil price risks). This shows the system aggregating information from different parts of the document.
