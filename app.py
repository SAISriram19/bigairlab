import streamlit as st
import os
import shutil
import logging
from modules.vector_store import VectorStore
from modules.llm_qa import LLMQA, SimpleQA
import modules.config as config
from modules.document_processor import DocumentProcessor
import json
import pandas as pd
from modules.evaluate import RAGEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('app')

st.set_page_config(
    page_title="RAG Multi-Modal System",
    layout="wide"
)

# --- Caching Resources ---
@st.cache_resource
def get_llm_qa():
    logger.info("Initializing LLMQA system (Cached)...")
    return LLMQA(model_name=config.LLM_MODEL)

@st.cache_resource
def get_embedding_model_name():
    return config.EMBEDDING_MODEL

# Initialize Session State
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'loaded' not in st.session_state:
    st.session_state.loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

def process_uploaded_file(uploaded_file):
    # Save uploaded file safely
    temp_dir = "temp_upload"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return file_path

def run_ingestion(pdf_path):
    st.session_state.processing = True
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. Processing
        status_text.text("Processing Document... (OCR & Text Extraction)")
        logger.info(f"Starting processing for {pdf_path}")
        
        processor = DocumentProcessor(pdf_path)
        chunks = processor.process_document()
        processor.close()
        
        # Save chunks
        config.create_directories()
        with open(config.CHUNKS_PATH, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
            
        progress_bar.progress(0.4)
        
        # 2. Embedding
        status_text.text("Creating Embeddings... (This may take a moment)")
        
        # Initialize VectorStore (Model load is cached internally by sentence-transformers usually, 
        # but we re-init class. The heavy lift is managed.)
        vector_store = VectorStore(model_name=get_embedding_model_name())
        vector_store.create_embeddings(chunks)
        vector_store.save(config.VECTOR_STORE_PATH)
        
        progress_bar.progress(0.8)
        
        # 3. Loading
        status_text.text("Loading QA System...")
        st.session_state.vector_store = vector_store
        
        try:
            # Use cached LLM loader
            st.session_state.qa_system = get_llm_qa()
        except Exception as e:
            logger.error(f"LLM Load Failed: {e}")
            st.warning("LLM Load Failed, defaulting to SimpleQA")
            st.session_state.qa_system = SimpleQA()
            
        st.session_state.loaded = True
        progress_bar.progress(1.0)
        status_text.success("Processing Complete!")
        logger.info("Ingestion pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        status_text.error(f"Error: {e}")
    finally:
        st.session_state.processing = False

# Sidebar
with st.sidebar:
    st.title("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if uploaded_file:
        # File Size Validation
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > config.MAX_FILE_SIZE_MB:
            st.error(f"File too large! Limit is {config.MAX_FILE_SIZE_MB}MB.")
        else:
            if st.button("Process Document"):
                with st.spinner("Starting Pipeline..."):
                    pdf_path = process_uploaded_file(uploaded_file)
                    run_ingestion(pdf_path)
    
    st.markdown("---")
    
    if st.session_state.loaded:
        st.success("✅ System Ready")
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# Main Interface
st.title("🤖 Multi-Modal RAG Assistant")

tab1, tab2 = st.tabs(["💬 Chat", "📊 Evaluation"])

with tab1:
    if not st.session_state.loaded:
        st.info("👈 Please upload a document to begin.")
    else:
        # Chat Interface
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "citations" in message and message["citations"]:
                    with st.expander("References"):
                        for cite in message["citations"]:
                            st.markdown(f"- **{cite['source']}** ({cite['type']}) - Score: {cite['relevance_score']:.2f}")

        query = st.chat_input("Ask a question about the document...")
        
        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Use vector store search
                    search_results = st.session_state.vector_store.search(query, k=5)
                    
                    # Generate Answer
                    result = st.session_state.qa_system.generate_answer_with_citations(query, search_results)
                    
                    st.markdown(result['answer'])
                    with st.expander("References"):
                        for cite in result['citations']:
                            st.markdown(f"- **{cite['source']}** ({cite['type']}) - Score: {cite['relevance_score']:.2f}")
                            
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "citations": result['citations']
                    })

with tab2:
    st.header("System Evaluation")
    
    if st.button("Run Benchmark"):
        if not st.session_state.loaded:
            st.error("Please load a document first.")
        else:
            with st.spinner("Running Benchmark..."):
                evaluator = RAGEvaluator()
                # Use current loaded system
                evaluator.vector_store = st.session_state.vector_store
                evaluator.qa_system = st.session_state.qa_system
                
                results_df = evaluator.run_benchmark()
                
                st.dataframe(results_df)
                
                avg_latency = results_df["Latency (s)"].mean()
                st.metric("Average Latency", f"{avg_latency:.2f} s")