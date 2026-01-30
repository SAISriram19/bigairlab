from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pickle
from rank_bm25 import BM25Okapi
import logging
import os
from modules import config

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, model_name=config.EMBEDDING_MODEL):
        logger.info(f"Loading embedding model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
        self.chunks = []
        self.bm25 = None

        logger.info("VectorStore initialized")

    def create_embeddings(self, chunks):
        self.chunks = chunks
        documents = []
        tokenized_corpus = []

        logger.info("Building BM25 index...")
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk['content'],
                metadata={
                    'page': chunk['page'],
                    'type': chunk['type'],
                    'source': chunk['source'],
                    'chunk_id': i
                }
            )
            documents.append(doc)
            
            # Simple tokenization for BM25
            tokenized_corpus.append(chunk['content'].lower().split())

        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built")

        logger.info("Building FAISS index...")
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        logger.info(f"FAISS index created with {len(documents)} vectors")

    def search(self, query, k=5):
        if self.vectorstore is None:
            logger.warning("Vectorstore not created")
            return []
            
        # 1. Vector Search
        vector_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
        
        # 2. Keyword Search (BM25)
        tokenized_query = query.lower().split()
        if self.bm25:
            bm25_scores = self.bm25.get_scores(tokenized_query)
        else:
            logger.warning("BM25 index not found, skipping keyword search")
            bm25_scores = [0] * len(self.chunks)

        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k*2]
        
        # 3. Combine Results (RRF - Reciprocal Rank Fusion)
        # Create a map of chunk_id -> score
        fused_scores = {}
        
        # Process Vector Results
        for rank, (doc, score) in enumerate(vector_results):
            chunk_id = doc.metadata.get('chunk_id')
            if chunk_id is not None:
                 # Note: FAISS returns distance (lower is better) but we treat rank 0 as best
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + (1 / (rank + 60))
            
        # Process BM25 Results
        for rank, idx in enumerate(top_bm25_indices):
            if idx < len(self.chunks):
                chunk_id = idx # In our case chunk_id corresponds to index in self.chunks
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + (1 / (rank + 60))
            
        # Sort by fused score
        sorted_chunk_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:k]
        
        formatted_results = []
        for rank, chunk_id in enumerate(sorted_chunk_ids):
            chunk = self.chunks[chunk_id]
            formatted_results.append({
                'chunk': chunk,
                'score': fused_scores[chunk_id], # RRF score
                'rank': rank + 1
            })

        return formatted_results

    def save(self, filepath='vector_store'):
        if self.vectorstore is None:
            logger.error("No vectorstore to save")
            return
        self.vectorstore.save_local(filepath)

        # Save chunks and BM25 object
        with open(f"{filepath}_data.pkl", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'bm25': self.bm25
            }, f)
        logger.info(f"Vector store data saved to {filepath}_data.pkl")

    def load(self, filepath='vector_store'):
        logger.info(f"Loading vector store from {filepath}")
        self.vectorstore = FAISS.load_local(
            filepath,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        if os.path.exists(f"{filepath}_data.pkl"):
            with open(f"{filepath}_data.pkl", 'rb') as f:
                data = pickle.load(f)
                self.chunks = data.get('chunks', [])
                self.bm25 = data.get('bm25') # Load pre-built BM25
                logger.info("Loaded chunks and cached BM25 index")
        elif os.path.exists(f"{filepath}_chunks.pkl"): # Legacy fallback
             with open(f"{filepath}_chunks.pkl", 'rb') as f:
                self.chunks = pickle.load(f)
                logger.warning("Loaded legacy chunks file. Rebuilding BM25...")
                tokenized_corpus = [chunk['content'].lower().split() for chunk in self.chunks]
                self.bm25 = BM25Okapi(tokenized_corpus)

    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_chunks = [
        {'content': 'Qatar has strong economic growth', 'page': 1, 'type': 'text', 'source': 'Page 1'},
        {'content': 'Banking sector remains healthy', 'page': 2, 'type': 'text', 'source': 'Page 2'},
        {'content': 'IMF recommendations for fiscal policy', 'page': 3, 'type': 'text', 'source': 'Page 3'}
    ]

    print("Testing LangChain Vector Store...")
    store = VectorStore()
    store.create_embeddings(test_chunks)

    results = store.search("What is Qatar's economic situation?", k=2)
    print(f"\nSearch Results:")
    for result in results:
        print(f"Rank {result['rank']}: {result['chunk']['content'][:50]}... (Score: {result['score']:.3f})")