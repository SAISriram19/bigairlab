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
    """
    Hybrid vector store combining dense embeddings and sparse keyword search.
    
    Features:
    - FAISS for semantic similarity search
    - BM25 for keyword matching
    - Reciprocal Rank Fusion for result combination
    """
    
    def __init__(self, model_name=config.EMBEDDING_MODEL):
        """
        Initialize vector store with embedding model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
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

    def search(self, query, k=config.SEARCH_K):
        """
        Hybrid search combining vector similarity and keyword matching.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of search results with chunks and relevance scores
        """
        if self.vectorstore is None:
            logger.warning("Vectorstore not created")
            return []
            
        # 1. Vector Search (semantic similarity)
        try:
            vector_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            vector_results = []
        
        # 2. Keyword Search (BM25)
        bm25_scores = []
        if self.bm25:
            try:
                tokenized_query = query.lower().split()
                bm25_scores = self.bm25.get_scores(tokenized_query)
            except Exception as e:
                logger.error(f"BM25 search failed: {e}")
                bm25_scores = [0] * len(self.chunks)
        else:
            logger.warning("BM25 index not found, skipping keyword search")
            bm25_scores = [0] * len(self.chunks)

        # Get top BM25 results
        top_bm25_indices = sorted(
            range(len(bm25_scores)), 
            key=lambda i: bm25_scores[i], 
            reverse=True
        )[:k*2]
        
        # 3. Reciprocal Rank Fusion (RRF)
        fused_scores = {}
        rrf_constant = 60  # Standard RRF constant
        
        # Process Vector Results (lower distance = higher relevance)
        for rank, (doc, distance) in enumerate(vector_results):
            chunk_id = doc.metadata.get('chunk_id')
            if chunk_id is not None:
                # Convert distance to similarity-like score
                vector_score = 1 / (rank + rrf_constant)
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + vector_score
            
        # Process BM25 Results (higher score = higher relevance)
        for rank, idx in enumerate(top_bm25_indices):
            if idx < len(self.chunks):
                bm25_score = 1 / (rank + rrf_constant)
                fused_scores[idx] = fused_scores.get(idx, 0) + bm25_score
            
        # Sort by fused score and take top k
        sorted_chunk_ids = sorted(
            fused_scores.keys(), 
            key=lambda x: fused_scores[x], 
            reverse=True
        )[:k]
        
        # Format results
        formatted_results = []
        for rank, chunk_id in enumerate(sorted_chunk_ids):
            if chunk_id < len(self.chunks):
                chunk = self.chunks[chunk_id]
                formatted_results.append({
                    'chunk': chunk,
                    'score': fused_scores[chunk_id],
                    'rank': rank + 1,
                    'chunk_id': chunk_id
                })

        logger.info(f"Search returned {len(formatted_results)} results for query: {query[:50]}...")
        return formatted_results

    def save(self, filepath='vector_store'):
        """
        Save vector store and associated data to disk.
        
        Args:
            filepath: Base path for saving files
        """
        if self.vectorstore is None:
            logger.error("No vectorstore to save")
            return
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save FAISS index
            self.vectorstore.save_local(filepath)
            
            # Save chunks and BM25 as JSON (more robust than pickle)
            import json
            data_file = f"{filepath}_data.json"
            
            # Prepare data for JSON serialization
            save_data = {
                'chunks': self.chunks,
                'bm25_corpus': [chunk['content'].lower().split() for chunk in self.chunks] if self.chunks else []
            }
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Vector store saved to {filepath}")
            logger.info(f"Metadata saved to {data_file}")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise

    def load(self, filepath='vector_store'):
        """
        Load vector store and associated data from disk.
        
        Args:
            filepath: Base path for loading files
        """
        logger.info(f"Loading vector store from {filepath}")
        
        try:
            # Load FAISS index
            self.vectorstore = FAISS.load_local(
                filepath,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Try to load JSON data first (new format)
            json_data_file = f"{filepath}_data.json"
            if os.path.exists(json_data_file):
                import json
                with open(json_data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.chunks = data.get('chunks', [])
                    
                    # Rebuild BM25 from corpus
                    bm25_corpus = data.get('bm25_corpus', [])
                    if bm25_corpus:
                        self.bm25 = BM25Okapi(bm25_corpus)
                    else:
                        logger.warning("No BM25 corpus found, rebuilding from chunks")
                        tokenized_corpus = [chunk['content'].lower().split() for chunk in self.chunks]
                        self.bm25 = BM25Okapi(tokenized_corpus)
                        
                logger.info("Loaded chunks and BM25 index from JSON")
                
            # Fallback to pickle format (legacy)
            elif os.path.exists(f"{filepath}_data.pkl"):
                with open(f"{filepath}_data.pkl", 'rb') as f:
                    data = pickle.load(f)
                    self.chunks = data.get('chunks', [])
                    self.bm25 = data.get('bm25')
                    logger.info("Loaded chunks and BM25 index from pickle (legacy)")
                    
            # Final fallback - rebuild BM25 if only chunks exist
            elif os.path.exists(f"{filepath}_chunks.pkl"):
                with open(f"{filepath}_chunks.pkl", 'rb') as f:
                    self.chunks = pickle.load(f)
                    logger.warning("Loaded legacy chunks file. Rebuilding BM25...")
                    tokenized_corpus = [chunk['content'].lower().split() for chunk in self.chunks]
                    self.bm25 = BM25Okapi(tokenized_corpus)
            else:
                logger.error(f"No data files found for {filepath}")
                raise FileNotFoundError(f"Vector store data not found at {filepath}")
                
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise

    
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