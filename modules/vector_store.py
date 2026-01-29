from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pickle
from rank_bm25 import BM25Okapi

class VectorStore:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
        self.chunks = []

        print("successfully loaded")

    def create_embeddings(self, chunks):
        self.chunks = chunks
        documents = []
        tokenized_corpus = []

        print("Building BM25 index...")
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
        print("BM25 index built")

        print("Building FAISS index...")
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        print(f"FAISS index with {len(documents)} vectors")

    def search(self, query, k=5):
        if self.vectorstore is None:
            print("Vectorstore not created")
            return []
            
        # 1. Vector Search
        vector_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
        
        # 2. Keyword Search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k*2]
        
        # 3. Combine Results (RRF - Reciprocal Rank Fusion)
        # Create a map of chunk_id -> score
        fused_scores = {}
        
        # Process Vector Results
        for rank, (doc, score) in enumerate(vector_results):
            chunk_id = doc.metadata['chunk_id']
            # Note: FAISS returns distance (lower is better) but we treat rank 0 as best
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + (1 / (rank + 60))
            
        # Process BM25 Results
        for rank, idx in enumerate(top_bm25_indices):
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
            print("No vectorstore to save")
            return
        self.vectorstore.save_local(filepath)

        with open(f"{filepath}_chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)

    def load(self, filepath='vector_store'):
        self.vectorstore = FAISS.load_local(
            filepath,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        with open(f"{filepath}_chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)

        # Rebuild BM25 index
        print("Rebuilding BM25 index...")
        tokenized_corpus = [chunk['content'].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"Loaded vector store chunks")

if __name__ == "__main__":
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