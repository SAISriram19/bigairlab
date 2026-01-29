import time
import pandas as pd
from vector_store import VectorStore
from llm_qa import LLMQA, SimpleQA
import config

class RAGEvaluator:
    def __init__(self):
        self.vector_store = None
        self.qa_system = None
        self.load_system()

    def load_system(self):
        try:
            self.vector_store = VectorStore(model_name=config.EMBEDDING_MODEL)
            self.vector_store.load(config.VECTOR_STORE_PATH)
            
            try:
                self.qa_system = LLMQA(model_name=config.LLM_MODEL)
            except:
                print("LLM failed to load, using SimpleQA")
                self.qa_system = SimpleQA()
                
        except Exception as e:
            print(f"Error loading system: {e}")

    def run_benchmark(self, queries=None):
        if not queries:
            queries = [
                "What is the projected economic growth for Qatar?",
                "Describe the fiscal policy recommendations.",
                "How is the banking sector performing?",
                "What are the key risks to the outlook?",
                "Summarize the structural reforms mentioned."
            ]

        results = []
        
        print(f"Running benchmark on {len(queries)} queries...")
        
        for q in queries:
            start_time = time.time()
            
            # 1. Retrieval
            search_results = self.vector_store.search(q, k=5)
            
            # 2. Generation
            response = self.qa_system.generate_answer_with_citations(q, search_results)
            
            latency = time.time() - start_time
            
            results.append({
                "Query": q,
                "Latency (s)": round(latency, 2),
                "Sources Found": len(response['citations']),
                "Top Source Type": response['citations'][0]['type'] if response['citations'] else "N/A",
                "Answer Length": len(response['answer'])
            })

        df = pd.DataFrame(results)
        return df

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    results_df = evaluator.run_benchmark()
    print("\nBenchmark Results:")
    print(results_df.to_string())
    
    # Save results
    results_df.to_csv("benchmark_results.csv", index=False)
