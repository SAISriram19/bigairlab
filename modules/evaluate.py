import time
import pandas as pd
from modules.vector_store import VectorStore
from modules.llm_qa import LLMQA, SimpleQA
from modules import config

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
        
        for i, q in enumerate(queries, 1):
            print(f"Processing query {i}/{len(queries)}: {q[:50]}...")
            
            start_time = time.time()
            
            # 1. Retrieval timing
            retrieval_start = time.time()
            search_results = self.vector_store.search(q, k=5)
            retrieval_time = time.time() - retrieval_start
            
            # 2. Generation timing
            generation_start = time.time()
            response = self.qa_system.generate_answer_with_citations(q, search_results)
            generation_time = time.time() - generation_start
            
            total_latency = time.time() - start_time
            
            results.append({
                "Query": q,
                "Total Latency (s)": round(total_latency, 2),
                "Retrieval Time (s)": round(retrieval_time, 2),
                "Generation Time (s)": round(generation_time, 2),
                "Sources Found": len(response['citations']),
                "Top Source Type": response['citations'][0]['type'] if response['citations'] else "N/A",
                "Answer Length": len(response['answer'])
            })
            
            print(f"  - Retrieval: {retrieval_time:.2f}s, Generation: {generation_time:.2f}s, Total: {total_latency:.2f}s")

        df = pd.DataFrame(results)
        
        # Print summary statistics
        print(f"\nPERFORMANCE SUMMARY")
        print(f"Average Total Latency: {df['Total Latency (s)'].mean():.2f}s")
        print(f"Average Retrieval Time: {df['Retrieval Time (s)'].mean():.2f}s")
        print(f"Average Generation Time: {df['Generation Time (s)'].mean():.2f}s")
        print(f"Retrieval % of total: {(df['Retrieval Time (s)'].mean() / df['Total Latency (s)'].mean() * 100):.1f}%")
        print(f"Generation % of total: {(df['Generation Time (s)'].mean() / df['Total Latency (s)'].mean() * 100):.1f}%")
        
        return df

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    results_df = evaluator.run_benchmark()
    print("\nBenchmark Results:")
    print(results_df.to_string())
    
    # Save results
    results_df.to_csv("benchmark_results.csv", index=False)
