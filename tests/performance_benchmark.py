import time
import json
import logging
from tabulate import tabulate
from rag_engine import RAGEngine
from graph_rag_engine import GraphRAGEngine
import config

# Configure logging to be less noisy during benchmarks
logging.getLogger("rag_engine").setLevel(logging.WARNING)
logging.getLogger("graph_rag_engine").setLevel(logging.WARNING)

class PerformanceBenchmarker:
    def __init__(self):
        print("Initializing Engines (this may take a moment)...")
        self.rag = RAGEngine()
        self.graph_rag = GraphRAGEngine()
        
        self.test_queries = [
            "Quelles sont les conditions d'ouverture d'un compte?",
            "Explique la circulaire n° 2016-03",
            "Qu'est-ce que le droit de tirage spécial?",
            "Quels sont les documents requis pour un crédit?",
            "Qui est le gouverneur de la BCT?"
        ]

    def benchmark_query(self, engine, query, engine_name):
        start_time = time.time()
        
        # Measure Retrieval
        retrieval_start = time.time()
        retrieved = engine.search(query)
        retrieval_time = (time.time() - retrieval_start) * 1000 # ms
        
        # Measure Generation (Total Query)
        total_start = time.time()
        result = engine.query(query)
        total_time = (time.time() - total_start) * 1000 # ms
        
        # Generation time is approximately total - retrieval
        generation_time = total_time - retrieval_time
        
        return {
            "engine": engine_name,
            "query": query[:30] + "...",
            "retrieval_ms": round(retrieval_time, 2),
            "generation_ms": round(generation_time, 2),
            "total_ms": round(total_time, 2),
            "sources": len(result.get("sources", [])),
            "related_nodes": len(result.get("related_nodes", [])) if "related_nodes" in result else 0
        }

    def run(self):
        print("\n" + "="*60)
        print("BCT RAG PERFORMANCE BENCHMARK")
        print("="*60)
        
        results = []
        for q in self.test_queries:
            print(f"Testing Query: {q}")
            # Standard RAG
            results.append(self.benchmark_query(self.rag, q, "Standard RAG"))
            # Graph RAG
            results.append(self.benchmark_query(self.graph_rag, q, "Graph RAG"))
            print("  - Done.")

        print("\n" + "="*80)
        print(tabulate(results, headers="keys", tablefmt="grid"))
        print("="*80)
        
        # Calculate Averages
        avg_std_retrieval = sum(r['retrieval_ms'] for r in results if r['engine'] == "Standard RAG") / len(self.test_queries)
        avg_graph_retrieval = sum(r['retrieval_ms'] for r in results if r['engine'] == "Graph RAG") / len(self.test_queries)
        avg_std_total = sum(r['total_ms'] for r in results if r['engine'] == "Standard RAG") / len(self.test_queries)
        avg_graph_total = sum(r['total_ms'] for r in results if r['engine'] == "Graph RAG") / len(self.test_queries)

        print(f"\nSummary Averages:")
        print(f"  Standard RAG Retrieval: {avg_std_retrieval:.2f} ms")
        print(f"  Graph RAG Retrieval:    {avg_graph_retrieval:.2f} ms (Overhead: {avg_graph_retrieval - avg_std_retrieval:.2f} ms)")
        print(f"  Standard RAG End-to-End: {avg_std_total/1000:.2f} s")
        print(f"  Graph RAG End-to-End:    {avg_graph_total/1000:.2f} s")

if __name__ == "__main__":
    if config.LLM_PROVIDER == "mock":
        print("WARNING: LLM_PROVIDER is set to 'mock'. Results will not reflect actual LLM generation time.")
    
    # Try tabulate, fallback if not available
    try:
        from tabulate import tabulate
    except ImportError:
        import subprocess
        print("Installing tabulate for better reporting...")
        subprocess.run(["pip", "install", "tabulate"])
        from tabulate import tabulate

    bench = PerformanceBenchmarker()
    bench.run()
