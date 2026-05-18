import sys
import os
from rag_engine import RAGEngine
from graph_rag_engine import GraphRAGEngine
import config

def run_tests():
    print("="*50)
    print("BCT RAG FINAL SYSTEM VERIFICATION")
    print("LLM Provider:", config.LLM_PROVIDER)
    print("Model:", config.OLLAMA_MODEL)
    print("="*50)

    rag = RAGEngine()
    grag = GraphRAGEngine()

    test_queries = [
        {
            "id": "Standard RAG Legitimate",
            "query": "Qu'est-ce qu'une circulaire de la BCT ?",
            "engine": rag
        },
        {
            "id": "Graph RAG Connectivity",
            "query": "Quels sont les liens entre la circulaire n°2017-08 et la loi n°2016-48 ?",
            "engine": grag
        },
        {
            "id": "Thematic Intent (No direct keyword)",
            "query": "Comment gérer les risques de blanchiment d'argent ?",
            "engine": rag
        },
        {
            "id": "Out of Scope Safety",
            "query": "Comment cuisiner un couscous tunisien ?",
            "engine": rag
        }
    ]

    for test in test_queries:
        print(f"\n[TEST] {test['id']}")
        print(f"Query: {test['query']}")
        try:
            result = test['engine'].query(test['query'])
            print(f"Answer: {result['answer']}")
        except Exception as e:
            print(f"FAILED: {e}")
        print("-" * 30)

if __name__ == "__main__":
    run_tests()
