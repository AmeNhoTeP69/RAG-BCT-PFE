import sys
import logging
import os

# Add current directory to path
sys.path.append(os.getcwd())

from graph_rag_engine import GraphRAGEngine

# Set up simple logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    print("="*60)
    print(" TESTING HYBRID GRAPH RAG ENGINE ")
    print("="*60)
    
    try:
        print("\n[1] Initializing GraphRAGEngine...")
        engine = GraphRAGEngine()
        print("Initialization successful.")
        
        queries = [
            "Quelles sont les conditions d'ouverture d'un compte en devises pour les exportateurs ?",
            "Comment les circulaires sur le blanchiment d'argent sont-elles liées aux opérations de change ?"
        ]
        
        for i, q in enumerate(queries, 1):
            print(f"\n--- Test Query {i} ---")
            print(f"Q: {q}")
            print("Processing (this may take a moment)...")
            
            result = engine.query(q)
            
            print("\n>> ANSWER:")
            print(result.get("answer", "No answer found."))
            
            print("\n>> RELEVANT DOCUMENTS:")
            sources = result.get("sources", [])
            if not sources:
                print("No sources found.")
            for doc in sources[:2]:
                metadata = doc.get('metadata', {}) if isinstance(doc, dict) else getattr(doc, 'metadata', {})
                print(f" - {metadata.get('source', 'Unknown') if isinstance(metadata, dict) else 'Unknown'}")
            
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
