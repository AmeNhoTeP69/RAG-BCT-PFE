import os
import requests
from rag_engine import RAGEngine

# Mock the search to avoid long embedding load if just testing LLM side
# but we want to test the WHOLE thing.

engine = RAGEngine()
print("--- TEST 1: COUSCOUS (OUT OF SCOPE) ---")
print(engine.query("how to cook couscous?")['answer'])

print("\n--- TEST 2: VALID BCT QUERY ---")
print(engine.query("Qu'est-ce qu'une circulaire de la BCT ?")['answer'])
