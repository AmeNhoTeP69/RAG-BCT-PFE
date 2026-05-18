import os
import requests
import json
from rag_engine import RAGEngine

def test():
    engine = RAGEngine()
    
    # Test 1: Out of scope question
    q1 = "how to cook couscous?"
    print(f"\nTesting Q1: {q1}")
    res1 = engine.query(q1)
    print(f"Answer: {res1['answer']}")
    
    # Test 2: In scope question (simulated context or actual retrieval)
    q2 = "Qu'est-ce qu'une circulaire de la BCT ?"
    print(f"\nTesting Q2: {q2}")
    res2 = engine.query(q2)
    print(f"Answer: {res2['answer']}")

if __name__ == "__main__":
    test()
