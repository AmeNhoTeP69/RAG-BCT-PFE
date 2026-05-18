import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import config

def check_scores():
    print("Loading model...")
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    print("Loading FAISS index...")
    index = faiss.read_index(str(config.FAISS_INDEX_PATH))
    
    print("Loading chunks...")
    with open(config.CHUNK_INDEX_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    query = "conditions d'ouverture d'un bureau de change"
    query_vec = model.encode([query])
    faiss.normalize_L2(query_vec)
    
    print(f"\nQuery: {query}")
    print("-" * 50)
    
    # 1. FAISS search
    distances, indices = index.search(query_vec, 20)
    
    print("\nTop 20 results from FAISS:")
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            score = float(distances[0][i])
            c = chunks[idx]
            print(f"Rank {i+1} | Score: {score:.4f} | Title: {c['metadata'].get('title')} | Text: {c['text'][:100]}...")
            if '2018 07' in c['metadata'].get('title', ''):
                print(">>> FOUND 2018-07 in top results!")

    # 2. Check if 2018-07 is even in the chunks list
    relevant_count = sum(1 for c in chunks if '2018 07' in c['metadata'].get('title', ''))
    print(f"\nTotal chunks for 2018-07 in index: {relevant_count}")

if __name__ == "__main__":
    check_scores()
