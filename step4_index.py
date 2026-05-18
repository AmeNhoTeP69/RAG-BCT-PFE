"""
step4_index.py
──────────────
STEP 4 — Build FAISS Index

Loads the generated embeddings and creates a highly optimized FAISS vector search index.
"""

import logging
import numpy as np
import faiss

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

def build_faiss_index():
    config.ensure_dirs()
    
    log.info(f"Loading embeddings from {config.EMBEDDINGS_PATH}...")
    try:
        embeddings = np.load(config.EMBEDDINGS_PATH)
    except FileNotFoundError:
        log.error(f"Embeddings file not found at {config.EMBEDDINGS_PATH}. Run step3_embed.py first.")
        return

    num_vectors, dim = embeddings.shape
    log.info(f"Loaded {num_vectors} vectors of dimension {dim}.")
    
    # We use IndexFlatIP for cosine similarity. 
    # For IP to equal cosine similarity, vectors must be L2 normalized.
    # sentence-transformers outputs aren't strictly L2 normalized by default for all models, 
    # but mpnet works well with IP or we can normalize. Let's L2 normalize just to be safe.
    log.info("L2 normalizing vectors...")
    faiss.normalize_L2(embeddings)
    
    log.info("Building FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    log.info(f"Saving FAISS index to {config.FAISS_INDEX_PATH}...")
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    
    log.info(f"Step 4 complete! Index contains {index.ntotal} vectors.")

if __name__ == "__main__":
    build_faiss_index()
