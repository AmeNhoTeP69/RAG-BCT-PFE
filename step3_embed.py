"""
step3_embed.py
──────────────
STEP 3 — Generate Embeddings

Loads the JSON chunks, passes their text through a multilingual sentence-transformer
to get dense vector representations, and saves them.
"""

import json
import logging
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

def generate_embeddings():
    config.ensure_dirs()
    
    log.info(f"Loading chunks from {config.CHUNKS_PATH}...")
    try:
        with open(config.CHUNKS_PATH, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    except FileNotFoundError:
        log.error(f"Chunks file not found at {config.CHUNKS_PATH}. Run step2_chunk.py first.")
        return

    if not chunks:
        log.warning("Chunks list is empty.")
        return

    texts = [chunk['text'] for chunk in chunks]
    chunk_ids = [chunk['chunk_id'] for chunk in chunks]
    
    log.info(f"Loaded {len(chunks)} chunks.")
    log.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
    
    # Load model
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    
    log.info("Generating embeddings (this may take a few minutes depending on hardware)...")
    # Generate embeddings in batches. show_progress_bar=True provides a nice tqdm bar
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32, convert_to_numpy=True)
    
    # Save embeddings
    log.info(f"Saving embeddings to {config.EMBEDDINGS_PATH}...")
    np.save(config.EMBEDDINGS_PATH, embeddings)
    
    # Save mapping (index in array -> chunk metadata/id)
    # We save the full chunk so we don't have to keep reloading chunks.json in full later if we don't want to,
    # or just saving the mapping is fine. For simplicity, we just save the list of chunks as the index.
    log.info(f"Saving chunk index mapping to {config.CHUNK_INDEX_PATH}...")
    with open(config.CHUNK_INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        
    log.info("Step 3 complete!")

if __name__ == "__main__":
    generate_embeddings()
