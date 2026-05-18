"""
step5_build_graph.py
────────────────────
STEP 5 — Build Knowledge Graph

Extracts entities (Organizations, Dates, Regulations) and relationships (CITES, MENTIONS)
to build a structured knowledge graph of the BCT regulatory landscape.
"""

import json
import logging
import re
import spacy
import networkx as nx
from pathlib import Path
from tqdm import tqdm

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Regex for detecting cross-references to other circulaires
# Matches formats like: "circulaire n° 2016-03", "note n°2022-10", etc.
REF_PATTERN = re.compile(r"(?i)(circulaire|note|décision)\s+(?:n°\s*)?(\d{4}[-/]\d{2,4})")

def build_graph():
    config.ensure_dirs()
    
    log.info(f"Loading documents from {config.EXTRACTED_DOCS_PATH}...")
    try:
        with open(config.EXTRACTED_DOCS_PATH, 'r', encoding='utf-8') as f:
            documents = json.load(f)
    except FileNotFoundError:
        log.error(f"Documents file not found. Run step1_extract.py first.")
        return

    log.info(f"Loading spaCy model: {config.SPACY_MODEL}...")
    try:
        nlp = spacy.load(config.SPACY_MODEL)
    except OSError:
        log.warning(f"spaCy model {config.SPACY_MODEL} not found. Downloading...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", config.SPACY_MODEL], check=True)
        nlp = spacy.load(config.SPACY_MODEL)

    G = nx.MultiDiGraph()
    entities_data = {}

    log.info("Extracting entities and relationships from documents...")
    for doc in tqdm(documents):
        doc_id = doc['doc_id']
        text = doc['text']
        title = doc['title']
        
        # Add Document node
        G.add_node(doc_id, type="document", label=title, filename=doc['filename'], year=doc['year'])
        
        # 1. Extract cross-references (Relationships between documents)
        refs = REF_PATTERN.findall(text)
        for ref_type, ref_val in refs:
            # We don't have a perfect doc_id mapping for all refs yet, 
            # so we add a generic "reference" node or try to match existing docs.
            ref_label = f"{ref_type} {ref_val}".strip()
            G.add_node(ref_label, type="reference", label=ref_label)
            G.add_edge(doc_id, ref_label, relation="CITES")

        # 2. Extract Entities using spaCy (NER)
        # We process only the first 10k chars to keep it fast, or the whole text?
        # Let's do snippets or key sections if possible. For now, first 5000 chars.
        doc_nlp = nlp(text[:5000])
        
        for ent in doc_nlp.ents:
            # Filter for interesting entities
            if ent.label_ in ["ORG", "LOC", "DATE", "PER"]:
                ent_text = ent.text.strip()
                if len(ent_text) < 3: continue
                
                ent_id = f"ent_{ent.label_}_{ent_text.lower().replace(' ', '_')}"
                
                if ent_id not in entities_data:
                    entities_data[ent_id] = {
                        "id": ent_id,
                        "text": ent_text,
                        "label": ent.label_
                    }
                    G.add_node(ent_id, type="entity", label=ent_text, category=ent.label_)
                
                G.add_edge(doc_id, ent_id, relation="MENTIONS")

    # Save Graph
    log.info(f"Saving graph to {config.GRAPH_PATH}...")
    # NetworkX to dict for JSON serialization
    data = nx.node_link_data(G)
    with open(config.GRAPH_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    log.info(f"Saving entity registry to {config.ENTITIES_PATH}...")
    with open(config.ENTITIES_PATH, 'w', encoding='utf-8') as f:
        json.dump(entities_data, f, ensure_ascii=False, indent=2)

    log.info(f"Step 5 complete! Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

if __name__ == "__main__":
    build_graph()
