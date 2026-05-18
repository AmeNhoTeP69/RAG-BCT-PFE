"""
step6_topic_modeling.py
────────────────────────────────────────────────────────────────────────────────
STEP 6 — Hybrid Feature Engineering: LDA Topic Modeling + Word2Vec

Architecture:
    Documents (Transformer Embeddings) ──► FAISS Vector Index
    Documents (Bag-of-Words)           ──► Gensim LDA Topics
                                               │
                                               ▼
                                    Knowledge Graph enriched with
                                    Topic Nodes + HAS_TOPIC edges

This script:
  1. Preprocesses the BCT corpus (French tokenization, stopwords removal).
  2. Trains a Gensim LDA model to discover latent regulatory themes.
  3. Trains a Word2Vec model to support Word Similarity queries.
  4. Saves per-document topic distributions (doc_topics.json).
  5. Updates the Knowledge Graph (graph.json) with Topic nodes.

Run after step5_build_graph.py:
    python step6_topic_modeling.py
"""

import json
import logging
import re
import warnings
from pathlib import Path

import numpy as np

# ── Gensim imports ────────────────────────────────────────────────────────────
from gensim import corpora
from gensim.models import LdaModel, Word2Vec
from gensim.models.coherencemodel import CoherenceModel

# ── NetworkX for graph enrichment ─────────────────────────────────────────────
import networkx as nx

import config

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── French Stopwords ──────────────────────────────────────────────────────────
# Core BCT-domain stopwords (NLTK French + regulatory jargon to ignore)
FRENCH_STOPWORDS = set("""
a au aux avec ce ces cet cette dans de des du elle elles en et eu eux je les
leur leurs lui ma mais me mes mon même ni nos notre nous on ou par pas pour
qu que qui sa se ses si son sur ta te tes ton tout toute toutes tous tu un une
vos votre vous y la le l les est sont avoir être fait peuvent doit doivent
peut avoir été sera seront avoir sous dont il ils y a au de du
dans les des est une pour qui que ce par sur avec plus cette dont les bien
banque centrale tunisie bct circulaire note article alinéa paragraphe dispositions
""".split())


def preprocess_text(text: str) -> list:
    """
    Tokenize and clean French regulatory text for LDA.
    Steps:
      - Lowercase
      - Remove digits, punctuation, and special chars
      - Remove short tokens (< LDA_MIN_TOKEN_LEN chars)
      - Remove stopwords
    """
    text = text.lower()
    # Remove numbers and punctuation (keep alphabetic French chars including accents)
    text = re.sub(r"[^a-zàâäéèêëîïôùûüç\s]", " ", text)
    tokens = text.split()
    tokens = [
        t for t in tokens
        if len(t) >= config.LDA_MIN_TOKEN_LEN and t not in FRENCH_STOPWORDS
    ]
    return tokens


def load_documents() -> list:
    log.info(f"Loading documents from {config.EXTRACTED_DOCS_PATH}...")
    with open(config.EXTRACTED_DOCS_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    log.info(f"Loaded {len(docs)} documents.")
    return docs


def train_lda(tokenized_docs: list, dictionary: corpora.Dictionary):
    """Train Gensim LDA model."""
    log.info(f"Building BoW corpus for {len(tokenized_docs)} documents...")
    bow_corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_docs]

    log.info(
        f"Training LDA: {config.LDA_NUM_TOPICS} topics, {config.LDA_PASSES} passes..."
    )
    lda_model = LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=config.LDA_NUM_TOPICS,
        passes=config.LDA_PASSES,
        alpha="auto",       # Asymmetric alpha (better for domain-specific corpora)
        eta="auto",         # Auto-tuned topic-word distribution
        minimum_probability=0.01,
        random_state=42,
    )
    return lda_model, bow_corpus


def compute_coherence(lda_model, tokenized_docs, dictionary):
    """Compute Cv coherence score to validate LDA quality."""
    try:
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence="c_v",
        )
        score = coherence_model.get_coherence()
        log.info(f"LDA Coherence Score (Cv): {score:.4f}  (higher is better, >0.5 is good)")
        return score
    except Exception as e:
        log.warning(f"Could not compute coherence: {e}")
        return None


def train_word2vec(tokenized_docs: list):
    """Train Word2Vec on the BCT corpus for word-level similarity."""
    log.info("Training Word2Vec model for word similarity / query expansion...")
    w2v_model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=config.W2V_VECTOR_SIZE,
        window=config.W2V_WINDOW,
        min_count=config.W2V_MIN_COUNT,
        workers=config.W2V_WORKERS,
        seed=42,
    )
    return w2v_model


def enrich_graph_with_topics(doc_topics: dict):
    """
    Load existing Knowledge Graph and add Topic nodes + HAS_TOPIC edges.
    Each document node gets edges to the topics it belongs to.
    """
    if not config.GRAPH_PATH.exists():
        log.warning("graph.json not found. Skipping graph enrichment. Run step5 first.")
        return

    log.info(f"Loading graph from {config.GRAPH_PATH}...")
    with open(config.GRAPH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    G = nx.node_link_graph(data)

    # Add Topic nodes
    for topic_id, label in enumerate(config.LDA_TOPIC_LABELS):
        node_id = f"topic_{topic_id}"
        G.add_node(node_id, type="topic", label=label, topic_id=topic_id)
        log.info(f"  ► Topic node added: [{node_id}] {label}")

    # Link documents to their dominant topics
    edges_added = 0
    for doc_id, topic_distribution in doc_topics.items():
        if not G.has_node(doc_id):
            continue
        for topic_id, prob in topic_distribution:
            if prob >= 0.10:  # Only link if doc has >= 10% membership in topic
                topic_node_id = f"topic_{topic_id}"
                G.add_edge(
                    doc_id,
                    topic_node_id,
                    relation="HAS_TOPIC",
                    weight=round(float(prob), 4),
                )
                edges_added += 1

    log.info(f"Graph enriched: {edges_added} HAS_TOPIC edges added.")

    # Save enriched graph back
    log.info(f"Saving enriched graph to {config.GRAPH_PATH}...")
    enriched_data = nx.node_link_data(G)
    with open(config.GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, ensure_ascii=False, indent=2)

    log.info(
        f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges."
    )


def format_topics_for_display(lda_model) -> list:
    """Format LDA topics as a human-readable list."""
    topics_out = []
    for i in range(config.LDA_NUM_TOPICS):
        label = config.LDA_TOPIC_LABELS[i] if i < len(config.LDA_TOPIC_LABELS) else f"Topic {i}"
        top_words = [w for w, _ in lda_model.show_topic(i, topn=10)]
        topics_out.append({
            "id": i,
            "label": label,
            "top_words": top_words,
        })
    return topics_out


def run():
    config.ensure_dirs()

    # 1. Load and preprocess documents
    documents = load_documents()
    tokenized_docs = []
    doc_ids = []
    for doc in documents:
        tokens = preprocess_text(doc.get("text", ""))
        if tokens:
            tokenized_docs.append(tokens)
            doc_ids.append(doc["doc_id"])

    log.info(f"Preprocessed {len(tokenized_docs)} documents with valid tokens.")

    # 2. Build Gensim Dictionary
    log.info("Building Gensim dictionary...")
    dictionary = corpora.Dictionary(tokenized_docs)
    # Filter extremes: remove tokens in <2 docs or >80% of docs
    dictionary.filter_extremes(no_below=2, no_above=0.80)
    log.info(f"Dictionary size after filtering: {len(dictionary)} unique tokens.")

    # 3. Train LDA
    lda_model, bow_corpus = train_lda(tokenized_docs, dictionary)

    # 4. Coherence validation
    compute_coherence(lda_model, tokenized_docs, dictionary)

    # 5. Train Word2Vec
    w2v_model = train_word2vec(tokenized_docs)

    # 6. Compute per-document topic distributions
    log.info("Computing per-document topic distributions...")
    doc_topics = {}
    for doc_id, bow in zip(doc_ids, bow_corpus):
        topic_dist = lda_model.get_document_topics(bow, minimum_probability=0.01)
        doc_topics[doc_id] = [(int(t), float(p)) for t, p in topic_dist]

    # 7. Save artifacts
    log.info(f"Saving LDA model to {config.LDA_MODEL_PATH}...")
    lda_model.save(str(config.LDA_MODEL_PATH))

    log.info(f"Saving dictionary to {config.LDA_DICTIONARY_PATH}...")
    dictionary.save(str(config.LDA_DICTIONARY_PATH))

    log.info(f"Saving Word2Vec model to {config.W2V_MODEL_PATH}...")
    w2v_model.save(str(config.W2V_MODEL_PATH))

    log.info(f"Saving doc_topics to {config.DOC_TOPICS_PATH}...")
    with open(config.DOC_TOPICS_PATH, "w", encoding="utf-8") as f:
        json.dump(doc_topics, f, ensure_ascii=False, indent=2)

    # Human-readable topics file
    topics_display = format_topics_for_display(lda_model)
    with open(config.LDA_TOPICS_PATH, "w", encoding="utf-8") as f:
        json.dump(topics_display, f, ensure_ascii=False, indent=2)

    # 8. Display discovered topics
    print("\n" + "="*70)
    print("  BCT REGULATORY TOPICS — Discovered by LDA")
    print("="*70)
    for t in topics_display:
        print(f"\n  Topic {t['id']:02d} │ {t['label']}")
        print(f"           └─ Keywords: {', '.join(t['top_words'][:7])}")

    # Demo: Word2Vec similarity
    print("\n" + "="*70)
    print("  WORD2VEC SIMILARITY DEMO")
    print("="*70)
    test_words = ["banque", "crédit", "blanchiment", "capital"]
    for word in test_words:
        try:
            similar = w2v_model.wv.most_similar(word, topn=3)
            print(f"  Similar to '{word}': {[w for w, _ in similar]}")
        except KeyError:
            print(f"  '{word}' not in Word2Vec vocabulary.")

    # 9. Enrich Knowledge Graph
    enrich_graph_with_topics(doc_topics)

    print("\n✓ Step 6 complete! Hybrid feature engineering done.")
    print(f"  ► LDA model:    {config.LDA_MODEL_PATH}")
    print(f"  ► Word2Vec:     {config.W2V_MODEL_PATH}")
    print(f"  ► Doc topics:   {config.DOC_TOPICS_PATH}")
    print(f"  ► Graph:        {config.GRAPH_PATH} (enriched with Topic nodes)")


if __name__ == "__main__":
    run()
