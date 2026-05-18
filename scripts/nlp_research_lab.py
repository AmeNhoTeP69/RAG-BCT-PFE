"""
nlp_research_lab.py
───────────────────
Academic Research Lab: Demonstrating traditional NLP techniques for the PFE report.
Covers: Lemmatization, POS, NER, Stopwords, and Word2Vec.
"""

import json
import spacy
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from pathlib import Path

# Setup NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Configuration
INPUT_FILE = Path("data/extracted/documents.json")
SPACY_MODEL = "fr_core_news_lg"  # Pre-trained French model

def load_text():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [d['text'] for d in data if d.get('language') == 'fr' or 'language' not in d]

def run_classical_nlp():
    texts = load_text()
    if not texts:
        return
    
    sample_text = texts[0][:1000] # Use a sample for demonstration
    
    print("--- 1. Traditional Preprocessing (NLTK) ---")
    stop_words = set(stopwords.words('french'))
    tokens = nltk.word_tokenize(sample_text.lower())
    filtered = [w for w in tokens if w.isalnum() and w not in stop_words]
    print(f"Raw tokens      : {len(tokens)}")
    print(f"Filtered tokens : {len(filtered)}")
    print(f"Sample filtered : {filtered[:10]}")

    print("\n--- 2. Advanced Preprocessing (SpaCy) ---")
    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        print(f"Warning: SpaCy model '{SPACY_MODEL}' not found. Run: python -m spacy download {SPACY_MODEL}")
        return

    doc = nlp(sample_text)
    
    # Lemmatization & POS
    print(f"{'Token':<15} | {'Lemma':<15} | {'POS':<10}")
    print("-" * 45)
    for token in list(doc)[:15]:
        if token.text.strip():
            print(f"{token.text:<15} | {token.lemma_:<15} | {token.pos_:<10}")

    # NER (Named Entity Recognition)
    print("\n--- 3. Entity Extraction (NER) ---")
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    for ent_text, ent_label in entities[:10]:
        print(f"[{ent_label}] {ent_text}")

    print("\n--- 4. Classical Embeddings (Gensim Word2Vec) ---")
    # Tokenize all documents for training
    print("Training Word2Vec on full corpus (this may take a moment)...")
    corpus_tokens = [nltk.word_tokenize(t.lower()) for t in texts]
    model = Word2Vec(sentences=corpus_tokens, vector_size=100, window=5, min_count=2, workers=4)
    
    # Showcase similarity
    test_word = "banque"
    if test_word in model.wv:
        print(f"Words most similar to '{test_word}' in Word2Vec:")
        for word, score in model.wv.most_similar(test_word, topn=5):
            print(f"  - {word}: {score:.4f}")
    
    print("\nLab Analysis Complete. Use these results in your PFE report to compare with RAG.")

if __name__ == "__main__":
    run_classical_nlp()
