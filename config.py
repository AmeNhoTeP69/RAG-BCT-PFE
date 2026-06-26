"""
config.py
─────────
Centralized settings for the RAG and Graph RAG pipeline.
Includes Hybrid RAG settings: LDA, Word2Vec, SMOTE Classifier, Query Reformulation.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# Base Paths
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"

# Step Paths
EXTRACTED_DOCS_PATH = DATA_DIR / "extracted" / "documents.json"
CHUNKS_PATH = DATA_DIR / "chunks" / "chunks.json"

FAISS_DIR = DATA_DIR / "faiss"
EMBEDDINGS_PATH = FAISS_DIR / "embeddings.npy"
CHUNK_INDEX_PATH = FAISS_DIR / "chunk_index.json"
FAISS_INDEX_PATH = FAISS_DIR / "bct_index.faiss"

GRAPH_DIR = DATA_DIR / "graph"
GRAPH_PATH = GRAPH_DIR / "graph.json"
ENTITIES_PATH = GRAPH_DIR / "entities.json"

# ── Hybrid RAG Paths ──────────────────────────────────────────────────────────
TOPICS_DIR = DATA_DIR / "topics"
LDA_MODEL_PATH = TOPICS_DIR / "lda_model"
LDA_DICTIONARY_PATH = TOPICS_DIR / "lda_dictionary.dict"
LDA_TOPICS_PATH = TOPICS_DIR / "topics.json"
W2V_MODEL_PATH = TOPICS_DIR / "word2vec.model"
DOC_TOPICS_PATH = TOPICS_DIR / "doc_topics.json"

CLASSIFIER_DIR = DATA_DIR / "classifier"
CLASSIFIER_MODEL_PATH = CLASSIFIER_DIR / "regulation_classifier.pt"
CLASSIFIER_META_PATH = CLASSIFIER_DIR / "classifier_meta.json"

# Models
# Multilingual transformer model mapping text to 768-dim vectors
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# spaCy NER model for French
SPACY_MODEL = "fr_core_news_lg"

# ── LDA Topic Modeling ────────────────────────────────────────────────────────
LDA_NUM_TOPICS = 8          # Number of latent topics to extract from BCT corpus
LDA_PASSES = 15             # Training passes — more = better quality
LDA_MIN_TOKEN_LEN = 3       # Minimum token length for preprocessing

# Thematic labels assigned manually after inspecting LDA output.
# Each index corresponds to a topic ID (0-7).
LDA_TOPIC_LABELS = [
    "Gouvernance et Fonds Propres",           # établissements, organe, fonds propres, risques
    "Cadre Légal et Réglementaire",           # banques, loi, relative, gouverneur, portant
    "Bureaux de Change",                      # change, billet, bureau, dinars, manuel
    "Systèmes de Paiement",                   # rtgs, elyssa, paiement, chèque, participant
    "Gestion des Risques Opérationnels",      # paiement, système, risques, activité
    "Politique Monétaire et Marchés",         # opérations, taux, contreparties, politique monétaire
    "Crédit et Financement des Entreprises",  # crédit, financement, pme, fonds, ligne
    "Intermédiaires Agréés et Devises",       # devises, agréés, intermédiaires, étranger, change
]

# ── Word2Vec / Similarity ─────────────────────────────────────────────────────
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_WORKERS = 4
W2V_QUERY_EXPANSION_TOPN = 2  # Number of similar words to add to query

# ── SMOTE + AdamW Classifier ─────────────────────────────────────────────────
CLASSIFIER_EPOCHS = 20
CLASSIFIER_LR = 2e-4          # AdamW learning rate
CLASSIFIER_WEIGHT_DECAY = 1e-2 # AdamW weight decay (L2 regularization)
CLASSIFIER_HIDDEN_DIM = 128

# AI Agent Settings (Query Reformulation)
QUERY_REFORMULATION_ENABLED = True
GEMMA_MODEL = "llama-3.1-8b-instant" # Fast model for reformulation
GEMMA_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMMA_API_KEY = os.getenv("GROQ_API_KEY", "")

# Vector Search configs
TOP_K_CHUNKS = 4           # Keep context tight to stay within Groq free-tier TPM budget
MIN_RETRIEVAL_SCORE = 0.60 # Discard chunks with cosine similarity below this threshold

# LLM Config (Using Groq Cloud for high-performance RAG)
LLM_PROVIDER = "openai"  # "ollama", "openai", or "mock"
LLM_TEMPERATURE = 0.0  # Force deterministic answers to prevent hallucinations

# Ollama settings
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Groq (OpenAI Compatible) settings
OPENAI_MODEL = "qwen/qwen3-32b"
OPENAI_API_KEY = os.getenv("GROQ_API_KEY", "")

def ensure_dirs():
    """Ensure all required data directories exist."""
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
