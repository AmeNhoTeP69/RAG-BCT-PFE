# BCT Hybrid Graph RAG — Intelligent Document Search for Tunisian Banking Regulations

> **Final Year Engineering Project (PFE)** — An advanced AI system for querying the official circulars and regulatory notes of the **Banque Centrale de Tunisie (BCT)** using Hybrid Retrieval-Augmented Generation.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)](https://react.dev)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Pipeline Steps](#pipeline-steps)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [API Reference](#api-reference)
- [Evaluation Results](#evaluation-results)
- [Project Structure](#project-structure)

---

## Overview

The BCT publishes hundreds of regulatory circulars and notes as PDFs — many bilingual (French/Arabic). This project builds an end-to-end AI pipeline that:

1. **Scrapes** the BCT website and downloads PDFs (with browser-session spoofing to bypass 503 blocks)
2. **Extracts** text using PyMuPDF and Tesseract OCR (with automatic language detection)
3. **Chunks** documents into semantically coherent sections (article-aware splitting)
4. **Indexes** them with Transformer embeddings (768-dim multilingual model) stored in FAISS
5. **Builds a Knowledge Graph** with SpaCy NER to capture entity relations between circulars
6. **Trains LDA topic models** and Word2Vec for hybrid sparse + dense retrieval
7. **Serves answers** via a FastAPI backend with a React web UI — grounded, cited, hallucination-resistant

### Key Differentiator — Hybrid Graph RAG

Standard RAG only does vector similarity. This system fuses **three retrieval signals**:

| Retrieval Mode | Technology | What it captures |
|---|---|---|
| **Dense** | FAISS + Sentence-Transformers | Deep semantic similarity |
| **Sparse** | Gensim LDA (8 topics) | Thematic category matching |
| **Structural** | NetworkX Knowledge Graph | Cross-document citations & entity links |
| **Query Expansion** | Word2Vec | Synonyms & domain terms |
| **Query Rewriting** | Llama 3.1 (Groq) | Reformulates ambiguous queries |
| **Classification** | PyTorch MLP + SMOTE | Regulatory category prediction |

---

## Architecture

```
BCT Website (bct.gov.tn)
        │
        ▼
  scraper.py ──────────────────────── Downloads PDFs with session cookies & retry
        │
        ▼
[Step 1] step1_extract.py ─────────── PyMuPDF + Tesseract OCR, language detection (fr/ar)
        │
        ▼
[Step 2] step2_chunk.py ───────────── Section-aware chunking (article / title / preamble)
        │
        ▼
[Step 3] step3_embed.py ───────────── paraphrase-multilingual-mpnet-base-v2 → 768-dim vectors
        │
        ▼
[Step 4] step4_index.py ───────────── FAISS cosine similarity index
        │
[Step 5] step5_build_graph.py ─────── SpaCy fr_core_news_lg NER → entities, CITES, MENTIONS
        │
[Step 6] step6_topic_modeling.py ──── LDA (8 topics) + Word2Vec + HAS_TOPIC graph enrichment
        │
[Step 7] step7_optimization_ml.py ─── TF-IDF features + SMOTE + AdamW MLP classifier
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│            HybridGraphRAGEngine (graph_rag_engine.py)       │
│                                                             │
│  User Query                                                 │
│      │                                                      │
│      ▼ [1] Query Reformulation (Llama 3.1 via Groq)         │
│      ▼ [2] Word2Vec Query Expansion                         │
│      ▼ [3] Hybrid Retrieval (FAISS + LDA + Graph)           │
│      ▼ [4] Context Fusion & Re-ranking                      │
│      ▼ [5] LLM Generation (Qwen 32B via Groq)               │
│           Grounded answer + cited sources + graph links     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
                     api.py (FastAPI)
                    ┌──────┴──────┐
             React Web UI    REST API
             (frontend/)   (localhost:8000)
```

---

## Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| **Scraping** | `requests` + `BeautifulSoup` | BCT website PDF download with browser session |
| **PDF Extraction** | `PyMuPDF (fitz)` + `Tesseract OCR` | Text extraction from native and scanned PDFs |
| **Language Detection** | `langdetect` | Auto-detect French / Arabic |
| **Embeddings** | `sentence-transformers` (`paraphrase-multilingual-mpnet-base-v2`) | 768-dim multilingual vectors |
| **Vector DB** | `FAISS` (CPU) | Sub-millisecond cosine similarity search |
| **Knowledge Graph** | `NetworkX` | In-memory entity-relation graph |
| **NER** | `spaCy` (`fr_core_news_lg`) | Named entity extraction for graph building |
| **Topic Modeling** | `Gensim LDA` | 8 regulatory topic clusters |
| **Word Embeddings** | `Gensim Word2Vec` | Domain-specific query expansion |
| **ML Classifier** | `PyTorch MLP` + `SMOTE` (imbalanced-learn) | Regulatory category classification |
| **LLM — Answers** | `Qwen/Qwen3-32B` via Groq | High-quality grounded answer generation |
| **LLM — Rewriting** | `Llama-3.1-8B-Instant` via Groq | Fast query reformulation |
| **Backend** | `FastAPI` + `Uvicorn` | Async REST API + static file serving |
| **Auth** | `SQLite` + bcrypt | JWT-style session token authentication |
| **Frontend** | `React 18` + `Vite` | Single-page chat UI |
| **Cache** | In-process semantic cache (`cache_engine.py`) | Avoid re-embedding identical queries |

---

## Pipeline Steps

| Step | Script | What it does |
|---|---|---|
| 0 | `scraper.py` | Downloads BCT PDFs (browser-session mode, retry+backoff) |
| 1 | `step1_extract.py` | Extracts text from PDFs (PyMuPDF + OCR fallback), detects language |
| 2 | `step2_chunk.py` | Splits documents into article-aware semantic chunks |
| 3 | `step3_embed.py` | Generates 768-dim Transformer embeddings for all chunks |
| 4 | `step4_index.py` | Builds FAISS index for cosine-similarity search |
| 5 | `step5_build_graph.py` | Builds Knowledge Graph: entities, CITES, MENTIONS edges |
| 6 | `step6_topic_modeling.py` | LDA topic model (8 topics) + Word2Vec + graph enrichment |
| 7 | `step7_optimization_ml.py` | TF-IDF + SMOTE + AdamW MLP regulatory classifier |

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed and on `PATH`
- [Groq API key](https://console.groq.com/) (free tier, for LLM calls)
- Node.js 18+ (for the React frontend only)

### 1. Clone & create a virtual environment

```bash
git clone https://github.com/AmeNhoTeP69/RAG-BCT-PFE.git
cd RAG-BCT-PFE
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt

# Download spaCy French model
python -m spacy download fr_core_news_lg

# Optional: multilingual model for better Arabic support
python -m spacy download xx_sent_ud_sm
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> Get a free Groq key at [console.groq.com](https://console.groq.com/). The system falls back gracefully if no key is set (skips query reformulation).

### 4. Run the data pipeline

Place your BCT PDFs in the `bct_documents/` folder (or run `scraper.py` to download them automatically), then:

```bash
python run_project.py
# Select option 1 → Full Pipeline (Extract → Index → Graph → Topics → ML)
```

Or run each step individually:

```bash
python step1_extract.py
python step2_chunk.py
python step3_embed.py
python step4_index.py
python step5_build_graph.py
python step6_topic_modeling.py
python step7_optimization_ml.py
```

### 5. Install and build the frontend (optional)

```bash
cd frontend
npm install
npm run build     # production build, served by FastAPI at /
# or for development:
npm run dev       # Vite dev server on http://localhost:5173
```

---

## Running the Project

```bash
# Master control menu (recommended)
python run_project.py

# Options:
#   1 → Run full data pipeline
#   2 → Generate EDA / analysis plots
#   3 → Launch web UI at http://localhost:8000
#   4 → Interactive CLI demo (terminal chat)
```

Or launch the FastAPI server directly:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://localhost:8000** in your browser.

---

## API Reference

All endpoints (except `/health` and `/api/auth/*`) require an `Authorization: Bearer <token>` header.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/login` | Authenticate, returns session token |
| `POST` | `/api/auth/register` | Register a new user |
| `POST` | `/api/query` | Standard RAG query |
| `POST` | `/api/query/graph` | Hybrid Graph RAG query (recommended) |
| `GET` | `/api/conversations` | List user's conversation threads |
| `POST` | `/api/conversations` | Create a new conversation |
| `GET` | `/api/conversations/{id}/messages` | Get full conversation history |
| `GET` | `/api/analytics/summary` | Query analytics (admin only) |
| `GET` | `/api/analytics/topics` | Topic distribution stats |
| `GET` | `/health` | Health check |

### Example — Graph RAG query

```bash
curl -X POST http://localhost:8000/api/query/graph \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"question": "Quelles sont les exigences de fonds propres pour les banques tunisiennes?", "conversation_id": "abc123"}'
```

**Response includes:** grounded answer, source chunk IDs with circulaire references, related graph nodes, and the LDA topic cluster matched.

---

## Evaluation Results

The system was evaluated on a hand-curated suite of **~80 regulatory questions** across all 8 LDA topic clusters. Ground-truth answers were validated against the original circulars.

| Metric | Standard RAG | Hybrid Graph RAG |
|---|---|---|
| Answer Relevance (human-judged) | ~72% | **~85%** |
| Source Grounding (cited correctly) | ~68% | **~83%** |
| Cross-document relations found | ❌ 0% | **✅ ~61%** |
| Hallucination Rate | ~18% | **~7%** |
| Avg. Latency — Groq P50 | ~1.2 s | ~1.8 s |

> Evaluation scripts: `run_evaluation.py` and `scripts/build_eval_suite.py`.
> Raw results: `eval_results_graph.json` and `eval_questions.json`.

---

## Project Structure

```
bct-rag/
├── scraper.py              # BCT website PDF scraper (session-based)
├── scraper_browser.py      # Playwright-based fallback scraper
│
├── step1_extract.py        # PDF text extraction + OCR
├── step2_chunk.py          # Semantic chunking (article-aware)
├── step3_embed.py          # Transformer embedding generation
├── step4_index.py          # FAISS index construction
├── step5_build_graph.py    # Knowledge graph construction
├── step6_topic_modeling.py # LDA + Word2Vec topic modeling
├── step7_optimization_ml.py# SMOTE + PyTorch MLP classifier
│
├── rag_engine.py           # Standard RAG engine
├── graph_rag_engine.py     # Hybrid Graph RAG engine (main)
├── bm25.py                 # BM25 sparse retrieval (fallback)
├── cache_engine.py         # Semantic answer cache
│
├── api.py                  # FastAPI backend (all REST endpoints)
├── auth.py                 # Authentication & session management
├── database.py             # SQLite persistence layer
├── analytics.py            # Query analytics & logging
├── corrections.py          # User feedback / correction tracking
├── links.py                # Cross-document link resolution
│
├── config.py               # Centralized configuration & paths
├── run_project.py          # Master entry point (menu-driven)
├── requirements.txt        # Python dependencies
│
├── frontend/               # React 18 + Vite web UI
│   ├── src/
│   │   ├── components/     # Chat, Auth, Graph Viewer components
│   │   └── main.jsx
│   └── package.json
│
├── scripts/                # Dev & evaluation utilities
│   ├── build_eval_suite.py # Build evaluation question set
│   ├── demo.py             # CLI demo
│   └── debug_retrieval.py  # Retrieval debugging tool
│
├── tests/                  # Test suite
│   ├── test_suite.py       # Full integration tests
│   ├── test_rag_logic.py   # RAG engine unit tests
│   └── performance_benchmark.py
│
├── utils/
│   ├── text_utils.py       # Arabic/French text normalization
│   └── connectivity.py     # Internet connectivity check
│
├── run_evaluation.py       # Evaluation pipeline runner
├── eval_questions.json     # Curated evaluation question set
├── eval_results_graph.json # Evaluation results (Graph RAG)
│
└── data/                   # Generated by pipeline (gitignored)
    ├── extracted/          # Raw extracted text
    ├── chunks/             # Chunked documents
    ├── faiss/              # FAISS index + embeddings
    ├── graph/              # Knowledge graph files
    ├── topics/             # LDA + Word2Vec models
    └── classifier/         # Trained MLP classifier
```

---

## Design Decisions

**Why not use LangChain or LlamaIndex?**
Built from scratch to have full control over every retrieval step, scoring function, and fusion strategy — and to deeply understand each component for the academic defense.

**Why FAISS instead of a managed vector DB (Pinecone, Weaviate)?**
The corpus (~200–300 BCT documents) fits in RAM. FAISS gives sub-millisecond search with zero infrastructure overhead — perfect for a self-contained demo.

**Why Groq instead of OpenAI?**
Groq's free tier (Qwen-32B) is fast enough for real-time demos and costs nothing. The API is OpenAI-compatible, so switching is a one-line config change.

**Why `LLM_TEMPERATURE = 0.0`?**
Legal/regulatory Q&A demands deterministic, reproducible answers. Non-zero temperature risks creative variation that could be mistaken for actual regulatory content.

---

## Authors

- **Salah-Eddine Sakhi** — Engineering student

---

## License

MIT License — see [LICENSE](LICENSE) for details.
