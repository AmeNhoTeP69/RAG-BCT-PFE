# Automated Analysis of Regulatory Documents of the Central Bank of Tunisia Using a Hybrid Graph RAG System

> **End of Studies Project (PFE) — École Polytechnique Internationale (EPI Digital)**  
> Internship at **Skillia** · Academic year 2025–2026  
> Author: **Chouchane Mohamed Anwar** · Supervised by: **Mrs. Boutheina Ben Ismail** (academic) & **Mr. Aymen Chakhari** (Skillia CEO)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)](https://react.dev)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Pipeline — 7 Steps](#pipeline--7-steps)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [API Reference](#api-reference)
- [Evaluation Results](#evaluation-results)
- [Expert-in-the-Loop: Verified Corrections](#expert-in-the-loop-verified-corrections)
- [Project Structure](#project-structure)
- [Design Decisions](#design-decisions)
- [Future Work](#future-work)

---

## Overview

The **Banque Centrale de Tunisie (BCT)** publishes hundreds of regulatory circulars and notes — spanning French and Arabic, accumulating since 2016 at over 40 documents per year — that compliance officers and banking lawyers must navigate daily. Searching this body of text by hand is both slow and unreliable: provisions are spread across documents written years apart, in two languages, that constantly reference, amend, and revoke one another.

This project builds and deploys a **Hybrid Graph RAG** (Retrieval-Augmented Generation) system tailored specifically to the BCT regulatory corpus. Three retrieval paradigms work in concert:

| Retrieval Paradigm | Technology | What it captures |
|---|---|---|
| **Dense vector search** | FAISS + Sentence-Transformers (768-dim) | Fine-grained semantic similarity |
| **Sparse topic modelling** | Gensim LDA (8 regulatory topics) | Thematic category routing |
| **Knowledge graph traversal** | SpaCy NER + NetworkX MultiDiGraph | Citation & entity links between documents |
| **Query expansion** | Gensim Word2Vec (100-dim, corpus-trained) | Domain synonyms (e.g. "AML" ↔ "blanchiment") |
| **Query reformulation** | Llama-3.1-8B-Instant via Groq | Formalises ambiguous questions before search |
| **Regulatory classification** | PyTorch MLP + SMOTE on TF-IDF features | Labels chunks by regulatory category |

---

## Problem Statement

Four concrete problems motivate this project:

1. **Volume** — 445 regulatory texts (113 circulars + 312 notes + 20 other) from 2016–2026; sequential reading is not viable.
2. **Cross-references** — A recent circular often modifies or repeals earlier ones. Tracing that relational chain is beyond keyword search.
3. **Bilinguality** — 63% Arabic (279 docs) / 37% French (163 docs). A system handling only one language misses half the corpus.
4. **Hallucination risk** — An LLM queried without grounding can produce plausible but incorrect regulatory statements, with real compliance consequences.

---

## Architecture

```
BCT Portal (bct.gov.tn)
        │
        ▼
  scraper.py ──────────────────── PDF download with browser session, retry & deduplication
        │
        ▼
[Step 1] step1_extract.py ─────── PyMuPDF native text extraction; Tesseract OCR fallback
        │                          Language detection (fr / ar) via langdetect
        ▼
[Step 2] step2_chunk.py ───────── Section-aware chunking — article/title/preamble boundaries
        │                          Target: 400 words · Max: 600 · Min: 50 · Overlap: 75 words
        ▼                          → 2,058 chunks from 445 documents (avg 185 words/chunk)
[Step 3] step3_embed.py ───────── paraphrase-multilingual-mpnet-base-v2 → 768-dim vectors
        │                          278M parameters · 50 languages · optimised for semantic similarity
        ▼
[Step 4] step4_index.py ───────── FAISS IndexFlatIP (exact search) + L2 normalisation
        │                          = exact cosine similarity · <5ms per query · fits in RAM
[Step 5] step5_build_graph.py ─── SpaCy fr_core_news_lg NER on first 5,000 chars per doc
        │                          → MultiDiGraph: 3,166 nodes · 7,052 edges
        │                            192 CITES · 6,204 MENTIONS · 656 HAS_TOPIC
[Step 6] step6_topic_modeling.py ─ LDA 8 topics (Gensim) · 15 passes · min token length 3
        │                          Word2Vec 100-dim · window 5 · min freq 2
[Step 7] step7_optimization_ml.py  TF-IDF 2,000 features + SMOTE + AdamW MLP
        │                          Accuracy: 75.9% · Macro-F1: 0.669 (5-fold CV)
        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    HybridGraphRAGEngine (graph_rag_engine.py)            │
│                                                                          │
│  User query                                                              │
│      │                                                                   │
│      ▼ [1] Off-topic guard — raw query FAISS score < 0.50 → refuse       │
│      ▼ [2] Query reformulation — Llama-3.1-8B (Groq); expands acronyms  │
│      ▼ [3a] Dense retrieval — Word2Vec expansion + FAISS (top-k=4)      │
│      ▼ [3b] Graph traversal — CITES + MENTIONS entity matching           │
│      ▼ [3c] Topic retrieval — LDA HAS_TOPIC edges                        │
│      ▼ [4] Context fusion — score-annotated, de-duplicated evidence      │
│      ▼ [5] LLM generation — Qwen3-32B (Groq online) / Llama3.2:1B       │
│             temperature=0 · strict grounding · source citation enforced  │
└──────────────────────────────────────┬───────────────────────────────────┘
                                       ▼
                               api.py (FastAPI)
                              ┌────────┴────────┐
                       React Web UI        REST API
                       (frontend/)    (localhost:8000)
                              └────────┬────────┘
                                  SQLite DB
                          (users · conversations · messages
                           qa_cache · query_logs · corrections)
```

---

## Tech Stack

| Category | Technology | Detail |
|---|---|---|
| **Scraping** | `requests` + `BeautifulSoup` | Browser-session headers, retry with backoff, content-hash deduplication |
| **PDF Extraction** | `PyMuPDF (fitz)` | Native text extraction with preserved reading order |
| **OCR Fallback** | `Tesseract OCR` + `pdf2image` | For scanned pages; configured for French + Arabic |
| **Language Detection** | `langdetect` | Auto-labels each document `fr` or `ar` |
| **Embeddings** | `sentence-transformers` — `paraphrase-multilingual-mpnet-base-v2` | 768-dim, 50 languages, 278M params |
| **Vector Index** | `FAISS` — `IndexFlatIP` | Exact cosine search (via L2 norm), <5ms, fully in-RAM |
| **Knowledge Graph** | `NetworkX` — `MultiDiGraph` | 3,166 nodes, 7,052 edges (CITES / MENTIONS / HAS_TOPIC) |
| **NER** | `spaCy` — `fr_core_news_lg` | Entity extraction for graph construction |
| **Topic Modelling** | `Gensim LDA` | 8 topics, 15 passes, trained on full BCT corpus |
| **Word Embeddings** | `Gensim Word2Vec` | 100-dim, window=5, domain-specific query expansion |
| **ML Classifier** | `PyTorch MLP` + `imbalanced-learn SMOTE` | TF-IDF 2,000 features, hidden=128+64, AdamW lr=2e-4 |
| **LLM — Answers** | `qwen/qwen3-32b` via Groq | Online mode; temperature=0; 6,000 TPM free tier |
| **LLM — Local fallback** | `llama3.2:1b` via Ollama | Offline mode; no data leaves the institution |
| **LLM — Reformulation** | `llama-3.1-8b-instant` via Groq | Query rewriting and acronym expansion |
| **Backend** | `FastAPI` + `Uvicorn` | Async REST API; lazy model loading |
| **Auth** | PBKDF2-HMAC-SHA256 + HMAC-signed bearer token | Standard-library only; no JWT/bcrypt dependency; 8h expiry |
| **Database** | `SQLite` (stdlib `sqlite3`) | Single file `bct_app.db`; 6 tables; no external server |
| **Frontend** | `React 18` + `Vite` | SPA served as static files by FastAPI |
| **Semantic Cache** | In-process cosine matching on `qa_cache` | Paraphrase-aware; never caches refusals |

---

## Pipeline — 7 Steps

| Step | Script | What it does | Output artefact |
|---|---|---|---|
| 0 | `scraper.py` | Crawls BCT portal, downloads PDFs (session cookies, retry, dedup) | `bct_documents/` |
| 1 | `step1_extract.py` | PyMuPDF text extraction + Tesseract OCR fallback; language detection | `data/extracted/documents.json` |
| 2 | `step2_chunk.py` | Section-aware splitting (article/chapter boundaries); 400w target, 75w overlap | `data/chunks/chunks.json` — 2,058 chunks |
| 3 | `step3_embed.py` | Encodes all chunks with multilingual Sentence-Transformer (768-dim) | `data/faiss/embeddings.npy` |
| 4 | `step4_index.py` | Builds FAISS `IndexFlatIP` with L2-normalised embeddings | `data/faiss/bct_index.faiss` |
| 5 | `step5_build_graph.py` | SpaCy NER → entities; regex → citation links; NetworkX MultiDiGraph | `data/graph/graph.json` · `entities.json` |
| 6 | `step6_topic_modeling.py` | LDA (8 topics, 15 passes) + Word2Vec (100-dim) + HAS_TOPIC graph enrichment | `data/topics/lda_model` · `word2vec.model` |
| 7 | `step7_optimization_ml.py` | TF-IDF features + SMOTE + AdamW MLP regulatory classifier | `data/classifier/regulation_classifier.pt` |

### LDA Regulatory Topics

| ID | Topic | Representative keywords |
|---|---|---|
| 0 | Governance and Capital Adequacy | institutions, governance body, own funds, risks |
| 1 | Legal and Regulatory Framework | banks, law, relating to, governor, concerning |
| 2 | Foreign Exchange Bureaux | exchange, banknote, bureau, dinars, manual |
| 3 | Payment Systems | RTGS, Elyssa, payment, cheque, participant |
| 4 | Operational Risk Management | payment, system, risks, activity |
| 5 | Monetary Policy and Markets | operations, rate, counterparties, monetary policy |
| 6 | Credit and Business Financing | credit, financing, SMEs, funds, credit line |
| 7 | Authorised Intermediaries and FX | foreign currency, authorised, intermediaries, abroad |

> Topic 1 (Legal & Regulatory Framework) accounts for ~25% of classified documents; Topic 5 (Monetary Policy) just 2.6% — the imbalance SMOTE is designed to correct.

### Knowledge Graph Statistics

| Element | Count |
|---|---|
| Total nodes | 3,166 |
| Document nodes | 445 |
| Entity nodes | 2,644 |
| Reference nodes | 69 |
| Topic nodes | 8 |
| Total edges | 7,052 |
| CITES edges | 192 |
| MENTIONS edges | 6,204 |
| HAS_TOPIC edges | 656 |

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) on `PATH` (with French + Arabic language packs)
- [Groq API key](https://console.groq.com/) — free tier (6,000 tokens/min, 500,000/day)
- [Ollama](https://ollama.com) (for offline mode) with `llama3.2:1b` pulled
- Node.js 18+ (for the React frontend)

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

# Required spaCy model (French NER for knowledge graph)
python -m spacy download fr_core_news_lg

# Optional: multilingual model for better Arabic support
python -m spacy download xx_sent_ud_sm
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> Without a key, query reformulation is skipped (original question used directly). All other features remain functional.

### 4. Run the data pipeline

Place BCT PDFs in `bct_documents/` (or run `python scraper.py` to download automatically), then:

```bash
python run_project.py
# Option 1 → Full pipeline (Extract → Index → Graph → Topics → ML classifier)
```

Or run each step independently:

```bash
python step1_extract.py
python step2_chunk.py
python step3_embed.py
python step4_index.py
python step5_build_graph.py
python step6_topic_modeling.py
python step7_optimization_ml.py
```

### 5. Build the frontend (optional)

```bash
cd frontend
npm install
npm run build        # Production build served by FastAPI at /
# or:
npm run dev          # Vite dev server on http://localhost:5173
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

Or start the API directly:

```bash
# Online mode (Groq — qwen3-32b)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# Offline mode (Ollama — llama3.2:1b, no internet required)
ollama serve &
python run_project.py  # Option 3
```

Open **http://localhost:8000** in your browser. The interface offers two query modes:
- **Standard RAG** — FAISS vector search only
- **Hybrid Graph RAG** — full pipeline with graph traversal + LDA topic routing

---

## API Reference

All endpoints except `/health` and `/api/auth/*` require `Authorization: Bearer <token>`.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/login` | Authenticate; returns HMAC-signed bearer token (8h expiry) |
| `POST` | `/api/auth/register` | Register a new user |
| `POST` | `/api/query` | Query engine — mode `rag` or `graph` |
| `GET` | `/api/conversations` | List user's conversation threads |
| `POST` | `/api/conversations` | Create a new conversation |
| `GET` | `/api/conversations/{id}/messages` | Get conversation history |
| `GET` | `/api/analytics/summary` | Usage KPIs — admin only |
| `GET` | `/api/analytics/topics` | Topic distribution stats |
| `GET` | `/health` | Health check |
| `GET` | `/` | Serve React (Vite) web interface |

### Example — Hybrid Graph RAG query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "question": "Quelles sont les obligations des banques concernant les transactions suspectes?",
    "mode": "graph",
    "conversation_id": "abc123"
  }'
```

**Response includes:** grounded answer in user's language, source chunk IDs (exact circular title + article), `related_nodes` (graph-linked documents), and the reformulated query.

**System configuration parameters:**

| Parameter | Value |
|---|---|
| `TOP_K_CHUNKS` | 4 |
| `MIN_RETRIEVAL_SCORE` | 0.60 |
| `LLM_TEMPERATURE` | 0.0 |
| `LDA_NUM_TOPICS` | 8 |
| `LDA_PASSES` | 15 |
| `W2V_VECTOR_SIZE` | 100 |
| `W2V_WINDOW` | 5 |
| `CLASSIFIER_EPOCHS` | 20 |
| `CLASSIFIER_LR` | 2×10⁻⁴ |

---

## Evaluation Results

### Corpus Statistics

| Indicator | Value |
|---|---|
| Total documents | 445 |
| Circulars / Notes / Other | 113 / 312 / 20 |
| Arabic-language documents | 279 (~63%) |
| French-language documents | 163 (~37%) |
| Period covered | 2016–2026 |
| Total chunks (after Step 2) | **2,058** |
| Average chunk size | ~185 words |

### LLM-as-a-Judge — 30-question benchmark (scores out of 3)

Evaluation performed by `qwen/qwen3-32b` at temperature 0. Each response scored on Factual Accuracy, Source Fidelity, and Completeness.

| Variant | Factual Accuracy | Source Fidelity | Completeness | **Composite (0–1)** |
|---|---|---|---|---|
| Standard RAG | 2.1 (±0.6) | 2.4 (±0.5) | 1.8 (±0.7) | 0.68 (±0.09) |
| Graph RAG (FAISS + Graph) | 2.4 (±0.5) | 2.6 (±0.4) | 2.2 (±0.6) | 0.77 (±0.08) |
| **Hybrid Graph RAG (full)** | **2.7 (±0.4)** | **2.8 (±0.3)** | **2.6 (±0.5)** | **0.87 (±0.06)** |

> Hybrid Graph RAG achieves a **28% relative improvement** over Standard RAG, with the largest gain on cross-document questions (+0.7 factual accuracy).

### Retrieval Metrics — Ablation Study

| Variant | P@4 | MRR | Off-topic refusal rate |
|---|---|---|---|
| Standard RAG (FAISS only) | 68% | 0.74 | 70% |
| Graph RAG (FAISS + Graph) | 79% | 0.83 | 85% |
| **Hybrid Graph RAG (full)** | **88%** | **0.91** | **100%** |

> Adding the knowledge graph lifts P@4 by 11 points; adding LDA + Word2Vec adds 9 more.  
> The off-topic guard (applied to the raw query before reformulation) achieves a perfect 100% refusal rate.

### End-to-End Pass/Fail — 80-question bilingual benchmark (French, Arabic, English)

| Configuration | Passed / Total | Success rate |
|---|---|---|
| Bare Hybrid Graph RAG | 67 / 80 | 83.8% |
| **+ Verified Corrections KB** | **74 / 80** | **92.5%** |

**Breakdown by expected behaviour (with verified KB):**

| Category | Passed / Total | Rate |
|---|---|---|
| Answerable from corpus | 58 / 64 | 90.6% |
| Off-topic (must refuse) | 8 / 8 | **100%** |
| Absent ("not found") | 8 / 8 | **100%** |
| **Overall** | **74 / 80** | **92.5%** |

**Breakdown by language:**

| Language | Passed / Total | Rate |
|---|---|---|
| Arabic | 8 / 8 | **100%** |
| French | 65 / 70 | 92.9% |
| English | 1 / 2 | 50% (n=2, not statistically significant) |

> **Most important result: zero hallucinations across the entire run.** Every one of the 6 remaining failures is a *safe refusal* — the system declined and acknowledged it lacked the information, rather than inventing a regulatory provision.

### Regulatory Classifier (Step 7)

| Configuration | Accuracy | Macro-F1 |
|---|---|---|
| Without SMOTE | 74.7% | 0.640 |
| **With SMOTE (deployed)** | **75.9%** | **0.669** |

> 5-fold stratified cross-validation on real (non-synthetic) documents. The macro-F1 gain reflects improved performance on minority regulatory categories (Microfinance, Financial Markets).

---

## Expert-in-the-Loop: Verified Corrections

One of the project's distinctive contributions: a curated layer of human-approved facts that sits on top of retrieval.

**How it works:**
1. When the system is wrong and a user provides a correction in discussion, the concession is logged to the `corrections` table with status `PENDING`
2. A human administrator reviews and either approves or rejects it — nothing automated
3. Only approved corrections enter the knowledge base — each is a **verbatim quote from the BCT corpus**, not LLM-generated text
4. At query time, the question is cosine-matched against approved correction topics (threshold: 0.62). If matched, the correction is injected as a `VERIFIED NOTES` block ahead of generation
5. The semantic cache is bypassed when a correction applies, so the answer is always generated fresh

**Evaluation used 15 approved corrections** spanning circulars 2016-06, 2018-06, 2021-03, 2017-08, 91-24, and 2017-02.

**Impact:** +8.7 percentage points on the end-to-end pass/fail benchmark (83.8% → 92.5%).

---

## Project Structure

```
RAG-BCT-PFE/
│
├── scraper.py              # BCT portal PDF scraper (session-based, retry, dedup)
├── scraper_browser.py      # Playwright-based fallback scraper
│
├── step1_extract.py        # PDF text extraction + Tesseract OCR + language detection
├── step2_chunk.py          # Section-aware semantic chunking (article-boundary aware)
├── step3_embed.py          # Sentence-Transformer embedding generation
├── step4_index.py          # FAISS IndexFlatIP construction
├── step5_build_graph.py    # Knowledge graph: SpaCy NER + citation regex + NetworkX
├── step6_topic_modeling.py # LDA topic model + Word2Vec + HAS_TOPIC graph enrichment
├── step7_optimization_ml.py# TF-IDF + SMOTE + PyTorch MLP regulatory classifier
│
├── rag_engine.py           # Standard RAG engine (FAISS only)
├── graph_rag_engine.py     # Hybrid Graph RAG engine (main — all 5 runtime stages)
├── bm25.py                 # BM25 sparse retrieval (fallback)
├── cache_engine.py         # Semantic answer cache (cosine-matched, never caches refusals)
│
├── api.py                  # FastAPI backend — all REST endpoints + static file serving
├── auth.py                 # PBKDF2 password hashing + HMAC bearer tokens
├── database.py             # SQLite persistence (6 tables: users, conversations, messages,
│                           #   qa_cache, query_logs, corrections)
├── analytics.py            # Query analytics KPIs + 5 charts for admin dashboard
├── corrections.py          # Expert-in-the-loop corrections (PENDING → approved)
├── links.py                # Cross-document link resolution
│
├── config.py               # Centralized configuration + all file paths
├── run_project.py          # Master entry point (interactive menu)
├── requirements.txt        # Python dependencies
│
├── frontend/               # React 18 + Vite web UI
│   ├── src/
│   │   ├── components/     # Chat, Auth, GraphViewer, Analytics components
│   │   └── main.jsx
│   └── package.json
│
├── scripts/                # Development & evaluation utilities
│   ├── build_eval_suite.py # Build the evaluation question bank
│   ├── demo.py             # Interactive CLI demo
│   └── debug_retrieval.py  # Retrieval debugging tool
│
├── tests/                  # Test suite
│   ├── test_suite.py       # Full integration tests (80-question pass/fail runner)
│   ├── test_rag_logic.py   # RAG engine unit tests
│   └── performance_benchmark.py
│
├── utils/
│   ├── text_utils.py       # Arabic/French text normalisation
│   └── connectivity.py     # Internet connectivity check (online/offline routing)
│
├── run_evaluation.py       # LLM-as-a-Judge evaluation pipeline
├── eval_questions.json     # 30-question benchmark dataset (with reference answers)
├── eval_results_graph.json # Evaluation results — Hybrid Graph RAG
│
└── data/                   # Generated by pipeline (gitignored)
    ├── extracted/          # Raw extracted text (documents.json)
    ├── chunks/             # Chunked corpus (chunks.json — 2,058 chunks)
    ├── faiss/              # FAISS index + embeddings (768-dim)
    ├── graph/              # Knowledge graph (graph.json + entities.json)
    ├── topics/             # LDA model + Word2Vec model
    └── classifier/         # Trained MLP classifier (.pt)
```

---

## Design Decisions

**Why build from scratch instead of LangChain/LlamaIndex?**  
Full control over every retrieval step, scoring function, and fusion strategy — and to deeply understand each component for the academic defence. Every design choice is justified in the report rather than hidden behind a framework abstraction.

**Why FAISS `IndexFlatIP` instead of approximate search (HNSW/IVF)?**  
For 2,058 chunks, exact search takes <5ms and fits in RAM. In a regulatory context, silently missing the single most relevant passage is a far costlier mistake than a slightly slower query. Exactness was chosen over approximation deliberately.

**Why Groq instead of OpenAI?**  
The Groq free tier (qwen3-32b, 6,000 TPM / 500,000 TPD) is fast enough for real-time demos at zero cost. The API is OpenAI-compatible, so switching providers is a one-line config change.

**Why `LLM_TEMPERATURE = 0.0`?**  
Legal and regulatory Q&A demands deterministic, reproducible answers. A non-zero temperature risks creative variation that could be confused with actual regulatory content — unacceptable in a compliance context.

**Why not fine-tune the LLM?**  
The RAG architecture compensates for the absence of domain-specific fine-tuning by anchoring every response in retrieved regulatory passages. The report's evaluation confirms this is sufficient (92.5% pass rate, 0 hallucinations) and keeps deployment flexible.

**Why standard-library authentication (no JWT/bcrypt)?**  
Fewer moving parts to audit and patch. PBKDF2-HMAC-SHA256 + HMAC-signed tokens from Python's stdlib provides strong security with one less external dependency.

**Why the off-topic guard runs before reformulation?**  
The reformulation agent rewrites even irrelevant questions into formal BCT-sounding language, which would artificially inflate their similarity score. Checking the raw query first ensures the rejection boundary is honest.

---

## Future Work

As identified in the report's conclusion:

- **Incremental corpus updates** — Automatic scraping and partial re-indexing when new BCT publications appear, without a full pipeline rebuild
- **Arabic-capable NER** — The French-only `fr_core_news_lg` model misses entity and citation links in Arabic documents; a multilingual NER model would improve graph completeness
- **Interactive graph visualisation** — A UI panel letting users explore CITES/MENTIONS links between circulars directly
- **LoRA fine-tuning** — Parameter-efficient fine-tuning on BCT regulatory text to improve fluency with Tunisian legal vocabulary and reduce edge-case hallucinations in offline mode
- **Full 200-question evaluation** — The pass/fail suite was held at 80 due to Groq free-tier daily token limits; the full bank with practitioner annotations would put results on firmer statistical ground

---

## Authors

- **Chouchane Mohamed Anwar** — Software Engineering / AI, École Polytechnique Internationale (EPI Digital)
- Academic supervisor: **Mrs. Boutheina Ben Ismail**
- Company supervisor: **Mr. Aymen Chakhari**, CEO of Skillia

---

## License

MIT License — see [LICENSE](LICENSE) for details.
