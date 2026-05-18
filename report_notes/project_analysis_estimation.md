# Project Analysis & Time Estimation Report
Date: 2026-05-09
Project: Hybrid Graph RAG for BCT Regulatory Documents

## 1. Executive Summary
This report provides a deep technical analysis of the BCT Hybrid Graph RAG system and a professional estimation of the development effort required for a single developer to build it from scratch.

## 2. Technical Complexity Analysis
The project implements a state-of-the-art "Hybrid" architecture that goes beyond standard RAG (Retrieval-Augmented Generation).

### A. The 7-Step Pipeline
1. **Extraction & OCR:** Processing bilingual (AR/FR) PDFs using PyMuPDF and Tesseract.
2. **Semantic Chunking:** Structural-aware document splitting.
3. **Embedding:** Multilingual vectorization (768-dim).
4. **Vector Indexing:** FAISS for dense retrieval.
5. **Knowledge Graph:** Structural mapping using SpaCy NER and NetworkX.
6. **Advanced NLP:** LDA Topic Modeling (Gensim) and Word2Vec Query Expansion.
7. **ML Optimization:** PyTorch MLP Classifier with SMOTE (Oversampling) and AdamW Optimizer.

### B. Core Challenges
- **Bilingual OCR:** Arabic/French alignment and denoising in scanned documents.
- **Hybrid Fusion:** Balancing dense (FAISS), sparse (LDA), and structural (Graph) retrieval.
- **Agentic Logic:** Implementing LLM-based query reformulation (Google Gemma).
- **Data Imbalance:** Using SMOTE to handle the unequal distribution of regulatory topics.

## 3. Development Time Estimation (Single Developer)

| Phase | Duration | Key Focus |
| :--- | :--- | :--- |
| **Data & Scraping** | 2.0 Weeks | Scrapers, OCR, multi-page Arabic handling. |
| **Core RAG Foundation** | 1.5 Weeks | Vectorization, FAISS, Chunking logic. |
| **Knowledge Graph** | 2.0 Weeks | Entity extraction, Relation mapping, Navigation. |
| **Advanced NLP** | 1.0 Week | LDA Topic tuning, Word2Vec training. |
| **ML Optimization** | 1.5 Weeks | SMOTE, PyTorch Classifier design, AdamW training. |
| **Integration & API** | 1.5 Weeks | FastAPI, Agentic query reformulation, Web UI. |
| **Testing & Hardening** | 1.5 Weeks | Anti-hallucination, System prompts, Reporting. |
| **TOTAL** | **~11 Weeks** | **Approx. 2.5 - 3 Months (Full-Time)** |

## 4. Professional Verdict (No Sugar)
This project is categorized as **High-Tier AI Engineering**. 

- **Why 3 Months?** Standard RAG can be built in days, but **Hybrid Graph RAG** requires solving complex integration issues. Debugging the "Context Fusion" between three different retrieval engines takes weeks of refinement to avoid noise.
- **Research vs. Implementation:** The inclusion of SMOTE and AdamW indicates a transition from simple development to active machine learning research, significantly increasing the required expertise.
- **Industrial Readiness:** The modularity (Step 1-7) and the hardening against hallucinations (System Prompts + Temperature 0) make this a production-ready MVP rather than a simple demonstration.

---
*Note: This estimation assumes a competent developer working 40 hours/week. A student or junior developer may require additional time for the learning curve associated with the diverse tech stack (PyTorch, FAISS, SpaCy, Gensim, NetworkX).*
