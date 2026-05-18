# BCT Hybrid RAG System — Guide Complet & Workflow de Démonstration

## Vue d'ensemble du Système

Ce projet implémente un **Hybrid Graph RAG** avancé pour interroger les circulaires et notes de la **Banque Centrale de Tunisie (BCT)**. L'architecture combine trois paradigmes de retrieval :

| Composant | Technologie | Rôle |
|---|---|---|
| Retrieval Dense | Transformer + FAISS | Similarité sémantique profonde |
| Retrieval Sparse | LDA (Gensim) | Thèmes latents réglementaires |
| Retrieval Structurel | Knowledge Graph (NetworkX) | Relations inter-documents |
| Query Expansion | Word2Vec (Gensim) | Synonymes & termes liés |
| Query Reformulation | Google Gemma (OpenRouter) | Réécriture de la question |
| Optimisation ML | SMOTE + AdamW (PyTorch) | Classification réglementaire |

---

## Architecture du Système

```
BCT Documents (PDF)
        │
        ▼
[Step 1] step1_extract.py       ── Extraction texte (PyMuPDF + Tesseract OCR)
        │                            Détection langue (fr/ar), métadonnées
        ▼
[Step 2] step2_chunk.py         ── Découpage en chunks sémantiques
        │                            Section-aware (article, titre, préambule)
        ▼
[Step 3] step3_embed.py         ── Vectorisation Transformer
        │                            paraphrase-multilingual-mpnet (768-dim)
        ▼
[Step 4] step4_index.py         ── Indexation FAISS (similarité cosinus)
        │
[Step 5] step5_build_graph.py   ── Construction Knowledge Graph
        │                            SpaCy NER → entités, CITES, MENTIONS
        ▼
[Step 6] step6_topic_modeling.py── Hybrid Feature Engineering
        │                            ├─ Gensim LDA (8 topics réglementaires)
        │                            ├─ Word2Vec (similarité lexicale)
        │                            └─ Enrichissement graphe (HAS_TOPIC)
        ▼
[Step 7] step7_optimization_ml.py─ Optimisation ML
        │                            ├─ TF-IDF Feature Engineering
        │                            ├─ SMOTE Oversampling (classes déséquilibrées)
        │                            └─ AdamW Classifier (PyTorch MLP)
        ▼
┌──────────────────────────────────────────────────────────┐
│              HybridGraphRAGEngine (graph_rag_engine.py)  │
│                                                          │
│  Question Utilisateur                                    │
│       │                                                  │
│       ▼ [1] Reformulation (Google Gemma — free)          │
│       │     Réécriture formelle de la question           │
│       ▼ [2] Expansion Word2Vec                           │
│       │     Ajout synonymes & termes liés                │
│       ▼ [3] Retrieval HYBRIDE                            │
│       │     ├─ Dense  : FAISS (Transformer embeddings)   │
│       │     ├─ Sparse : LDA topic matching               │
│       │     └─ Graph  : entités + citations              │
│       ▼ [4] Fusion de Contexte                           │
│       │     Fusion vecteur + graphe + topics             │
│       ▼ [5] Génération LLM (Llama / Groq)               │
│             Réponse ancrée + sources citées              │
└──────────────────────┬───────────────────────────────────┘
                       ▼
                api.py (FastAPI)
                       │
                static/ (Web UI)
```

---

## Lancement du Système

### Prérequis
- ✅ Ollama installé et lancé : `ollama serve`
- ✅ Modèle présent : `ollama run llama3.2:1b`
- ✅ Python venv activé

### Commandes

```bash
# Lancer l'interface complète
python run_project.py
# Choisir option 3 → Interface Web (http://localhost:8000)
# Choisir option 4 → Mode CLI interactif
```

### Construire/Reconstruire le pipeline
```bash
python run_project.py  # Option 1 → Full Pipeline
# Ou étape par étape :
python step1_extract.py
python step2_chunk.py
python step3_embed.py
python step4_index.py
python step5_build_graph.py

# Nouvelles étapes Hybrid RAG :
python step6_topic_modeling.py   # LDA + Word2Vec + enrichissement graphe
python step7_optimization_ml.py  # SMOTE + AdamW classifier
```

### Variable d'environnement (optionnel — Gemma Query Reformulation)
```bash
# Obtenir une clé gratuite sur https://openrouter.ai/
set OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxx
# Sinon, la reformulation est ignorée et la question originale est utilisée.
```

---

## Fonctionnalités Complètes du Système

### 1. Retrieval Hybride (Transformer + LDA + Graph)
L'architecture combine trois sources de contexte fusionnées avant la génération LLM :

| Capacité | Standard RAG | Hybrid Graph RAG |
|---|---|---|
| Recherche vectorielle FAISS (dense) | ✅ | ✅ |
| Modélisation thématique LDA (sparse) | ❌ | ✅ |
| Navigation Knowledge Graph | ❌ | ✅ |
| Expansion de requête Word2Vec | ❌ | ✅ |
| Reformulation Gemma (LLM agent) | ❌ | ✅ |
| Nœuds Topic `HAS_TOPIC` dans le graphe | ❌ | ✅ |
| Annotation catégorie réglementaire | ❌ | ✅ (step 7) |
| Champ `related_nodes` dans la réponse API | ❌ | ✅ |

### 2. Grounding Anti-Hallucination (Prompt Hardening)
Le prompt système impose des règles strictes :
- ✅ Questions sur circulaires, lois tunisiennes, BCT → **autorisées**
- ✅ Si le contexte contient la réponse → réponse ancrée avec citation source
- ❌ Questions hors-sujet (cuisine, sport) → refus poli immédiat
- ❌ Informations absentes du contexte → aveu explicite "L'information n'est pas dans les documents"

### 3. Température 0 (Déterminisme Total)
`LLM_TEMPERATURE = 0.0` — Les réponses sont parfaitement reproductibles. Deux appels identiques donnent toujours la même réponse.

### 4. Backend Local Ollama (Aucune Connexion Internet Requise)
Le système tourne entièrement en local sur `llama3.2:1b` via **Ollama**. La configuration dans `config.py` pointe vers `http://localhost:11434`. Aucune clé API, aucune dépendance cloud.

### 5. Fast Path pour Greetings
```python
def is_simple_greeting(query) -> bool:
```
Les messages courts (`hi`, `bonjour`, `test`, etc.) sont détectés et répondus instantanément **sans déclencher** FAISS ni le LLM.

### 6. Lazy Loading des Moteurs
Les moteurs RAG et Graph RAG sont chargés **à la demande** (seulement au premier appel API), évitant de charger inutilement les deux modèles en mémoire simultanément.

### 7. Recherche Bilingue AR + FR
Le modèle `paraphrase-multilingual-mpnet-base-v2` produit des embeddings de 768 dimensions supportant nativement l'arabe et le français. Une question en français peut retrouver des passages en arabe et vice-versa.

### 8. OCR Intégré (Fallback Tesseract)
Les PDFs scannés (images) sont traités automatiquement avec **Tesseract OCR** dans `step1_extract.py`. Aucun document n'est perdu même s'il est non-sélectionnable.

### 9. Détection de Langue Automatique (langdetect)
Chaque document est étiqueté `fr` ou `ar` dans ses métadonnées lors de l'extraction, permettant une analysis EDA de la répartition linguistique (63% AR / 37% FR).

### 10. Knowledge Graph Sémantique Enrichi (NetworkX)
- **Nœuds** : Documents, Entités (ORG, DATE, PER), Topics LDA
- **Relations** : `CITES`, `MENTIONS`, **`HAS_TOPIC`** _(nouveau — step 6)_
- **Navigation** : Entités + citations + topics pour contexte multi-hop
- **Format** : JSON (node-link format), rechargeable via NetworkX

### 10b. Modélisation Thématique LDA (Gensim)
- **8 Topics** réglementaires extraits automatiquement du corpus BCT
- **Cohérence Cv** calculée pour valider la qualité du modèle
- **Topics** : Changes, LCB-FT, Reporting, Crédit, Risques, Marchés, Microfinance, Gouvernance
- **Fichiers** : `data/topics/lda_model`, `doc_topics.json`, `topics.json`

### 10c. Word2Vec — Similarité Lexicale
- Entraîné sur le corpus BCT (domaine réglementaire spécifique)
- Utilisé pour **l'expansion de requête** : `"capital"` → `"fonds propres", "solvabilité"`
- Seuil de similarité configurable (`W2V_QUERY_EXPANSION_TOPN=3`, score > 0.65)

### 10d. SMOTE + AdamW — Optimisation ML
- **Feature Engineering** : TF-IDF (2000 features, bigrammes)
- **SMOTE** : Génération synthétique d'exemples pour classes minoritaires
- **AdamW** : Adam avec weight decay découplé (meilleure généralisation)
- **Scheduler** : CosineAnnealingLR pour convergence douce
- **Sortie** : chunks annotés avec `regulation_category` dans leurs métadonnées

### 10e. Reformulation de Question (Google Gemma — Agent)
- Réécriture formelle via `google/gemma-3-4b-it:free` sur OpenRouter
- Gratuit, aucune clé payante requise
- Comparable au pattern **Agent n8n** : trigger → LLM → reformulation → retrieval
- Désactivable via `QUERY_REFORMULATION_ENABLED = False` dans `config.py`

### 11. Pipeline Modulaire en 5 Étapes
Chaque étape peut être relancée indépendamment :
```
step1 (extraction) → step2 (chunking) → step3 (embedding) → step4 (FAISS) → step5 (graph)
```

### 12. Interface Web + API REST (FastAPI)
- `POST /api/query` : Requête RAG ou Graph RAG avec sélection de mode
- `GET /` : Interface Web (HTML/CSS/JS dans `static/`)
- CORS activé pour le développement local

### 13. Sources Citées dans la Réponse API
Chaque réponse retourne :
```json
{
  "answer": "...",
  "sources": ["CB 2017 08 FR (Article 5)", ...],
  "related_nodes": ["bct_doc_02", ...],  ← Graph RAG uniquement
  "retrieved_chunks": [{"text": "...", "metadata": {...}}]
}
```

### 14. Section-Aware Chunking
Le découpage en Step 2 est conscient de la structure des articles (`Article 1`, `Article 2`, `Titre I`) pour éviter de couper au milieu d'une disposition réglementaire.

---

## Scénarios de Démonstration

### Test 1 — Réponse factuelle (Standard RAG baseline)
> "Qu'est-ce qu'une circulaire de la BCT ?"

**Attendu** : Définition ancrée dans un chunk de document réel, sources citées.

### Test 2 — Connectivité inter-documents (Graph RAG)
> "Quels sont les liens entre la circulaire n°2017-08 et la loi n°2016-48 ?"

**Attendu** : Le champ `related_nodes` liste les arcs `CITES` reliant les deux documents.

### Test 3 — Topic Boosting (LDA Hybrid)
> "Quelles sont les exigences de la BCT en matière de capital réglementaire ?"

**Attendu** : Le moteur identifie le topic dominant (`Reporting et Supervision Prudentielle`), remonte tous les documents taggués `HAS_TOPIC` → enrichit le contexte.

### Test 4 — Expansion Word2Vec
> "Quelle est la politique de la BCT sur le blanchiment ?"

**Attendu** : Word2Vec ajoute `fraude`, `lcb`, `vigilance` → meilleure couverture. Le champ `expanded_query` dans la réponse API montre les termes ajoutés.

### Test 5 — Reformulation Gemma (Agent)
> "c'est quoi le truc pour les devises ?"

**Attendu** : Gemma reformule en → `"Quelles sont les réglementations BCT relatives aux opérations de change en devises étrangères ?"` avant la recherche.

### Test 6 — Sécurité hors-sujet
> "Comment cuisiner un couscous tunisien ?"

**Attendu** : Refus poli — "Désolé, je ne réponds qu'à la réglementation BCT."

### Test 7 — Annotation Classification (step 7)
```python
# Après step7, les chunks ont une catégorie :
chunk["metadata"]["regulation_category"]  # → "Blanchiment d'Argent / LCB-FT"
chunk["metadata"]["category_id"]           # → 1
```

---

## Structure des Fichiers

```
Racine du Projet/
├── 📋 Documents
│   ├── academic_summary.md          ← Justification technologique (rapport PFE)
│   ├── supervisor_notes_response.md ← Réponses aux notes de l'encadrante
│   └── workflow_demo_guide.md       ← Ce fichier
│
├── ⚙️ Moteurs RAG
│   ├── rag_engine.py                ← Standard RAG (FAISS + LLM)
│   └── graph_rag_engine.py          ← Hybrid Graph RAG (5-stage pipeline)
│
├── 🚀 Pipeline de Traitement (7 étapes)
│   ├── step1_extract.py             ← Extraction PDF + OCR
│   ├── step2_chunk.py               ← Découpage en chunks sémantiques
│   ├── step3_embed.py               ← Vectorisation Transformer
│   ├── step4_index.py               ← Indexation FAISS
│   ├── step5_build_graph.py         ← Knowledge Graph (SpaCy NER)
│   ├── step6_topic_modeling.py      ← [NOUVEAU] LDA + Word2Vec + enrichissement graphe
│   └── step7_optimization_ml.py     ← [NOUVEAU] SMOTE + AdamW classifier
│
├── 🌐 Application Web
│   ├── api.py                       ← API FastAPI
│   ├── run_project.py               ← Lanceur principal
│   └── static/                      ← Interface Web (HTML/JS/CSS)
│
├── 🔍 Scraping
│   ├── scraper.py / scraper_browser.py
│   └── links.py / links.txt
│
├── 🛠️ Configuration
│   ├── config.py                    ← Tous les paramètres centralisés
│   └── requirements.txt             ← Dépendances Python
│
└── 📁 Données (générées)
    └── data/
        ├── extracted/               ← documents.json
        ├── chunks/                  ← chunks.json + chunks_annotated.json
        ├── faiss/                   ← bct_index.faiss + embeddings
        ├── graph/                   ← graph.json (enrichi topics) + entities.json
        ├── topics/                  ← lda_model + word2vec.model + doc_topics.json
        └── classifier/              ← regulation_classifier.pt + classifier_meta.json
```

---

## Note sur l'Architecture Agent (n8n)

La reformulation via Google Gemma implémente le **pattern agent** mentionné par l'encadrante :

```
Trigger (Question utilisateur)
    │
    ▼  [Agent Step 1]
Google Gemma — Reformulation de la question
    │
    ▼  [Agent Step 2]
Hybrid RAG — Retrieval (FAISS + LDA + Graph)
    │
    ▼  [Agent Step 3]
LLM (Llama/Groq) — Génération réponse ancrée
    │
    ▼
Réponse finale avec sources + relations graphe
```

En **n8n**, ce workflow serait orchestré avec des nœuds HTTP Request → AI Agent → Response. Le code Python ici implémente la même logique directement.
