"""
rag_engine.py
─────────────
Standard RAG Engine

Combines FAISS vector search with LLM generation to answer questions based on 
retrieved document chunks.
"""

import json
import logging
import re
import requests
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

import bm25

from utils.connectivity import check_internet

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        self.config = config
        self.model = None
        self.index = None
        self.chunks = None
        self.bm25 = None       # lexical index for hybrid retrieval (built lazily below)
        self.online = True # Track last known state

        self.load_resources()

    def load_resources(self):
        log.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        
        log.info(f"Loading FAISS index from {config.FAISS_INDEX_PATH}...")
        try:
            self.index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        except RuntimeError:
            log.error(f"FAISS index not found. Run step4_index.py first.")
        
        log.info(f"Loading chunk index mappings from {config.CHUNK_INDEX_PATH}...")
        try:
            with open(config.CHUNK_INDEX_PATH, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
        except FileNotFoundError:
            log.error(f"Chunk index not found. Run step3_embed.py first.")

        # Build the BM25 lexical index for hybrid retrieval (dense + lexical).
        if self.chunks:
            self.bm25 = bm25.build_from_chunks(self.chunks)
            if self.bm25:
                log.info(f"BM25 lexical index built over {self.bm25.N} chunks (hybrid retrieval).")
            else:
                log.warning("BM25 index unavailable; falling back to dense-only retrieval.")

    def search(self, query: str, k: int = config.TOP_K_CHUNKS):
        """Hybrid retrieval: dense (FAISS) recall + BM25 lexical precision, fused
        with Reciprocal Rank Fusion. The lexical re-rank runs ONLY over chunks
        that already cleared the dense similarity threshold, so off-topic queries
        (which clear nothing) still fall through to the single-hit fallback the
        off-topic guard relies on."""
        # 1. Embed query
        query_vec = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)

        # 2. Dense retrieval over a WIDE candidate pool — we still return only k,
        #    but a wider pool lets the lexical re-ranker recover the precise
        #    section when dense score alone ranks it just below the top.
        pool = max(k, 4) * 10
        distances, indices = self.index.search(query_vec, pool)

        # 3. Relevant set = dense candidates above threshold (+ original title boost).
        min_score = getattr(config, 'MIN_RETRIEVAL_SCORE', 0.65)
        title_kw = [w.lower() for w in query.split() if len(w) > 3]
        relevant = []  # (cosine_final, chunk_idx, chunk)
        for i, idx in enumerate(indices[0]):
            if idx >= len(self.chunks):
                continue
            score = float(distances[0][i])
            chunk = self.chunks[idx]
            title = (chunk['metadata'].get('title', '') or '').lower()
            boost = sum(0.05 for kw in title_kw if kw in title)
            if score + boost >= min_score:
                relevant.append((score + boost, idx, chunk))

        if not relevant:
            log.warning(
                f"No chunks above score threshold {min_score:.2f}. "
                "Falling back to top result regardless of score."
            )
            if len(indices[0]) > 0 and indices[0][0] < len(self.chunks):
                return [{"score": float(distances[0][0]), "chunk": self.chunks[indices[0][0]]}]
            return []

        # 4. Weighted Reciprocal Rank Fusion over the relevant set. Dense gets
        #    full weight (it already passed the relevance gate); BM25 contributes
        #    at half weight, and only for sufficiently discriminative terms, so a
        #    doc strong on BOTH signals (the precise section) rises, while a doc
        #    matching only common terms lexically does not hijack good dense hits.
        dense_order = sorted(relevant, key=lambda r: r[0], reverse=True)
        order = dense_order

        if self.bm25 is not None and len(relevant) > 1:
            idxs = [r[1] for r in relevant]
            # Keep only discriminative query terms (drop very common ones) for BM25.
            q_terms = [t for t in bm25.tokenize(query) if self.bm25.idf.get(t, 0.0) >= 1.5]
            bm_scores = self.bm25.scores(q_terms, idxs) if q_terms else {}
            dense_rank = {r[1]: rnk for rnk, r in enumerate(dense_order)}
            bm_rank = {i: rnk for rnk, i in enumerate(sorted(idxs, key=lambda i: bm_scores.get(i, 0.0), reverse=True))}
            RRF, W_BM25 = 60, 0.5

            def fused(r):
                idx = r[1]
                s = 1.0 / (RRF + dense_rank[idx])
                if bm_scores.get(idx, 0.0) > 0:
                    s += W_BM25 / (RRF + bm_rank[idx])
                return s

            order = sorted(relevant, key=fused, reverse=True)

        results = [{"score": r[0], "chunk": r[2]} for r in order[:k]]
        for r in results:
            m = r["chunk"].get("metadata", {})
            log.info(f"Retrieved: {m.get('title')} {(m.get('section_header', '') or '')[:32]} (Score: {r['score']:.4f})")
        log.info(f"Retrieved {len(results)} chunks (dense+BM25 hybrid, pool={pool}).")
        return results

    def _call_ollama(self, prompt: str):
        log.info("Using Local Ollama for generation...")
        payload = {
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.LLM_TEMPERATURE
            }
        }
        try:
            response = requests.post(config.OLLAMA_URL, json=payload, timeout=300)
            return response.json().get('response', "Error: No response from Ollama.")
        except Exception as e:
            return f"Error contacting Ollama: {str(e)}"

    def _call_openai(self, prompt: str):
        """Call OpenAI/Groq API with retry logic for rate limits."""
        import time
        log.info("Using Groq Cloud (Online) for generation...")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.OPENAI_API_KEY}"
        }
        payload = {
            "model": config.OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": 2000
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=15)
                if response.status_code == 200:
                    import re
                    content = response.json()['choices'][0]['message']['content']
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                    return content
                elif response.status_code == 429:
                    wait_time = 20 + attempt * 20  # 20s, 40s, 60s
                    log.warning(f"Rate limit hit (429). Waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    log.error(f"LLM API Error: {response.status_code} - {response.text}")
                    return f"Erreur API LLM: {response.status_code}"
            except Exception as e:
                log.error(f"LLM Connection Error: {str(e)}")
                if attempt == max_retries - 1:
                    return f"Erreur de connexion LLM: {str(e)}"
                time.sleep(1)
        
        return "Erreur: Limite de taux dépassée après plusieurs tentatives."

    def _call_mock(self, query: str, context: str):
        return f"Ceci est une réponse simulée (MOCK MODE). \n\nVous avez posé la question: '{query}'. \n\nDans un environnement de production, l'LLM utiliserait les sources fournies pour générer une réponse détaillée."

    def _format_history(self, history, max_turns: int = 6) -> str:
        """Render the last few conversation turns for the LLM prompt."""
        if not history:
            return ""
        lines = []
        for turn in history[-max_turns:]:
            role = "Utilisateur" if turn.get("role") == "user" else "Assistant"
            content = (turn.get("content") or "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    # Cue words signalling a question depends on the previous turn.
    _FOLLOWUP_CUES = (
        "deuxième", "troisième", "premier", "précédent", "ci-dessus", "ce cas",
        "cela", "celui", "celle", "ceux", "cette", " ce ", " cet ", " ça",
        "the second", "the first", "the third", "previous", "above", "that case",
        "this one", "those", "these", " it ", " its ", "their",
        "الثاني", "الأول", "السابق", "ذلك", "هذا", "هذه",
    )

    def _is_followup(self, query: str) -> bool:
        """Heuristic: does this question depend on earlier turns (so retrieval
        should fold them in), or is it self-contained (retrieve on its own)?
        Keeps 'explique le deuxième point' working WITHOUT dragging an unrelated
        new question toward the previous topic."""
        q = f" {query.lower().strip()} "
        if len(query.split()) <= 4:                      # very short -> almost always a follow-up
            return True
        if re.match(r"^\s*(et|and|ok|oui|non|donc|alors|aussi|encore|puis|mais)\b", query.lower()):
            return True
        return any(cue in q for cue in self._FOLLOWUP_CUES)

    def _contextualize_query(self, query: str, history, max_turns: int = 2) -> str:
        """Fold recent USER turns into the retrieval query ONLY for follow-up
        questions. Self-contained questions are retrieved on their own, so an
        unrelated new question is never dragged toward the previous topic.
        Returns the query unchanged when there's no history or it stands alone."""
        if not history or not self._is_followup(query):
            return query
        prev_user = [t.get("content", "") for t in history if t.get("role") == "user"]
        prev_user = [c for c in prev_user if c][-max_turns:]
        if not prev_user:
            return query
        return " ".join(prev_user + [query])

    def _build_prompt(self, query: str, context_text: str, **kwargs):
        history_text = self._format_history(kwargs.get("history"))
        history_block = (
            f"\nHISTORIQUE DE LA CONVERSATION (pour résoudre les références comme "
            f"« le deuxième point » ou « ce cas ») :\n{history_text}\n"
            if history_text else ""
        )
        notes = kwargs.get("notes")
        notes_block = ""
        if notes:
            joined = "\n".join(f"- {n}" for n in notes)
            notes_block = (
                "\nNOTES VÉRIFIÉES (corrections factuelles validées par l'administrateur). "
                "Elles sont FIABLES : applique-les si elles sont pertinentes, même si elles ne "
                f"figurent pas dans le CONTEXTE :\n{joined}\n"
            )
        return f"""Tu es un expert en réglementation de la Banque Centrale de Tunisie (BCT).
RÈGLES :
1. ANALYSE : Si la question porte sur une circulaire (ex: 2017-08), une loi tunisienne, ou la BCT, elle est VALIDÉE.
2. HORS SUJET : Si la question est totalement étrangère (ex: cuisine, sport), réponds : "Désolé, je ne réponds qu'à la réglementation BCT."
3. RIGUEUR : **Réponds UNIQUEMENT selon le CONTEXTE fourni.** Si l'information précise est absente (ex: montant exact, liste de documents), réponds : "L'information n'est pas disponible dans les documents fournis."
   **NE PAS INVENTER** de règles ou suggérer des lois non pertinentes (ex: mesures COVID) si elles ne répondent pas directement à la question.
4. CITATION : Utilise DIRECTEMENT les titres des documents (ex: [Circulaire 2017-08]) dans ton texte au lieu de tags génériques comme [Source X].
5. CONTINUITÉ : Sers-toi de l'HISTORIQUE DE LA CONVERSATION (s'il est fourni) pour comprendre les questions de suivi et les références implicites, mais fonde toujours la réponse sur le CONTEXTE.
6. DISCUSSION & AUTO-CORRECTION : Tu es dans une vraie conversation. L'utilisateur peut commenter, contester ou corriger ta réponse précédente, reformuler/corriger sa propre question, ou te demander d'approfondir ou de réorienter.
   - S'il signale une erreur ou un oubli : RELIS attentivement le CONTEXTE et l'HISTORIQUE avant de répondre.
     • Si le CONTEXTE lui donne raison → reconnais l'erreur explicitement (ex: « Vous avez raison, … »), corrige-toi, explique ce qui change et cite le passage exact.
     • Si le CONTEXTE le CONTREDIT → ne te corrige PAS par complaisance ; signale poliment l'écart et cite le document qui le prouve. L'utilisateur peut se tromper.
     • Si le CONTEXTE ne permet pas de trancher → dis-le clairement, ne devine pas.
   - Ne change d'avis QUE si une preuve documentaire le justifie — jamais par simple insistance.
   - Tu peux discuter et raisonner autour des circulaires et notes BCT, mais distingue toujours ce qui provient du CONTEXTE de ton propre raisonnement, et reste dans le domaine de la réglementation BCT.
{notes_block}{history_block}
CONTEXTE :
{context_text}

QUESTION :
{query}

RÉPONSE :"""

    def generate_answer(self, query: str, context_chunks: list, **kwargs):
        # Build context
        context_text = ""
        sources = []
        for i, res in enumerate(context_chunks):
            chunk = res['chunk']
            title = chunk['metadata'].get('title', 'Unknown Title')
            header = chunk['metadata'].get('section_header', '')
            text = chunk['text'][:config.MAX_CONTEXT_CHUNK_CHARS]
            context_text += f"\n--- Source {i+1} : {title} {header} ---\n{text}\n"
            sources.append(f"{title} ({header})")

        prompt = self._build_prompt(query, context_text, **kwargs)
            
        # Connectivity detection and fallback logic
        self.online = check_internet()
        
        if config.LLM_PROVIDER == "mock":
            answer = self._call_mock(query, context_text)
        elif self.online:
            log.info("Internet detected. Routing to Groq...")
            answer = self._call_openai(prompt)
            # If for some reason Groq failed but internet was "up", attempt Ollama as last resort
            if "Erreur" in answer and "API" in answer:
                log.warning("Groq failed despite connectivity. Falling back to Ollama...")
                answer = self._call_ollama(prompt)
        else:
            log.warning("No internet connection detected. Falling back to Local Ollama...")
            answer = self._call_ollama(prompt)

        return {
            "answer": answer,
            "sources": list(set(sources)), # Unique source names
            "retrieved_chunks": [
                {
                    "text": res['chunk']['text'],
                    "metadata": res['chunk']['metadata']
                } for res in context_chunks
            ]
        }

    def is_simple_greeting(self, query: str) -> bool:
        """Check if the query is a simple greeting to bypass heavy RAG logic."""
        greetings = ["hi", "hello", "salut", "bonjour", "hey", "cc", "test"]
        q = query.lower().strip().replace("?", "").replace("!", "")
        return q in greetings or len(q) < 4

    def query(self, query: str, k: int = config.TOP_K_CHUNKS, history=None, notes=None):
        log.info(f"Querying: {query}")

        # Fast path for greetings — only on the first turn (no history), so short
        # follow-ups like "oui" inside an ongoing conversation are not swallowed.
        if not history and self.is_simple_greeting(query):
            return {
                "answer": "Hello! I'm your expert assistant for Central Bank of Tunisia (BCT) regulations. How can I help you today?",
                "sources": []
            }

        # Fold recent conversation context into the retrieval query for follow-ups.
        search_query = self._contextualize_query(query, history)
        retrieved = self.search(search_query, k)

        # Off-topic hard guard: only 1 fallback result AND below the relevance threshold → unrelated to BCT
        if len(retrieved) == 1 and retrieved[0]["score"] < 0.50:
            log.info(f"Off-topic guard triggered (score={retrieved[0]['score']:.4f}). Refusing.")
            return {
                "answer": "Sorry, I can only answer questions about Central Bank of Tunisia (BCT) regulations.",
                "sources": []
            }

        result = self.generate_answer(query, retrieved, history=history, notes=notes)
        return result

if __name__ == "__main__":
    # Test
    engine = RAGEngine()
    q = "Quels sont les documents à fournir pour l'ouverture d'un compte?"
    res = engine.query(q)
    print(f"\nQuestion: {q}")
    print(f"Answer: {res['answer']}")
    print(f"Sources: {res['sources']}")
