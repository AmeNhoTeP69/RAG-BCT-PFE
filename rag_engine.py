"""
rag_engine.py
─────────────
Standard RAG Engine

Combines FAISS vector search with LLM generation to answer questions based on 
retrieved document chunks.
"""

import json
import logging
import requests
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

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

    def search(self, query: str, k: int = config.TOP_K_CHUNKS):
        # 1. Embed query
        query_vec = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        
        # 2. Vector search — fetch more candidates for filtering
        distances, indices = self.index.search(query_vec, k)
        
        # 3. Retrieve chunks and filter by minimum relevance score
        min_score = getattr(config, 'MIN_RETRIEVAL_SCORE', 0.65)
        keywords = [w.lower() for w in query.split() if len(w) > 3]
        results = []
        
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                score = float(distances[0][i])
                chunk = self.chunks[idx]
                title = chunk['metadata'].get('title', '').lower()
                
                # Simple keyword boost in title
                boost = 0.0
                for kw in keywords:
                    if kw in title:
                        boost += 0.05
                
                final_score = score + boost
                
                if final_score >= min_score:
                    log.info(f"Retrieved: {chunk['metadata'].get('title')} (Score: {score:.4f}, Boost: {boost:.2f}, Final: {final_score:.4f})")
                    results.append({
                        "score": final_score,
                        "chunk": chunk
                    })
                else:
                    log.debug(f"Skipping chunk {idx} with low score: {final_score:.4f}")
        
        if not results:
            log.warning(
                f"No chunks above score threshold {min_score:.2f}. "
                "Falling back to top result regardless of score."
            )
            # Fall back: return only the single best result to avoid empty context
            if len(indices[0]) > 0 and indices[0][0] < len(self.chunks):
                results.append({
                    "score": float(distances[0][0]),
                    "chunk": self.chunks[indices[0][0]]
                })
        
        log.info(
            f"Retrieved {len(results)} chunks above score threshold {min_score:.2f}."
        )
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
            "temperature": config.LLM_TEMPERATURE
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=15)
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                elif response.status_code == 429:
                    wait_time = (attempt + 1) * 5
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

    def _build_prompt(self, query: str, context_text: str, **kwargs):
        return f"""Tu es un expert en réglementation de la Banque Centrale de Tunisie (BCT).
RÈGLES :
1. ANALYSE : Si la question porte sur une circulaire (ex: 2017-08), une loi tunisienne, ou la BCT, elle est VALIDÉE.
2. HORS SUJET : Si la question est totalement étrangère (ex: cuisine, sport), réponds : "Désolé, je ne réponds qu'à la réglementation BCT."
3. RIGUEUR : **Réponds UNIQUEMENT selon le CONTEXTE fourni.** Si l'information précise est absente (ex: montant exact, liste de documents), réponds : "L'information n'est pas disponible dans les documents fournis." 
   **NE PAS INVENTER** de règles ou suggérer des lois non pertinentes (ex: mesures COVID) si elles ne répondent pas directement à la question.
4. CITATION : Utilise DIRECTEMENT les titres des documents (ex: [Circulaire 2017-08]) dans ton texte au lieu de tags génériques comme [Source X].

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
            context_text += f"\n--- Source {i+1} : {title} {header} ---\n{chunk['text']}\n"
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

    def query(self, query: str, k: int = config.TOP_K_CHUNKS):
        log.info(f"Querying: {query}")
        
        # Fast path for greetings
        if self.is_simple_greeting(query):
            return {
                "answer": "Bonjour! Je suis votre assistant expert pour la réglementation de la Banque Centrale de Tunisie. Comment puis-je vous aider aujourd'hui?",
                "sources": []
            }
            
        retrieved = self.search(query, k)
        result = self.generate_answer(query, retrieved)
        return result

if __name__ == "__main__":
    # Test
    engine = RAGEngine()
    q = "Quels sont les documents à fournir pour l'ouverture d'un compte?"
    res = engine.query(q)
    print(f"\nQuestion: {q}")
    print(f"Answer: {res['answer']}")
    print(f"Sources: {res['sources']}")
