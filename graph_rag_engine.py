"""
graph_rag_engine.py
────────────────────────────────────────────────────────────────────────────────
Hybrid Graph RAG Engine — Enhanced Architecture

Pipeline for each query:

  User Question
       │
       ▼
  [1] Query Reformulation (Google Gemma via OpenRouter — free)
       │   Rewrites ambiguous questions for better retrieval
       │
       ▼
  [2] Word2Vec Query Expansion (Gensim)
       │   Adds semantically similar terms to catch synonyms
       │   e.g., "blanchiment" → also search "fraude", "lcb"
       │
       ▼
  [3] Hybrid Retrieval
       ├─► FAISS Vector Search (Transformer embeddings — dense)
       └─► Topic-Boosted Graph Traversal (LDA topics — sparse)
              │
              ▼
  [4] Context Fusion
       │   Merges vector results + graph neighbors + topic links
       │
       ▼
  [5] LLM Answer Generation (Ollama / Groq)
       │   Grounded, cited answer
       ▼
  Final Answer + Sources + Graph Relations
"""

import json
import logging
import re
import requests
import networkx as nx
import spacy

import config
from utils.connectivity import check_internet
from rag_engine import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class HybridGraphRAGEngine(RAGEngine):
    """
    Hybrid RAG Engine combining:
      - Dense vector retrieval (Transformer + FAISS)
      - Sparse topic retrieval (LDA topic matching)
      - Knowledge Graph traversal (entity + citation links)
      - Word2Vec query expansion
      - LLM query reformulation (Google Gemma)
    """

    def __init__(self):
        super().__init__()
        self.G = None
        self.nlp = None
        self.w2v_model = None
        self.lda_model = None
        self.lda_dictionary = None
        self.doc_topics = {}

        self._load_graph()
        self._load_spacy()
        self._load_topic_models()

    # ── Resource Loading ──────────────────────────────────────────────────────

    def _load_graph(self):
        log.info(f"Loading knowledge graph from {config.GRAPH_PATH}...")
        try:
            with open(config.GRAPH_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.G = nx.node_link_graph(data)
            log.info(
                f"Graph loaded: {self.G.number_of_nodes()} nodes, "
                f"{self.G.number_of_edges()} edges."
            )
        except Exception as e:
            log.error(f"Could not load graph: {e}. Run step5_build_graph.py first.")

    def _load_spacy(self):
        log.info(f"Loading spaCy model: {config.SPACY_MODEL}...")
        try:
            self.nlp = spacy.load(config.SPACY_MODEL)
        except OSError:
            log.warning("spaCy model not found. Entity extraction disabled.")

    def _load_topic_models(self):
        """Load Gensim LDA + Word2Vec models (from step6)."""
        # Word2Vec
        if config.W2V_MODEL_PATH.exists():
            try:
                from gensim.models import Word2Vec
                self.w2v_model = Word2Vec.load(str(config.W2V_MODEL_PATH))
                log.info(
                    f"Word2Vec loaded: {len(self.w2v_model.wv)} vocab terms."
                )
            except Exception as e:
                log.warning(f"Could not load Word2Vec: {e}")
        else:
            log.warning(
                "Word2Vec model not found. Run step6_topic_modeling.py for query expansion."
            )

        # LDA
        if config.LDA_MODEL_PATH.exists():
            try:
                from gensim.models import LdaModel
                from gensim import corpora
                self.lda_model = LdaModel.load(str(config.LDA_MODEL_PATH))
                self.lda_dictionary = corpora.Dictionary.load(
                    str(config.LDA_DICTIONARY_PATH)
                )
                log.info(
                    f"LDA loaded: {self.lda_model.num_topics} topics."
                )
            except Exception as e:
                log.warning(f"Could not load LDA: {e}")
        else:
            log.warning(
                "LDA model not found. Run step6_topic_modeling.py for topic boosting."
            )

        # Doc-topic distributions
        if config.DOC_TOPICS_PATH.exists():
            with open(config.DOC_TOPICS_PATH, "r", encoding="utf-8") as f:
                self.doc_topics = json.load(f)

    # ── Stage 1: Query Reformulation (Gemma) ──────────────────────────────────

    # ── Stage 1: Query Reformulation (Gemma / Fallback to Ollama) ──────────────────────────────
    def reformulate_query(self, query: str) -> str:
        """
        Use Groq (Llama) to rewrite the user's question.
        Fallback to Ollama if offline.
        """
        if not config.QUERY_REFORMULATION_ENABLED:
            return query

        self.online = check_internet()

        log.info(f"Reformulating query ({'Groq' if self.online else 'Ollama'})...")
        prompt = (
            "Tu es un expert en réglementation bancaire tunisienne.\n"
            "Reformule la question suivante en une question précise et formelle.\n"
            "IMPORTANT : Si la question est en ARABE, la reformulation doit rester en ARABE.\n"
            "Réponds UNIQUEMENT avec la question reformulée, sans explication.\n\n"
            f"Question originale: {query}\n\n"
            "Question reformulée:"
        )

        if self.online:
            headers = {
                "Authorization": f"Bearer {config.GEMMA_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": config.GEMMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 200,
            }
            try:
                resp = requests.post(config.GEMMA_API_URL, headers=headers, json=payload, timeout=15)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                log.warning(f"Groq reformulation failed: {e}")
        
        # Fallback to Ollama for reformulation
        try:
            log.info("Attempting reformulation with Local Ollama...")
            return self._call_ollama(prompt)
        except Exception as e:
            log.warning(f"Ollama reformulation failed: {e}")
            return query

    # ── Stage 2: Word2Vec Query Expansion ─────────────────────────────────────

    def expand_query_with_w2v(self, query: str) -> str:
        """
        Expand the query by appending semantically similar terms found
        via Word2Vec word similarity.

        This captures synonyms and related concepts not in the original question.
        Example: "capital" → also adds "fonds propres", "solvabilité"
        """
        if not self.w2v_model:
            return query

        tokens = re.sub(r"[^a-zàâäéèêëîïôùûüç\s]", " ", query.lower()).split()
        expansion_terms = set()

        for token in tokens:
            if len(token) < 3:
                continue
            try:
                similar = self.w2v_model.wv.most_similar(
                    token, topn=config.W2V_QUERY_EXPANSION_TOPN
                )
                for word, score in similar:
                    if score > 0.65:  # Similarity threshold to avoid noise
                        expansion_terms.add(word)
            except KeyError:
                pass  # Token not in W2V vocabulary

        if expansion_terms:
            expansion_str = " ".join(expansion_terms)
            expanded = f"{query} {expansion_str}"
            log.info(f"Query expanded with W2V terms: {expansion_terms}")
            return expanded

        return query

    # ── Stage 3a: Topic-Based Graph Retrieval ─────────────────────────────────

    def get_topic_related_docs(self, query: str) -> set:
        """
        Infer the dominant topic of the query using the LDA model,
        then retrieve all document IDs from the graph that share that topic.

        This is the "sparse" retrieval component of the Hybrid RAG.
        """
        if not self.lda_model or not self.lda_dictionary or not self.G:
            return set()

        # Tokenize query
        tokens = re.sub(r"[^a-zàâäéèêëîïôùûüç\s]", " ", query.lower()).split()
        tokens = [t for t in tokens if len(t) >= 3]

        if not tokens:
            return set()

        bow = self.lda_dictionary.doc2bow(tokens)
        if not bow:
            return set()

        # Get query's topic distribution
        topic_dist = self.lda_model.get_document_topics(bow, minimum_probability=0.05)
        if not topic_dist:
            return set()

        # Find dominant topic
        dominant_topic_id, prob = max(topic_dist, key=lambda x: x[1])
        topic_node_id = f"topic_{dominant_topic_id}"
        topic_label = (
            config.LDA_TOPIC_LABELS[dominant_topic_id]
            if dominant_topic_id < len(config.LDA_TOPIC_LABELS)
            else f"Topic {dominant_topic_id}"
        )
        log.info(
            f"Query topic inference: '{topic_label}' (prob={prob:.3f})"
        )

        # Find all document nodes linked to this topic via HAS_TOPIC edges
        related_docs = set()
        if self.G.has_node(topic_node_id):
            for source, target, data in self.G.in_edges(topic_node_id, data=True):
                if data.get("relation") == "HAS_TOPIC":
                    related_docs.add(source)

        log.info(
            f"Topic-boosted retrieval found {len(related_docs)} related documents."
        )
        return related_docs

    # ── Stage 3b: Entity + Citation Graph Traversal ───────────────────────────

    def find_entities_in_query(self, query: str) -> list:
        """
        Extract named entities and regulatory patterns (like 2017-08) 
        from query and match to graph nodes using fuzzy/partial matching.
        """
        if not self.G:
            return []
            
        matched = []
        query_lower = query.lower()
        
        # 1. Regex Pattern Matching (Catching IDs like 2017-08)
        # This catches "2017-08", "2017/08", "93-08" etc.
        patterns = re.findall(r"(\d{2,4}[-/]\d{2,4})", query_lower)
        for p in patterns:
            for n, d in self.G.nodes(data=True):
                # Match if pattern is in the ID or the Label
                if p in n.lower() or p in d.get("label", "").lower():
                    matched.append(n)
                    log.info(f"Graph fuzzy match found for pattern '{p}': {n}")

        # 2. SpaCy Entity Matching (Fallback)
        if self.nlp:
            doc_nlp = self.nlp(query)
            for ent in doc_nlp.ents:
                ent_text = ent.text.lower().strip()
                if len(ent_text) < 3: continue
                
                for n, d in self.G.nodes(data=True):
                    label = d.get("label", "").lower()
                    if ent_text in label or label in ent_text:
                        matched.append(n)
        
        return list(set(matched))

    def get_citation_neighbors(self, doc_id: str) -> set:
        """Find documents that this doc cites (CITES edges)."""
        if not self.G or not self.G.has_node(doc_id):
            return set()
        cited = set()
        for source, target, data in self.G.edges(doc_id, data=True):
            if data.get("relation") == "CITES":
                cited.add(target)
        return cited

    # ── Stage 4: Context Fusion ───────────────────────────────────────────────

    def _build_context(self, vector_results: list, all_related_ids: set) -> tuple:
        """
        Merge vector search results and graph-related document IDs
        into a single context string for the LLM.
        Returns (context_text, sources_list).
        """
        context_text = ""
        sources = []

        # Primary context from vector search
        for i, res in enumerate(vector_results):
            chunk = res["chunk"]
            title = chunk["metadata"].get("title", "Document inconnu")
            header = chunk["metadata"].get("section_header", "")
            category = chunk["metadata"].get("regulation_category", "")
            score = res.get("score", 0.0)

            context_text += (
                f"\n--- Source {i+1} [{category}] : {title} {header} "
                f"(score={score:.3f}) ---\n{chunk['text']}\n"
            )
            sources.append(title)

        # ── Graph-derived context: fetch ACTUAL TEXT for graph-matched doc IDs ──
        # Core fix: instead of listing node IDs as hints, we look up real chunk
        # text for those nodes and inject it into the LLM context.
        already_seen_titles = set(sources)
        graph_chunks_added = 0

        if all_related_ids and self.chunks:
            for chunk in self.chunks:
                doc_id = chunk.get("metadata", {}).get("doc_id", "")
                title = chunk.get("metadata", {}).get("title", "")

                is_related = any(
                    rel_id.lower() in doc_id.lower() or
                    rel_id.lower() in title.lower() or
                    doc_id.lower() in rel_id.lower()
                    for rel_id in all_related_ids
                )

                if is_related and title not in already_seen_titles:
                    header = chunk.get("metadata", {}).get("section_header", "")
                    context_text += (
                        f"\n--- [GRAPH] Source {len(sources)+graph_chunks_added+1} : "
                        f"{title} {header} ---\n{chunk['text']}\n"
                    )
                    sources.append(title)
                    already_seen_titles.add(title)
                    graph_chunks_added += 1
                    if graph_chunks_added >= 3:
                        break

        if graph_chunks_added == 0 and all_related_ids:
            rel_str = ", ".join(list(all_related_ids)[:12])
            context_text += (
                f"\n\n[RELATIONS GRAPHE] Documents lies dans le graphe BCT :\n{rel_str}\n"
            )

        return context_text, sources

    # ── Stage 5: Answer Generation ────────────────────────────────────────────

    def _build_prompt(self, query: str, context_text: str, **kwargs) -> str:
        related_ids = kwargs.get("related_ids", set())
        relations_str = (
            ", ".join(list(related_ids)[:10]) if related_ids else "Aucune relation trouvée."
        )

        return f"""ROLE: You are an expert on Central Bank of Tunisia (BCT) regulations.
- Use the provided context (VECTOR SEARCH and GRAPH) to answer precisely.
- LANGUAGE RULE: You MUST respond in the SAME language as the user's question (Arabic, French, or English).
- CITATION: Use specific document titles (e.g., [Circular 2017-08]) directly in your response instead of generic tags like [Source X].
- SAFETY: If the answer is not in the context, say you don't know. Do not hallucinate.

CONTEXT PROVIDED:
{context_text}

GRAPH RELATIONS:
{relations_str}

USER QUESTION:
{query}

YOUR RESPONSE:"""


    # ── Main Query Method ─────────────────────────────────────────────────────

    def query(self, query: str, k: int = config.TOP_K_CHUNKS) -> dict:
        log.info(f"{'='*60}")
        log.info(f"Hybrid Graph RAG Query: {query}")
        log.info(f"{'='*60}")

        # Fast path for greetings
        if self.is_simple_greeting(query):
            return {
                "answer": (
                    "Bonjour ! Je suis votre assistant expert pour la réglementation "
                    "de la Banque Centrale de Tunisie. Comment puis-je vous aider ?"
                ),
                "sources": [],
                "pipeline": "greeting",
            }

        # ── Stage 1: Query Reformulation ──────────────────────────────
        reformulated_query = self.reformulate_query(query)

        # ── Stage 2: Word2Vec Query Expansion ─────────────────────────────────
        expanded_query = self.expand_query_with_w2v(reformulated_query)

        # ── Stage 3a: Dense Vector Retrieval (expanded query) ─────────────────
        vector_results = self.search(expanded_query, k)
        log.info(f"Vector retrieval: {len(vector_results)} chunks found.")

        # ── Stage 3b: Graph Retrieval (entities + citations) ──────────────────
        all_related_ids = set()

        # Entity-based graph traversal
        query_entities = self.find_entities_in_query(reformulated_query)
        for ent_node in query_entities:
            for doc_id, _, data in self.G.in_edges(ent_node, data=True) if self.G else []:
                if data.get("relation") == "MENTIONS":
                    all_related_ids.add(doc_id)

        # Citation-based graph traversal
        for res in vector_results:
            doc_id = res["chunk"]["metadata"].get("doc_id", "")
            all_related_ids.update(self.get_citation_neighbors(doc_id))

        # ── Stage 3c: Topic-Based Graph Retrieval (LDA) ───────────────────────
        topic_related_docs = self.get_topic_related_docs(reformulated_query)
        all_related_ids.update(topic_related_docs)

        log.info(
            f"Total graph-related nodes: {len(all_related_ids)} "
            f"(entities + citations + topics)"
        )

        # ── Stage 4: Context Fusion ────────────────────────────────────────────
        context_text, sources = self._build_context(vector_results, all_related_ids)

        # ── Stage 5: LLM Answer Generation ────────────────────────────────────
        # Pass related_ids to parent generate_answer which will use our overridden _build_prompt
        result = self.generate_answer(query, vector_results, related_ids=all_related_ids)
        
        # Merge graph-specific metadata
        result.update({
            "related_nodes": list(all_related_ids),
            "reformulated_query": reformulated_query,
            "expanded_query": expanded_query,
            "topic_docs": list(topic_related_docs),
            "pipeline": "hybrid_lda_graph_transformer",
        })
        
        return result


# ── Backwards Compatibility Alias ─────────────────────────────────────────────
# Keep old class name working so api.py doesn't break
GraphRAGEngine = HybridGraphRAGEngine


if __name__ == "__main__":
    engine = HybridGraphRAGEngine()

    test_queries = [
        "Quelles sont les obligations en matière de lutte contre le blanchiment?",
        "Quelle circulaire traite des ratios de solvabilité des banques?",
        "Qu'est-ce que la BCT exige pour l'ouverture d'un compte en devises?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Question: {q}")
        res = engine.query(q)
        print(f"Answer  : {res['answer'][:300]}...")
        print(f"Sources : {res['sources']}")
        print(f"Pipeline: {res['pipeline']}")
        print(f"Reformulated: {res.get('reformulated_query', 'N/A')}")
