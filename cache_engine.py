"""
cache_engine.py
───────────────
Semantic question/answer cache.

An incoming question is embedded with the SAME multilingual transformer the RAG
engine already uses, then compared (cosine similarity) against every previously
cached question. If the closest match clears a high threshold, the stored answer
is returned instantly — skipping query reformulation, retrieval, graph traversal
and the LLM generation call entirely.

Safety: refusals and "information not available" answers are never cached. They
are not reliable enough to replay blindly, and replaying a refusal would be worse
than recomputing.
"""

import json
import logging
import re

import numpy as np

import database as db

log = logging.getLogger(__name__)

# Deliberately high — only near-duplicate questions reuse an answer, so we never
# serve a cached answer to a genuinely different question.
CACHE_SIMILARITY_THRESHOLD = 0.93

# Robust decline / "not-found" detection. Caching a decline would replay a wrong
# "I don't know" forever, so we err broad: a false positive only means we recompute
# next time. A single regex captures the structural patterns (negation + "contain /
# available / mention / access / know") across French and English; Arabic phrases
# are matched as substrings. This catches phrasings like "The provided context does
# not contain specific details…" that flat substring lists keep missing.
_DECLINE_RE = re.compile(
    r"""(
        ne\s+contien(?:t|nent)\s+pas | does\s+not\s+contain | do\s+not\s+contain | doesn'?t\s+contain
      | (?:provided\s+)?context\s+does\s+not | n'?(?:est|ont)\s+pas\s+(?:disponible|mentionn|précis|present|présent|inclus)
      | is\s+not\s+(?:mentioned|provided|available|specified|included|present)
      | not\s+(?:provided|mentioned|specified|available|included)\s+in
      | don'?t\s+have\s+(?:access|information|specific|details) | do\s+not\s+have\s+(?:access|information|details)
      | no\s+(?:specific\s+)?(?:information|details?|mention|data)\s+(?:about|on|regarding|of|available)
      | (?:n'?(?:est|sont)\s+pas\s+disponible) | pas\s+disponible | non\s+disponible
      | ne\s+(?:dispose|disposons)\s+pas | n'?ai\s+pas\s+accès | pas\s+d'?information | ne\s+figure\s+pas
      | je\s+ne\s+(?:sais\s+pas|peux\s+répondre|réponds\s+qu) | désolé
      | i\s+(?:don'?t|do\s+not)\s+know | cannot\s+(?:find|provide|answer) | unable\s+to\s+(?:find|provide|answer)
    )""",
    re.IGNORECASE | re.VERBOSE,
)
_DECLINE_SUBSTR_AR = (
    "لا أعرف", "لا أملك", "لا تتوفر", "غير متوفر", "لا توجد معلومات", "لا يمكنني", "لا أستطيع",
)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def embed_question(model, question: str) -> np.ndarray:
    """Embed + L2-normalize a question with the engine's transformer model.
    Normalizing means a plain dot product equals cosine similarity later."""
    vec = model.encode([question], convert_to_numpy=True)[0].astype(np.float32)
    return _normalize(vec)


def is_cacheable(answer: str, sources: list) -> bool:
    """A response is cacheable only if it is a grounded, non-refusal answer.
    Declines, refusals and "the context doesn't contain this" answers are never
    cached — replaying them would be worse than recomputing."""
    if not answer or not answer.strip():
        return False
    if not sources:                 # greetings, refusals and "not found" carry no real sources
        return False
    if _DECLINE_RE.search(answer):
        return False
    if any(s in answer for s in _DECLINE_SUBSTR_AR):
        return False
    return True


def lookup(vec: np.ndarray, mode: str):
    """Return (answer_dict, cache_id, similarity) for the best match above the
    threshold, or None on a miss."""
    rows = db.get_cache_entries(mode)
    if not rows:
        return None
    # Vectorized cosine: stack stored (normalized) embeddings and dot with vec.
    mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    sims = mat @ vec
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    if best_sim >= CACHE_SIMILARITY_THRESHOLD:
        row = rows[best_idx]
        return json.loads(row["answer_json"]), row["id"], best_sim
    return None


def store(question: str, vec: np.ndarray, result: dict, mode: str) -> None:
    """Persist a fresh answer (and its question embedding) for future reuse."""
    payload = {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "related_nodes": result.get("related_nodes", []),
        "pipeline": result.get("pipeline"),
    }
    db.add_cache_entry(
        question, vec.tobytes(), json.dumps(payload, ensure_ascii=False), mode
    )
