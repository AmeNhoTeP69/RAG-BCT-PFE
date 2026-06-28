"""
corrections.py
──────────────
Admin-gated "learning". When the assistant concedes to a correction during a
conversation, a PENDING correction is captured. An admin reviews and approves it;
approved corrections are then semantically matched to future questions and
injected into the prompt as VERIFIED NOTES, so a confirmed fix persists across
conversations — without an LLM ever silently "learning" something wrong (a human
gates every note).
"""

import logging
import re

import numpy as np

import database as db

log = logging.getLogger(__name__)

# Cosine threshold for considering an approved correction relevant to a new query.
# BCT regulatory questions cluster tightly in embedding space, so a low threshold
# matches "everything to everything". 0.62 keeps only genuinely on-subject hits.
CORRECTION_MATCH_THRESHOLD = 0.62
# Inject only the single best-matching verified note. With multiple notes a question
# could pick up cross-topic ones (a governance Q grabbing an FX-position fact); the
# top-1 is the most relevant and avoids that noise.
MAX_INJECTED = 1

# Phrases that signal the assistant accepted a correction (it conceded a mistake).
_CONCESSION_RE = re.compile(
    r"""(
        vous\s+avez\s+raison | you'?re\s+right | you\s+are\s+right
      | je\s+me\s+corrige | j'?ai\s+omis | j'?ai\s+eu\s+tort
      | effectivement,?\s+(?:la|le|les|vous|cette|ce) | en\s+effet,?\s+vous\s+avez
      | i\s+stand\s+corrected | my\s+mistake | thank\s+you\s+for\s+the\s+correct
      | correction\s+notée | merci\s+pour\s+(?:la\s+|cette\s+)?correction
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def detect_concession(answer: str) -> bool:
    """True if the assistant's answer reads as accepting a user's correction."""
    return bool(answer) and bool(_CONCESSION_RE.search(answer))


def embed(model, text: str) -> np.ndarray:
    """Embed + L2-normalize text with the engine's transformer (dot == cosine)."""
    vec = model.encode([text], convert_to_numpy=True)[0].astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def find_relevant(model, query: str, threshold: float = CORRECTION_MATCH_THRESHOLD,
                  limit: int = MAX_INJECTED) -> list[str]:
    """Return approved correction notes semantically relevant to `query`."""
    rows = db.get_approved_corrections()
    if not rows:
        return []
    qv = embed(model, query)
    scored = []
    for r in rows:
        emb = r["embedding"]
        if not emb:
            continue
        v = np.frombuffer(emb, dtype=np.float32)
        sim = float(v @ qv)
        # A correction may set its own higher bar (min_sim) so a narrow note only
        # fires for its true subject, not semantically-adjacent neighbours.
        try:
            row_min = r["min_sim"]
        except (KeyError, IndexError):
            row_min = None
        bar = max(threshold, row_min) if row_min is not None else threshold
        if sim >= bar:
            scored.append((sim, r["correction"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:limit]]
