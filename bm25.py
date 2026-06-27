"""
bm25.py
───────
Minimal BM25 lexical index (pure standard-library + math) used to complement the
dense FAISS retrieval. Dense embeddings capture meaning and cross-lingual matches
but, for natural-language questions, can rank chunks that merely *mention* a term
above the precise section that *defines* it. BM25 scores exact term overlap
(IDF-weighted, length-normalized), so fusing the two surfaces precise sections
(e.g. "Exigences de fonds propres") without losing semantic recall.

No third-party dependency — keeps the single-server deployment reproducible.
"""

import math
import re
from collections import Counter

# Tokens: Latin words (with French accents) and digits, OR Arabic words.
_TOKEN_RE = re.compile(r"[a-zàâäéèêëîïôùûüç0-9]+|[؀-ۿ]+", re.IGNORECASE)

# Light multilingual stop-word list (French / English) — drops words that carry
# no retrieval signal so the lexical match focuses on content terms.
_STOP = set("""
le la les un une des du de d l au aux et en à a dans par pour sur avec sans sous
ce cet cette ces son sa ses leur leurs qui que quoi dont où est sont être quel
quelle quelles quels comment pourquoi quand sur vers chez ne pas plus moins
the a an of and or to in on for with without by is are be was were what which who
whom whose how why when where as at from this that these those it its their
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 1 and t not in _STOP]


class BM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.tf = [Counter(d) for d in corpus_tokens]
        df = Counter()
        for d in corpus_tokens:
            for t in set(d):
                df[t] += 1
        # Robertson/Sparck-Jones IDF (always positive variant).
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query_tokens: list[str], candidate_idxs) -> dict:
        """BM25 score for each candidate document index against the query."""
        out = {}
        for i in candidate_idxs:
            dl = self.doc_len[i]
            tfi = self.tf[i]
            s = 0.0
            for t in query_tokens:
                f = tfi.get(t, 0)
                if f and t in self.idf:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = s
        return out


def build_from_chunks(chunks: list) -> "BM25 | None":
    """Build a BM25 index from chunk dicts (text + section_header). Returns None
    on any failure so the caller can fall back to dense-only retrieval safely."""
    try:
        corpus = []
        for c in chunks:
            meta = c.get("metadata", {}) if isinstance(c, dict) else {}
            text = (c.get("text", "") if isinstance(c, dict) else "")
            header = meta.get("section_header", "") or ""
            title = meta.get("title", "") or ""
            # Header/title repeated so section labels weigh in lexical matching.
            corpus.append(tokenize(f"{title} {header} {header} {text}"))
        return BM25(corpus)
    except Exception:
        return None
