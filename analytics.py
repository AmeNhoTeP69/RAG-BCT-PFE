"""
analytics.py
────────────
Usage analytics for the admin dashboard.

  - infer_topic()     : assign an incoming question to one of the 8 LDA topics,
                        independently of which engine answered it (lazy-loads the
                        LDA model once; shared across all requests).
  - build_dashboard() : aggregate the query_logs table into everything the
                        dashboard needs — KPIs + the 5 charts — in one payload,
                        all from REAL logged usage (never fabricated).
"""

import logging
import re
from datetime import datetime, timezone, timedelta

import config
import database as db

log = logging.getLogger(__name__)

# ── Lazy LDA topic inference ──────────────────────────────────────────────────
_lda = None
_dict = None
_lda_attempted = False


def _ensure_lda() -> None:
    global _lda, _dict, _lda_attempted
    if _lda_attempted:
        return
    _lda_attempted = True
    try:
        from gensim.models import LdaModel
        from gensim import corpora
        if config.LDA_MODEL_PATH.exists() and config.LDA_DICTIONARY_PATH.exists():
            _lda = LdaModel.load(str(config.LDA_MODEL_PATH))
            _dict = corpora.Dictionary.load(str(config.LDA_DICTIONARY_PATH))
            log.info("Analytics: LDA topic model loaded for query tagging.")
    except Exception as e:
        log.warning(f"Analytics: LDA topic inference unavailable ({e}).")


def infer_topic(question: str):
    """Return (topic_id, topic_label) for the question's dominant LDA topic, or
    (None, None) when it cannot be inferred (e.g. an Arabic-only question, since
    the LDA model is French — those are simply logged without a topic)."""
    _ensure_lda()
    if not _lda or not _dict:
        return None, None
    tokens = re.sub(r"[^a-zàâäéèêëîïôùûüç\s]", " ", question.lower()).split()
    tokens = [t for t in tokens if len(t) >= config.LDA_MIN_TOKEN_LEN]
    if not tokens:
        return None, None
    bow = _dict.doc2bow(tokens)
    if not bow:
        return None, None
    dist = _lda.get_document_topics(bow, minimum_probability=0.0)
    if not dist:
        return None, None
    tid, _prob = max(dist, key=lambda x: x[1])
    label = (config.LDA_TOPIC_LABELS[tid]
             if tid < len(config.LDA_TOPIC_LABELS) else f"Topic {tid}")
    return int(tid), label


# ── Dashboard aggregation ─────────────────────────────────────────────────────
def _day_list(days: int) -> list[str]:
    """Last `days` calendar days as YYYY-MM-DD, oldest first."""
    today = datetime.now(timezone.utc).date()
    return [(today - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]


def build_dashboard(days: int = 30) -> dict:
    days = max(1, min(days, 365))
    day_list = _day_list(days)
    cutoff_iso = day_list[0] + "T00:00:00+00:00"
    cutoff7_iso = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    conn = db.get_db()
    try:
        total = conn.execute("SELECT COUNT(*) c FROM query_logs").fetchone()["c"]

        active7 = conn.execute(
            "SELECT COUNT(DISTINCT user_id) c FROM query_logs WHERE created_at >= ?",
            (cutoff7_iso,),
        ).fetchone()["c"]

        # Average response time over the REAL pipeline (cache hits ~0ms would skew
        # the headline 'how fast does the AI answer' number, so exclude them).
        avg_rt = conn.execute(
            "SELECT AVG(response_time_ms) a FROM query_logs WHERE cache_hit = 0 AND response_time_ms IS NOT NULL"
        ).fetchone()["a"]

        hits = conn.execute("SELECT COUNT(*) c FROM query_logs WHERE cache_hit = 1").fetchone()["c"]
        cache_rate = (hits / total) if total else 0.0

        # Chart 1 — questions per day (continuous axis, zero-filled).
        rows = conn.execute(
            "SELECT substr(created_at,1,10) d, COUNT(*) c FROM query_logs WHERE created_at >= ? GROUP BY d",
            (cutoff_iso,),
        ).fetchall()
        by_day = {r["d"]: r["c"] for r in rows}
        questions_per_day = [{"date": d, "count": by_day.get(d, 0)} for d in day_list]

        # Chart 2 — topic distribution across all 8 LDA labels (always all 8).
        trows = conn.execute(
            "SELECT topic_id, COUNT(*) c FROM query_logs WHERE topic_id IS NOT NULL GROUP BY topic_id"
        ).fetchall()
        tcount = {r["topic_id"]: r["c"] for r in trows}
        topics = [
            {"topic_id": i, "label": config.LDA_TOPIC_LABELS[i], "count": tcount.get(i, 0)}
            for i in range(len(config.LDA_TOPIC_LABELS))
        ]

        # Chart 3 — RAG vs Graph usage.
        modes = {"rag": 0, "graph": 0}
        for r in conn.execute("SELECT mode, COUNT(*) c FROM query_logs GROUP BY mode").fetchall():
            if r["mode"] in modes:
                modes[r["mode"]] = r["c"]

        # Chart 4 — most active banks / users.
        urows = conn.execute(
            """
            SELECT COALESCE(NULLIF(q.bank_name, ''), u.username, 'Inconnu') AS label,
                   COUNT(*) AS c
            FROM query_logs q
            LEFT JOIN users u ON u.id = q.user_id
            GROUP BY label
            ORDER BY c DESC
            LIMIT 8
            """
        ).fetchall()
        top_users = [{"label": r["label"], "count": r["c"]} for r in urows]

        # Chart 5 — cache hit rate per day.
        crows = conn.execute(
            """SELECT substr(created_at,1,10) d, COUNT(*) total, SUM(cache_hit) hits
               FROM query_logs WHERE created_at >= ? GROUP BY d""",
            (cutoff_iso,),
        ).fetchall()
        cmap = {r["d"]: (r["total"], r["hits"] or 0) for r in crows}
        cache_rate_per_day = []
        for d in day_list:
            tot, ht = cmap.get(d, (0, 0))
            cache_rate_per_day.append({"date": d, "rate": round(ht / tot, 4) if tot else 0.0})
    finally:
        conn.close()

    return {
        "kpis": {
            "total_questions": total,
            "active_users_7d": active7,
            "avg_response_time_ms": int(avg_rt) if avg_rt else 0,
            "cache_hit_rate": round(cache_rate, 4),
        },
        "questions_per_day": questions_per_day,
        "topics": topics,
        "modes": modes,
        "top_users": top_users,
        "cache_rate_per_day": cache_rate_per_day,
        "range_days": days,
    }
