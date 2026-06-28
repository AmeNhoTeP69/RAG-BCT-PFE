"""
database.py
───────────
SQLite persistence layer for the BCT RAG web application.

A single self-contained database file (no external server required) holds:
  - users          : admin + bank accounts (authentication)
  - conversations  : per-user chat threads
  - messages       : individual turns inside a conversation
  - qa_cache       : semantic question/answer cache
  - query_logs     : analytics event log (one row per answered query)

Every helper opens a short-lived connection so the module is safe to call from
FastAPI's threadpool without sharing a single connection across threads. The
volume here (a defense demo) is tiny, so per-call connections are more than fast
enough and keep the code simple and robust.
"""

import sqlite3
from datetime import datetime, timezone

import config

# Single-file database living next to the code.
DB_PATH = config.BASE_DIR / "bct_app.db"


# ── Connection ────────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    """Open a new connection with row access by column name + FK enforcement."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def utcnow() -> str:
    """ISO-8601 UTC timestamp used for all created_at / updated_at columns."""
    return datetime.now(timezone.utc).isoformat()


# ── Schema ────────────────────────────────────────────────────────────────────
def init_db() -> None:
    """Create every table if it does not exist. Idempotent — safe to call on
    every startup."""
    conn = get_db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt          TEXT NOT NULL,
            role          TEXT NOT NULL DEFAULT 'bank',    -- 'admin' | 'bank'
            bank_name     TEXT,
            status        TEXT NOT NULL DEFAULT 'active',   -- 'active' | 'suspended'
            created_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            title      TEXT NOT NULL DEFAULT 'Nouvelle conversation',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role            TEXT NOT NULL,               -- 'user' | 'ai'
            content         TEXT NOT NULL,
            sources_json    TEXT,
            chunks_json     TEXT,
            related_json    TEXT,
            mode            TEXT,
            cached          INTEGER NOT NULL DEFAULT 0,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS qa_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            question    TEXT NOT NULL,
            embedding   BLOB NOT NULL,
            answer_json TEXT NOT NULL,
            mode        TEXT,
            hit_count   INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS query_logs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          INTEGER,
            bank_name        TEXT,
            conversation_id  INTEGER,
            question         TEXT,
            mode             TEXT,
            topic_id         INTEGER,
            topic_label      TEXT,
            response_time_ms INTEGER,
            cache_hit        INTEGER NOT NULL DEFAULT 0,
            created_at       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS corrections (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            topic                TEXT NOT NULL,          -- subject/question the correction is about
            correction           TEXT NOT NULL,          -- the verified note injected into future answers
            embedding            BLOB,                   -- embedding of `topic` for semantic matching
            status               TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'approved' | 'rejected'
            source_conversation_id INTEGER,
            created_by           INTEGER,
            reviewed_by          INTEGER,
            created_at           TEXT NOT NULL,
            reviewed_at          TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_conv_user   ON conversations(user_id);
        CREATE INDEX IF NOT EXISTS idx_msg_conv    ON messages(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_logs_created ON query_logs(created_at);
        CREATE INDEX IF NOT EXISTS idx_logs_user   ON query_logs(user_id);
        CREATE INDEX IF NOT EXISTS idx_corr_status ON corrections(status);
        """
    )
    # Migration: per-correction minimum match similarity. A "narrow" correction
    # (e.g. a specific account-type rule) can require a tighter cosine than the
    # global threshold so it only fires for its true subject, not near-neighbours.
    try:
        conn.execute("ALTER TABLE corrections ADD COLUMN min_sim REAL")
    except Exception:
        pass  # column already exists
    conn.commit()
    conn.close()


# ── User CRUD ─────────────────────────────────────────────────────────────────
# These store/read raw rows only. Password hashing lives in auth.py so this
# module stays free of any crypto / auth import (avoids a circular import).

def create_user(username: str, password_hash: str, salt: str,
                role: str = "bank", bank_name: str | None = None,
                status: str = "active") -> int:
    """Insert a user and return the new id. Raises sqlite3.IntegrityError if the
    username already exists (caller maps this to a 409)."""
    conn = get_db()
    try:
        cur = conn.execute(
            """INSERT INTO users (username, password_hash, salt, role, bank_name, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (username, password_hash, salt, role, bank_name, status, utcnow()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_user_by_username(username: str) -> sqlite3.Row | None:
    conn = get_db()
    try:
        return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> sqlite3.Row | None:
    conn = get_db()
    try:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    finally:
        conn.close()


def list_users() -> list[sqlite3.Row]:
    conn = get_db()
    try:
        return conn.execute(
            "SELECT * FROM users ORDER BY role = 'admin' DESC, bank_name, username"
        ).fetchall()
    finally:
        conn.close()


def count_users() -> int:
    conn = get_db()
    try:
        return conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    finally:
        conn.close()


def update_user(user_id: int, *, password_hash: str | None = None,
                salt: str | None = None, bank_name: str | None = None,
                role: str | None = None, status: str | None = None) -> None:
    """Update only the provided fields. Passing None leaves a column unchanged."""
    fields, values = [], []
    if password_hash is not None and salt is not None:
        fields += ["password_hash = ?", "salt = ?"]
        values += [password_hash, salt]
    if bank_name is not None:
        fields.append("bank_name = ?"); values.append(bank_name)
    if role is not None:
        fields.append("role = ?"); values.append(role)
    if status is not None:
        fields.append("status = ?"); values.append(status)
    if not fields:
        return
    values.append(user_id)
    conn = get_db()
    try:
        conn.execute(f"UPDATE users SET {', '.join(fields)} WHERE id = ?", values)
        conn.commit()
    finally:
        conn.close()


def delete_user(user_id: int) -> None:
    conn = get_db()
    try:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


def user_to_public(row: sqlite3.Row) -> dict:
    """Serialize a user row for API responses, omitting password_hash + salt."""
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "bank_name": row["bank_name"],
        "status": row["status"],
        "created_at": row["created_at"],
    }


# ── Conversation CRUD ─────────────────────────────────────────────────────────
def create_conversation(user_id: int, title: str = "Nouvelle conversation") -> int:
    conn = get_db()
    try:
        now = utcnow()
        cur = conn.execute(
            "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (user_id, title[:120], now, now),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_conversation(conv_id: int) -> sqlite3.Row | None:
    conn = get_db()
    try:
        return conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    finally:
        conn.close()


def list_conversations(user_id: int) -> list[sqlite3.Row]:
    """Conversations for a user, newest activity first, with a message count and
    the timestamp of the last message (for the 'time elapsed' label)."""
    conn = get_db()
    try:
        return conn.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(m.id)        AS message_count,
                   MAX(m.created_at)  AS last_message_at
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.user_id = ?
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            """,
            (user_id,),
        ).fetchall()
    finally:
        conn.close()


def update_conversation_title(conv_id: int, title: str) -> None:
    conn = get_db()
    try:
        conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title[:120], conv_id))
        conn.commit()
    finally:
        conn.close()


def touch_conversation(conv_id: int) -> None:
    """Bump updated_at so the conversation rises to the top of the sidebar."""
    conn = get_db()
    try:
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (utcnow(), conv_id))
        conn.commit()
    finally:
        conn.close()


def delete_conversation(conv_id: int) -> None:
    conn = get_db()
    try:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))  # cascades to messages
        conn.commit()
    finally:
        conn.close()


# ── Message CRUD ──────────────────────────────────────────────────────────────
def add_message(conversation_id: int, role: str, content: str,
                sources_json: str | None = None, chunks_json: str | None = None,
                related_json: str | None = None, mode: str | None = None,
                cached: bool = False) -> int:
    conn = get_db()
    try:
        cur = conn.execute(
            """INSERT INTO messages
                 (conversation_id, role, content, sources_json, chunks_json,
                  related_json, mode, cached, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (conversation_id, role, content, sources_json, chunks_json,
             related_json, mode, 1 if cached else 0, utcnow()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_messages(conversation_id: int) -> list[sqlite3.Row]:
    conn = get_db()
    try:
        return conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
    finally:
        conn.close()


def get_recent_history(conversation_id: int, limit: int = 6) -> list[dict]:
    """Return the last `limit` turns as [{role, content}] in chronological order,
    for feeding conversational context to the LLM and the retriever."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    finally:
        conn.close()


# ── Semantic cache CRUD ───────────────────────────────────────────────────────
def add_cache_entry(question: str, embedding: bytes, answer_json: str, mode: str) -> int:
    conn = get_db()
    try:
        cur = conn.execute(
            """INSERT INTO qa_cache (question, embedding, answer_json, mode, hit_count, created_at)
               VALUES (?, ?, ?, ?, 0, ?)""",
            (question, embedding, answer_json, mode, utcnow()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_cache_entries(mode: str | None = None) -> list[sqlite3.Row]:
    """All cached entries (optionally for one mode) — id, embedding, answer_json."""
    conn = get_db()
    try:
        if mode:
            return conn.execute(
                "SELECT id, embedding, answer_json FROM qa_cache WHERE mode = ?", (mode,)
            ).fetchall()
        return conn.execute("SELECT id, embedding, answer_json FROM qa_cache").fetchall()
    finally:
        conn.close()


def increment_cache_hit(cache_id: int) -> None:
    conn = get_db()
    try:
        conn.execute("UPDATE qa_cache SET hit_count = hit_count + 1 WHERE id = ?", (cache_id,))
        conn.commit()
    finally:
        conn.close()


def count_cache_entries() -> int:
    conn = get_db()
    try:
        return conn.execute("SELECT COUNT(*) AS c FROM qa_cache").fetchone()["c"]
    finally:
        conn.close()


# ── Analytics event log ───────────────────────────────────────────────────────
def add_query_log(user_id: int | None, bank_name: str | None, conversation_id: int | None,
                  question: str, mode: str, topic_id: int | None, topic_label: str | None,
                  response_time_ms: int | None, cache_hit: bool) -> None:
    conn = get_db()
    try:
        conn.execute(
            """INSERT INTO query_logs
                 (user_id, bank_name, conversation_id, question, mode, topic_id,
                  topic_label, response_time_ms, cache_hit, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, bank_name, conversation_id, question, mode, topic_id,
             topic_label, response_time_ms, 1 if cache_hit else 0, utcnow()),
        )
        conn.commit()
    finally:
        conn.close()


# ── Verified corrections (admin-gated cross-session learning) ─────────────────
def add_correction(topic: str, correction: str, embedding: bytes | None,
                   status: str = "pending", source_conversation_id: int | None = None,
                   created_by: int | None = None, min_sim: float | None = None) -> int:
    conn = get_db()
    try:
        cur = conn.execute(
            """INSERT INTO corrections
                 (topic, correction, embedding, status, source_conversation_id, created_by, created_at, min_sim)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (topic, correction, embedding, status, source_conversation_id, created_by, utcnow(), min_sim),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_corrections(status: str | None = None) -> list[sqlite3.Row]:
    conn = get_db()
    try:
        if status:
            return conn.execute(
                "SELECT * FROM corrections WHERE status = ? ORDER BY created_at DESC", (status,)
            ).fetchall()
        return conn.execute("SELECT * FROM corrections ORDER BY created_at DESC").fetchall()
    finally:
        conn.close()


def get_correction(corr_id: int) -> sqlite3.Row | None:
    conn = get_db()
    try:
        return conn.execute("SELECT * FROM corrections WHERE id = ?", (corr_id,)).fetchone()
    finally:
        conn.close()


def get_approved_corrections() -> list[sqlite3.Row]:
    conn = get_db()
    try:
        return conn.execute(
            "SELECT id, topic, correction, embedding, min_sim FROM corrections WHERE status = 'approved'"
        ).fetchall()
    finally:
        conn.close()


def update_correction(corr_id: int, *, correction: str | None = None, topic: str | None = None,
                      status: str | None = None, reviewed_by: int | None = None) -> None:
    fields, values = [], []
    if correction is not None:
        fields.append("correction = ?"); values.append(correction)
    if topic is not None:
        fields.append("topic = ?"); values.append(topic)
    if status is not None:
        fields.append("status = ?"); values.append(status)
        fields.append("reviewed_at = ?"); values.append(utcnow())
        fields.append("reviewed_by = ?"); values.append(reviewed_by)
    if not fields:
        return
    values.append(corr_id)
    conn = get_db()
    try:
        conn.execute(f"UPDATE corrections SET {', '.join(fields)} WHERE id = ?", values)
        conn.commit()
    finally:
        conn.close()


def delete_correction(corr_id: int) -> None:
    conn = get_db()
    try:
        conn.execute("DELETE FROM corrections WHERE id = ?", (corr_id,))
        conn.commit()
    finally:
        conn.close()


def correction_to_public(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "topic": row["topic"],
        "correction": row["correction"],
        "status": row["status"],
        "source_conversation_id": row["source_conversation_id"],
        "created_at": row["created_at"],
        "reviewed_at": row["reviewed_at"] if "reviewed_at" in row.keys() else None,
    }
