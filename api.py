"""
api.py
──────
FastAPI Backend for BCT RAG & Graph RAG

Exposes endpoints for the web frontend to query the RAG and Graph RAG engines.
"""

import json
import logging
import sqlite3
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

import config
import database as db
import auth
import cache_engine
import analytics
import corrections
from rag_engine import RAGEngine
from graph_rag_engine import GraphRAGEngine

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the database and ensure the default admin exists on startup."""
    db.init_db()
    auth.ensure_default_admin()
    log.info("Database ready; default admin ensured.")
    yield


app = FastAPI(title="BCT RAG API", lifespan=lifespan)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
# Note: We load them lazily or global to avoid double loading memory
rag_engine = None
graph_engine = None

def get_rag_engine():
    global rag_engine
    if rag_engine is None:
        log.info("Lazy loading Standard RAG Engine...")
        rag_engine = RAGEngine()
    return rag_engine

def get_graph_engine():
    global graph_engine
    if graph_engine is None:
        log.info("Lazy loading Graph RAG Engine...")
        graph_engine = GraphRAGEngine()
    return graph_engine

# Models
class QueryRequest(BaseModel):
    query: str
    mode: str = "rag"  # "rag" or "graph"

# Endpoints
@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        if request.mode == "graph":
            engine = get_graph_engine()
        else:
            engine = get_rag_engine()
            
        result = engine.query(request.query)
        return result
    except Exception as e:
        log.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ──────────────────────────────────────────────────────────────────────────────
# Authentication
# ──────────────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    bank_name: str | None = None
    role: str = "bank"


class UpdateUserRequest(BaseModel):
    password: str | None = None
    bank_name: str | None = None
    role: str | None = None


class StatusRequest(BaseModel):
    status: str  # 'active' | 'suspended'


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    result = auth.authenticate(req.username, req.password)
    if result is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if result.get("suspended"):
        raise HTTPException(status_code=403, detail="This account is suspended. Contact the administrator.")
    return result


@app.get("/api/auth/me")
async def whoami(user: dict = Depends(auth.get_current_user)):
    return user


# ──────────────────────────────────────────────────────────────────────────────
# Admin — account management (admin role only)
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/admin/users")
async def admin_list_users(admin: dict = Depends(auth.require_admin)):
    return [db.user_to_public(u) for u in db.list_users()]


@app.post("/api/admin/users")
async def admin_create_user(req: CreateUserRequest, admin: dict = Depends(auth.require_admin)):
    username = req.username.strip()
    if not username or not req.password:
        raise HTTPException(status_code=400, detail="Username and password are required")
    if req.role not in ("admin", "bank"):
        raise HTTPException(status_code=400, detail="Invalid role")
    pwd_hash, salt = auth.hash_password(req.password)
    try:
        uid = db.create_user(username, pwd_hash, salt, role=req.role, bank_name=req.bank_name)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="This username already exists")
    return db.user_to_public(db.get_user_by_id(uid))


@app.put("/api/admin/users/{user_id}")
async def admin_update_user(user_id: int, req: UpdateUserRequest, admin: dict = Depends(auth.require_admin)):
    row = db.get_user_by_id(user_id)
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    if req.role is not None and req.role not in ("admin", "bank"):
        raise HTTPException(status_code=400, detail="Invalid role")
    if auth.is_default_admin(row) and req.role is not None and req.role != "admin":
        raise HTTPException(status_code=400, detail="The default admin account's role cannot be changed")
    pwd_hash = salt = None
    if req.password:
        pwd_hash, salt = auth.hash_password(req.password)
    db.update_user(user_id, password_hash=pwd_hash, salt=salt, bank_name=req.bank_name, role=req.role)
    return db.user_to_public(db.get_user_by_id(user_id))


@app.post("/api/admin/users/{user_id}/status")
async def admin_set_status(user_id: int, req: StatusRequest, admin: dict = Depends(auth.require_admin)):
    if req.status not in ("active", "suspended"):
        raise HTTPException(status_code=400, detail="Invalid status")
    row = db.get_user_by_id(user_id)
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    if auth.is_default_admin(row) and req.status != "active":
        raise HTTPException(status_code=400, detail="The default admin account cannot be suspended")
    if admin["id"] == user_id and req.status != "active":
        raise HTTPException(status_code=400, detail="You cannot suspend your own account")
    db.update_user(user_id, status=req.status)
    return db.user_to_public(db.get_user_by_id(user_id))


@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(user_id: int, admin: dict = Depends(auth.require_admin)):
    row = db.get_user_by_id(user_id)
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    if auth.is_default_admin(row):
        raise HTTPException(status_code=400, detail="The default admin account cannot be deleted")
    if admin["id"] == user_id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account")
    db.delete_user(user_id)
    return {"deleted": True, "id": user_id}


# ──────────────────────────────────────────────────────────────────────────────
# Admin — analytics dashboard (admin role only)
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/admin/analytics")
async def admin_analytics(days: int = 30, admin: dict = Depends(auth.require_admin)):
    """Aggregated, real-usage analytics for the dashboard (KPIs + 5 charts)."""
    return analytics.build_dashboard(days=days)


# ──────────────────────────────────────────────────────────────────────────────
# Admin — verified corrections (admin-gated cross-session learning)
# ──────────────────────────────────────────────────────────────────────────────
class CreateCorrectionRequest(BaseModel):
    topic: str
    correction: str


class UpdateCorrectionRequest(BaseModel):
    topic: str | None = None
    correction: str | None = None
    status: str | None = None  # 'pending' | 'approved' | 'rejected'


def _embed_topic(topic: str) -> bytes | None:
    """Embed a correction's topic using the (lighter) standard RAG engine model."""
    try:
        eng = get_rag_engine()
        if getattr(eng, "model", None) is not None:
            return corrections.embed(eng.model, topic).tobytes()
    except Exception as e:
        log.warning(f"Correction embed skipped: {e}")
    return None


@app.get("/api/admin/corrections")
async def admin_list_corrections(status: str | None = None, admin: dict = Depends(auth.require_admin)):
    return [db.correction_to_public(c) for c in db.list_corrections(status)]


@app.post("/api/admin/corrections")
async def admin_create_correction(req: CreateCorrectionRequest, admin: dict = Depends(auth.require_admin)):
    topic, correction = req.topic.strip(), req.correction.strip()
    if not topic or not correction:
        raise HTTPException(status_code=400, detail="Topic and correction are required")
    # Admin-authored notes are trusted → stored directly as approved.
    cid = db.add_correction(topic, correction, _embed_topic(topic),
                            status="approved", created_by=admin["id"])
    return db.correction_to_public(db.get_correction(cid))


@app.put("/api/admin/corrections/{corr_id}")
async def admin_update_correction(corr_id: int, req: UpdateCorrectionRequest, admin: dict = Depends(auth.require_admin)):
    row = db.get_correction(corr_id)
    if not row:
        raise HTTPException(status_code=404, detail="Correction not found")
    if req.status is not None and req.status not in ("pending", "approved", "rejected"):
        raise HTTPException(status_code=400, detail="Statut invalide")
    db.update_correction(corr_id, correction=req.correction, topic=req.topic,
                         status=req.status, reviewed_by=admin["id"])
    # Re-embed if the topic text changed, so semantic matching stays accurate.
    if req.topic:
        emb = _embed_topic(req.topic)
        if emb is not None:
            conn = db.get_db()
            try:
                conn.execute("UPDATE corrections SET embedding = ? WHERE id = ?", (emb, corr_id))
                conn.commit()
            finally:
                conn.close()
    return db.correction_to_public(db.get_correction(corr_id))


@app.delete("/api/admin/corrections/{corr_id}")
async def admin_delete_correction(corr_id: int, admin: dict = Depends(auth.require_admin)):
    if not db.get_correction(corr_id):
        raise HTTPException(status_code=404, detail="Correction not found")
    db.delete_correction(corr_id)
    return {"deleted": True, "id": corr_id}


# ──────────────────────────────────────────────────────────────────────────────
# Authenticated chat (persistent conversations + continuous context)
# ──────────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    mode: str = "rag"                       # "rag" | "graph"
    conversation_id: int | None = None      # None -> start a new conversation


def _derive_title(text: str) -> str:
    """Conversation title from the first user message."""
    t = " ".join(text.strip().split())
    return (t[:48] + "…") if len(t) > 48 else (t or "Nouvelle conversation")


def _own_conversation_or_404(conv_id: int, user: dict):
    conv = db.get_conversation(conv_id)
    if not conv or conv["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest, user: dict = Depends(auth.get_current_user)):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    mode = "graph" if req.mode == "graph" else "rag"

    # Load prior context for an existing conversation (verifying ownership).
    history: list[dict] = []
    conv_id = None
    if req.conversation_id is not None:
        conv = _own_conversation_or_404(req.conversation_id, user)
        conv_id = conv["id"]
        history = db.get_recent_history(conv_id, limit=6)

    # The engine is needed either way — its transformer powers the cache embedding.
    try:
        engine = get_graph_engine() if mode == "graph" else get_rag_engine()
    except Exception as e:
        log.error(f"Error loading engine: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    t0 = time.perf_counter()
    q_vec = None
    notes = []
    if getattr(engine, "model", None) is not None:
        try:
            q_vec = cache_engine.embed_question(engine.model, req.query)
        except Exception as e:
            log.warning(f"Embedding skipped: {e}")
        # Relevant admin-approved corrections to inject as authoritative notes.
        try:
            notes = corrections.find_relevant(engine.model, req.query)
        except Exception as e:
            log.warning(f"Corrections lookup skipped: {e}")

    # ── Semantic cache ────────────────────────────────────────────────────────
    # Look up on every turn (a high threshold prevents false hits); only store
    # context-free first-turn answers. Skip the cache entirely when a verified
    # correction applies, so we never replay a pre-correction answer.
    cached = False
    hit = None
    if not notes and q_vec is not None:
        try:
            hit = cache_engine.lookup(q_vec, mode)
        except Exception as e:
            log.warning(f"Cache lookup skipped: {e}")

    if hit is not None:
        payload, cache_id, sim = hit
        db.increment_cache_hit(cache_id)
        result = payload
        cached = True
        log.info(f"Semantic cache HIT (sim={sim:.3f}) for: {req.query[:60]}")
    else:
        try:
            result = engine.query(req.query, history=history, notes=notes)
        except Exception as e:
            log.error(f"Error processing chat: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

        # Capture a candidate correction when the assistant concedes mid-conversation.
        # Stored as PENDING — an admin must approve it before it ever influences
        # future answers (so a wrong concession can't propagate).
        if history and corrections.detect_concession(result.get("answer", "")):
            try:
                emb = (corrections.embed(engine.model, req.query).tobytes()
                       if getattr(engine, "model", None) is not None else None)
                db.add_correction(
                    topic=req.query,
                    correction=result.get("answer", "")[:1500],
                    embedding=emb, status="pending",
                    source_conversation_id=conv_id, created_by=user["id"],
                )
                log.info("Captured a PENDING correction from a conversation concession.")
            except Exception as e:
                log.warning(f"Correction capture skipped: {e}")

        # Store only self-contained (first-turn) grounded answers, and never when
        # a correction was injected (that answer is correction-influenced).
        if (not history and not notes and q_vec is not None
                and cache_engine.is_cacheable(result.get("answer", ""), result.get("sources", []))):
            try:
                cache_engine.store(req.query, q_vec, result, mode)
            except Exception as e:
                log.warning(f"Cache store skipped: {e}")

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    chunks = result.get("retrieved_chunks", [])
    related = result.get("related_nodes", [])

    # Persist only on success (avoids orphan conversations on engine failure).
    if conv_id is None:
        conv_id = db.create_conversation(user["id"], title=_derive_title(req.query))
    db.add_message(conv_id, "user", req.query, mode=mode)
    msg_id = db.add_message(
        conv_id, "ai", answer,
        sources_json=json.dumps(sources, ensure_ascii=False),
        chunks_json=json.dumps(chunks, ensure_ascii=False),
        related_json=json.dumps(related, ensure_ascii=False),
        mode=mode, cached=cached,
    )
    db.touch_conversation(conv_id)

    # Analytics: log every answered query with real metadata (best-effort).
    try:
        topic_id, topic_label = analytics.infer_topic(req.query)
        db.add_query_log(
            user_id=user["id"], bank_name=user.get("bank_name"),
            conversation_id=conv_id, question=req.query, mode=mode,
            topic_id=topic_id, topic_label=topic_label,
            response_time_ms=elapsed_ms, cache_hit=cached,
        )
    except Exception as e:
        log.warning(f"Analytics logging skipped: {e}")

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": chunks,
        "related_nodes": related,
        "pipeline": result.get("pipeline"),
        "conversation_id": conv_id,
        "message_id": msg_id,
        "cached": cached,
        "mode": mode,
    }


@app.get("/api/conversations")
async def list_user_conversations(user: dict = Depends(auth.get_current_user)):
    rows = db.list_conversations(user["id"])
    return [
        {
            "id": r["id"],
            "title": r["title"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "message_count": r["message_count"],
            "last_message_at": r["last_message_at"] or r["updated_at"],
        }
        for r in rows
    ]


@app.get("/api/conversations/{conv_id}")
async def get_conversation_detail(conv_id: int, user: dict = Depends(auth.get_current_user)):
    _own_conversation_or_404(conv_id, user)
    out = []
    for m in db.get_messages(conv_id):
        out.append({
            "id": m["id"],
            "role": m["role"],
            "text": m["content"],
            "sources": json.loads(m["sources_json"]) if m["sources_json"] else [],
            "chunks": json.loads(m["chunks_json"]) if m["chunks_json"] else [],
            "relatedNodes": json.loads(m["related_json"]) if m["related_json"] else [],
            "mode": m["mode"],
            "cached": bool(m["cached"]),
            "created_at": m["created_at"],
        })
    return {"id": conv_id, "messages": out}


@app.delete("/api/conversations/{conv_id}")
async def delete_user_conversation(conv_id: int, user: dict = Depends(auth.get_current_user)):
    _own_conversation_or_404(conv_id, user)
    db.delete_conversation(conv_id)
    return {"deleted": True, "id": conv_id}


# Serve static files (React build output goes into static/)
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)

# Mount /assets for Vite's hashed JS/CSS bundles
assets_path = static_path / "assets"
if assets_path.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    """Serve React app entry point."""
    index_file = static_path / "index.html"
    if not index_file.exists():
        return """<html><body>
            <h1 style='font-family:sans-serif;color:#D4AF37'>BCT Intel-Graph</h1>
            <p style='color:#aaa'>Build not found. Run: <code>cd frontend && npm run build</code></p>
        </body></html>"""
    with open(index_file, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/{full_path:path}", response_class=HTMLResponse)
async def spa_fallback(full_path: str):
    """Catch-all: return index.html for React client-side routing."""
    # Don't intercept /api routes or actual static files
    if full_path.startswith("api/") or full_path.startswith("assets/"):
        from fastapi import HTTPException
        raise HTTPException(status_code=404)
    index_file = static_path / "index.html"
    if not index_file.exists():
        return "Not found"
    with open(index_file, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
