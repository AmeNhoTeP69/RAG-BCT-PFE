"""
api.py
──────
FastAPI Backend for BCT RAG & Graph RAG

Exposes endpoints for the web frontend to query the RAG and Graph RAG engines.
"""

import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

import config
from rag_engine import RAGEngine
from graph_rag_engine import GraphRAGEngine

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

app = FastAPI(title="BCT RAG API")

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
