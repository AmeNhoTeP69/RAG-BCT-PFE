"""
seed_corrections.py
───────────────────
Load the verified-corrections knowledge base (scripts/corrections_seed.json) into
bct_app.db. Idempotent: skips a correction whose `topic` already exists. Embeddings
are computed from `topic` with the engine's sentence-transformer so semantic
matching works. Run after a fresh DB to restore the curated KB behind the 92.5% eval.

    python scripts/seed_corrections.py
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
import config
import database as db
import corrections as co
from sentence_transformers import SentenceTransformer

seed = json.loads((Path(__file__).resolve().parent / "corrections_seed.json").read_text(encoding="utf-8"))
db.init_db()
conn = db.get_db()
existing = {r["topic"] for r in conn.execute("SELECT topic FROM corrections").fetchall()}
conn.close()

model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
added = 0
for c in seed:
    if c["topic"] in existing:
        continue
    emb = co.embed(model, c["topic"]).tobytes()
    db.add_correction(c["topic"], c["correction"], emb,
                      status=c.get("status", "approved"), min_sim=c.get("min_sim"))
    added += 1
print(f"Seeded {added} correction(s); {len(seed) - added} already present.")
