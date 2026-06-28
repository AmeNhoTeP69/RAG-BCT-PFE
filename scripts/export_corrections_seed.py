"""
export_corrections_seed.py
──────────────────────────
Dump the approved verified-corrections (the curated knowledge base that lifts
eval accuracy to 92.5%) to a committable JSON seed. The live store lives in the
git-ignored bct_app.db, so this preserves the facts outside the DB. Re-load with
scripts/seed_corrections.py (embeddings are recomputed from `topic` on load).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import database as db

db.init_db()
conn = db.get_db()
rows = conn.execute(
    "SELECT topic, correction, status, min_sim FROM corrections ORDER BY id"
).fetchall()
conn.close()

seed = [{"topic": r["topic"], "correction": r["correction"],
         "status": r["status"], "min_sim": r["min_sim"]} for r in rows]
out = Path(__file__).resolve().parent / "corrections_seed.json"
out.write_text(json.dumps(seed, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Exported {len(seed)} corrections to {out.name}")
