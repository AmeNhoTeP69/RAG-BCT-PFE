"""
run_evaluation.py
─────────────────
Batch evaluation runner for the BCT RAG system (RandomForce-style methodology).

Runs the 200-question suite (eval_questions.json) through the REAL engine — bypassing
the web cache so it measures true pipeline accuracy — in batches, with a delay
between queries to respect Groq free-tier rate limits. It prints a clean summary
table and saves full results (accuracy broken down by category, language and
difficulty) to JSON.

Examples
--------
  # One batch (the first 20 questions) against the Hybrid Graph RAG
  python run_evaluation.py --mode graph --batch 1 --batch-size 20 --delay 2

  # The whole suite against the standard RAG, 3s between calls
  python run_evaluation.py --mode rag --all --batch-size 20 --delay 3

Honest results are the point: if accuracy is below 95%, that is reported as-is.
The methodology — balanced suite, documented expected behavior, transparent
scoring — is what matters.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent
QUESTIONS_PATH = BASE / "eval_questions.json"


# ── Pass/fail classification (reused & hardened from full_test.py) ─────────────
_REFUSAL_MARKERS = (
    # French
    "désolé", "desolé", "je ne réponds qu", "je ne reponds qu",
    "je ne peux répondre qu", "je ne peux repondre qu",
    # English
    "sorry, i can only", "i can only answer", "only answer questions about",
    "i can only respond to", "outside the scope of bct", "not related to bct",
    # Arabic
    "آسف", "لا أجيب إلا", "أسئلة المتعلقة بالبنك",
)
_NOT_FOUND_MARKERS = (
    # French — explicit "not available / I don't have access" only (avoid hedges)
    "n'est pas disponible", "pas disponible", "non disponible",
    "ne trouve pas", "aucune information", "aucun document",
    "je ne sais pas", "n'ai pas accès", "ne dispose pas", "ne disposons pas",
    # English — explicit declines only
    "not available", "i don't know", "i do not know",
    "don't have access", "do not have access",
    "don't have information", "do not have information", "no information about",
    "don't have specific information", "do not have specific information",
    "no specific information", "don't have the", "do not contain",
    "i don't have details", "i do not have details",
    # clear "topic not covered" declines (caught honestly, not counted as answers)
    "do not specifically address", "does not specifically address",
    "do not address", "does not address", "do not specifically mention",
    "not addressed in", "is not addressed", "isn't addressed",
    "not covered in", "is not covered", "isn't covered",
    "not mentioned in", "not specified in the", "do not provide information",
    "ne traite pas", "ne traitent pas", "n'aborde pas", "n'abordent pas",
    "n'est pas abordé", "n'est pas abordée", "pas abordé dans", "pas abordée dans",
    "ne sont pas abordés", "ne sont pas abordées",
    "n'est pas traité", "n'est pas traitée", "ne figure pas", "ne figurent pas",
    "n'est pas mentionné", "n'est pas mentionnée", "n'est pas précisé",
    "n'est pas couvert", "n'est pas couverte", "ne contiennent pas",
    # "the BCT has not issued / does not regulate X" — a clear abstention
    "n'a pas émis", "n'a émis aucun", "n'a pas publié", "n'a pas adopté",
    "n'a pas pris de position", "ne réglemente pas", "n'a pas de réglementation",
    "has not issued", "has not published", "has not adopted", "does not regulate",
    "no specific regulation", "there is no specific regulation",
    # "context does not contain / sources don't address it" — honest declines that
    # otherwise read as an answer attempt; counted as declines, not answers.
    "does not contain specific information", "do not contain specific information",
    "does not contain any information", "does not contain information",
    "does not contain specific", "does not explicitly mention", "do not explicitly mention",
    "does not provide specific information", "not contain specific information",
    "aucune des sources", "ne traite explicitement", "ne traitent explicitement",
    "aucune disposition spécifique", "n'est mentionnée dans le contexte",
    "n'est mentionné dans le contexte", "ne sont pas mentionnées dans le contexte",
    "ne sont pas explicitement détaillé", "n'est pas explicitement détaillé",
    "ne sont pas explicitement mentionné", "n'est pas explicitement mentionné",
    "ne sont pas détaillées dans les sources", "ne sont pas précisées dans les sources",
    # Arabic
    "لا توجد معلومات", "لا أستطيع", "غير متوفر", "لا تتوفر", "لا أملك", "لا يمكنني",
    "لم يتم التطرق", "لا يتناول", "غير مذكور", "لم يرد", "لا يرد", "لا تتضمن",
    "لم تصدر", "لم يصدر", "لا تنظم",
)


def classify(answer: str, expected: str) -> str:
    # Strip markdown emphasis (**bold**, _italic_, `code`) so markers match even
    # when the model wraps the decisive phrase, e.g. "n'est **pas abordée**".
    ans = re.sub(r"[*_`]+", "", (answer or "").lower())
    refused = any(m in ans for m in _REFUSAL_MARKERS)
    is_error = (
        ans.startswith("error")
        or "erreur api" in ans
        or "request too large" in ans
        or "too large for model" in ans
        or "contacting ollama" in ans
        or "erreur de connexion" in ans
        or ("erreur" in ans[:40] and "limite" in ans[:80])
    )
    not_found = any(m in ans for m in _NOT_FOUND_MARKERS)

    if is_error:
        return "ERROR"
    if expected == "refuse":
        # An off-topic question is correctly handled whether the model refuses
        # outright OR declines that it has no such info / only covers BCT.
        return "PASS" if (refused or not_found) else "FAIL"
    if expected == "not_found":
        return "PASS" if (not_found or refused) else "FAIL"
    # expected == "answer"
    return "FAIL" if (refused or not_found) else "PASS"


# ── Aggregation ───────────────────────────────────────────────────────────────
def _accuracy_table(results: list, key: str) -> dict:
    buckets = defaultdict(lambda: {"passed": 0, "total": 0})
    for r in results:
        b = buckets[r[key]]
        b["total"] += 1
        if r["status"] == "PASS":
            b["passed"] += 1
    return {
        k: {"passed": v["passed"], "total": v["total"],
            "accuracy": round(v["passed"] / v["total"], 4) if v["total"] else 0.0}
        for k, v in sorted(buckets.items())
    }


def _print_table(title: str, table: dict) -> None:
    print(f"\n  {title}")
    print(f"  {'-' * 52}")
    for k, v in table.items():
        bar = "█" * int(v["accuracy"] * 20)
        print(f"  {k:<26} {v['passed']:>3}/{v['total']:<3} "
              f"{v['accuracy']*100:5.1f}%  {bar}")


def _build_payload(results: list, mode: str, size: int, selected: list) -> dict:
    passed = sum(1 for r in results if r["status"] == "PASS")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    total = len(results)
    return {
        "mode": mode,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "batch_size": size,
        "batches_run": [b + 1 for b in selected],
        "total": total, "passed": passed, "errors": errors,
        "accuracy": round(passed / total, 4) if total else 0.0,
        "by_expected_behavior": _accuracy_table(results, "expected_behavior"),
        "by_category": _accuracy_table(results, "category"),
        "by_language": _accuracy_table(results, "language"),
        "by_difficulty": _accuracy_table(results, "difficulty"),
        "results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="BCT RAG batch evaluation runner")
    ap.add_argument("--mode", choices=["rag", "graph"], default="graph",
                    help="Which engine to evaluate (default: graph)")
    ap.add_argument("--batch-size", type=int, default=20, help="Questions per batch (default: 20)")
    ap.add_argument("--batch", type=int, default=1, help="Which batch to run (1-indexed)")
    ap.add_argument("--all", action="store_true", help="Run all batches sequentially")
    ap.add_argument("--delay", type=float, default=2.0,
                    help="Seconds to wait between queries (rate-limit safety, default: 2)")
    ap.add_argument("--out", default="", help="Results JSON path (default: eval_results_<mode>.json)")
    ap.add_argument("--limit", type=int, default=0, help="Cap total questions (smoke test); 0 = no cap")
    ap.add_argument("--skip", type=int, default=0, help="Skip the first N selected questions (resume)")
    ap.add_argument("--append", action="store_true", help="Accumulate onto existing --out results")
    ap.add_argument("--ids", default="", help="Comma-separated question IDs to run (overrides batches)")
    ap.add_argument("--use-corrections", action="store_true",
                    help="Inject approved verified corrections (matches the live /api/chat system)")
    args = ap.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    size = max(1, args.batch_size)
    batches = [questions[i:i + size] for i in range(0, len(questions), size)]
    n_batches = len(batches)

    if args.all:
        selected = list(range(n_batches))
    else:
        if not (1 <= args.batch <= n_batches):
            ap.error(f"--batch must be between 1 and {n_batches}")
        selected = [args.batch - 1]

    to_run = [q for b in selected for q in batches[b]]
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        to_run = [q for q in questions if q["id"] in want]   # IDs override batch selection
    if args.skip:
        to_run = to_run[args.skip:]
    if args.limit:
        to_run = to_run[:args.limit]

    print("=" * 70)
    print(f" BCT RAG EVALUATION  ·  mode={args.mode.upper()}  ·  "
          f"{'ALL batches' if args.all else f'batch {args.batch}/{n_batches}'}  ·  "
          f"{len(to_run)} questions")
    print("=" * 70)

    # Load the engine (real pipeline). Imported here so --help stays instant.
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
    os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

    if args.mode == "graph":
        from graph_rag_engine import HybridGraphRAGEngine
        engine = HybridGraphRAGEngine()
    else:
        from rag_engine import RAGEngine
        engine = RAGEngine()

    corrections_mod = None
    if args.use_corrections:
        import corrections as corrections_mod
        print("  Verified corrections injection: ON", flush=True)

    out_path = Path(args.out) if args.out else BASE / f"eval_results_{args.mode}.json"

    results = []
    if args.append and out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8")).get("results", [])
            run_ids = {q["id"] for q in to_run}
            results = [r for r in prev if r["id"] not in run_ids]   # keep prior, drop any we re-run
            print(f"  Appending onto {len(results)} existing result(s).", flush=True)
        except Exception as e:
            print(f"  (could not load existing results: {e})", flush=True)

    for i, q in enumerate(to_run):
        t0 = time.time()
        try:
            notes = []
            if corrections_mod is not None and getattr(engine, "model", None) is not None:
                try:
                    notes = corrections_mod.find_relevant(engine.model, q["question"])
                except Exception:
                    notes = []
            res = engine.query(q["question"], notes=notes)
            answer = res.get("answer", "")
            sources = res.get("sources", [])
            status = classify(answer, q["expected_behavior"])
        except Exception as e:
            answer, sources, status = f"ERROR: {e}", [], "ERROR"
        elapsed = round(time.time() - t0, 2)

        results.append({
            "id": q["id"], "question": q["question"], "language": q["language"],
            "category": q["category"], "difficulty": q["difficulty"],
            "expected_behavior": q["expected_behavior"], "status": status,
            "response_time_s": elapsed, "answer_excerpt": answer[:200],
            "sources": [s.split("(")[0].strip() for s in sources[:3]],
        })

        mark = {"PASS": "PASS", "FAIL": "FAIL", "ERROR": "ERR "}[status]
        print(f"  [{mark}] {q['id']} · {q['language']} · {q['category']:<20} "
              f"· {q['expected_behavior']:<9} · {elapsed:>5.1f}s", flush=True)

        # Incremental save — progress survives an interruption.
        out_path.write_text(json.dumps(_build_payload(results, args.mode, size, selected),
                                       ensure_ascii=False, indent=2), encoding="utf-8")

        if args.delay and i < len(to_run) - 1:
            time.sleep(args.delay)

    # ── Summary ──
    payload = _build_payload(results, args.mode, size, selected)
    passed, total, errors, acc = payload["passed"], payload["total"], payload["errors"], payload["accuracy"]
    print("\n" + "=" * 70)
    print(f" RESULTS  ·  {passed}/{total} PASS  ·  accuracy {acc*100:.1f}%"
          + (f"  ·  {errors} API errors" if errors else ""))
    print("=" * 70)
    _print_table("By expected behavior", _accuracy_table(results, "expected_behavior"))
    _print_table("By category", _accuracy_table(results, "category"))
    _print_table("By language", _accuracy_table(results, "language"))
    _print_table("By difficulty", _accuracy_table(results, "difficulty"))
    print(f"\n  Full results saved to: {out_path.name}")


if __name__ == "__main__":
    main()
