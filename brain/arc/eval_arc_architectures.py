"""
brain/arc/eval_arc_architectures.py
-----------------------------------
Evaluate a brain/ RAG architecture (simple_hybrid_rag, final_arch, ...) on ARC
multiple-choice questions.

ARC is multiple-choice, so the metric is accuracy (did the pipeline pick the
correct letter), not Token F1. The brain/ architectures are free-form QA
pipelines, so this harness:

  1. Points retrieval at the `arc_corpus` Qdrant collection (via QDRANT_COLLECTION),
     so the pipeline retrieves ARC science sentences, not academic_papers.
  2. Feeds each architecture an MCQ-formatted query (question + choices +
     "answer with the letter").
  3. Parses the chosen letter out of the pipeline's free-form answer.
  4. Scores accuracy against answerKey.

Run ONE architecture per process (each arch folder has same-named modules).

Usage (from brain/):
    python arc/eval_arc_architectures.py --arch simple_hybrid_rag --n-questions 30
    python arc/eval_arc_architectures.py --arch final_arch       --n-questions 30
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_ARC_DIR = Path(__file__).resolve().parent
_DATA = _ARC_DIR / "data"
_BRAIN_ROOT = _ARC_DIR.parent

# Retrieval collection MUST be set before qdrant_config / the architecture
# import (qdrant_config reads QDRANT_COLLECTION at import time).
os.environ["QDRANT_COLLECTION"] = os.environ.get("QDRANT_COLLECTION", "arc_corpus")

try:
    from dotenv import load_dotenv
    for _ep in (_BRAIN_ROOT / ".env", _BRAIN_ROOT.parent / ".env"):
        if _ep.exists():
            load_dotenv(_ep)            # override=False: keeps QDRANT_COLLECTION
            break
except ImportError:
    pass

_SEED = 42


def _mcq_query(question: str, choices: List[dict]) -> str:
    block = "\n".join(f"{c['label']}) {c['text']}" for c in choices)
    labels = ", ".join(c["label"] for c in choices)
    return (f"{question}\n\nChoices:\n{block}\n\n"
            f"Select the single best answer. Respond with only the choice "
            f"letter ({labels}).")


def _parse_letter(reply: str, choices: List[dict]) -> str:
    """Extract the chosen choice label from a free-form reply, robustly."""
    valid = [c["label"] for c in choices]
    upper = {v.upper(): v for v in valid}
    for tok in re.findall(r"[A-Za-z0-9]+", reply or ""):
        if tok.upper() in upper:
            return upper[tok.upper()]
    low = (reply or "").lower()
    for c in choices:
        if c["text"] and c["text"].lower() in low:
            return c["label"]
    return ""   # unparseable -> wrong


def _initial_state(query: str) -> Dict[str, Any]:
    """Superset initial state covering every architecture's GraphState fields."""
    return {
        "original_query": query, "search_query": query,
        "retrieved_docs": [], "candidate_docs": [],
        "weak_signal_docs": [], "graded_docs": [],
        "generation": "", "crag_retries": 0, "verify_retries": 0,
        "citations_pass": True, "auditor_feedback": "",
        "claim_verification": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser("Evaluate a brain/ architecture on ARC MCQ")
    ap.add_argument("--arch", required=True,
                    help="Architecture folder under brain/ (simple_hybrid_rag, final_arch, ...)")
    ap.add_argument("--benchmark", default=str(_DATA / "arc_benchmark.jsonl"),
                    help="Path to the ARC benchmark .jsonl")
    ap.add_argument("--n-questions", type=int, default=30,
                    help="Sample N from the benchmark (0 = use all)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    arch_dir = _BRAIN_ROOT / args.arch
    if not arch_dir.is_dir():
        print(f"[arc-arch] architecture folder not found: {arch_dir}")
        return 1
    for p in (_BRAIN_ROOT, arch_dir):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    with open(args.benchmark, encoding="utf-8") as fh:
        questions = [json.loads(l) for l in fh if l.strip()]
    if args.n_questions > 0 and args.n_questions < len(questions):
        questions = random.Random(_SEED).sample(questions, args.n_questions)
    print(f"[arc-arch] arch={args.arch}  questions={len(questions)}  "
          f"collection={os.environ['QDRANT_COLLECTION']}")

    from graph import build_graph
    app_graph = build_graph()

    results: List[Dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        query = _mcq_query(q["question"], q["choices"])
        t0 = time.time()
        status, reply = "ok", ""
        try:
            state = app_graph.invoke(_initial_state(query))
            reply = (state.get("generation") or "").strip()
            if not reply:
                status = "generation_failed"
        except Exception as exc:
            status = "error"
            print(f"  [err] {q['id']}: {exc}")
        predicted = _parse_letter(reply, q["choices"])
        correct = int(predicted == q["answerKey"])
        results.append({
            "id": q["id"], "arc_split": q.get("arc_split", "ARC-Challenge"),
            "question": q["question"], "answerKey": q["answerKey"],
            "predicted": predicted, "correct": correct, "status": status,
            "reply": reply, "latency_seconds": round(time.time() - t0, 3),
        })
        mark = "OK " if correct else "XX "
        print(f"  [{i:3d}/{len(questions)}] {mark} pred={predicted or '?'} "
              f"gold={q['answerKey']}  ({time.time()-t0:.1f}s)")

    n = len(results) or 1
    acc = sum(r["correct"] for r in results) / n
    print(f"\n=== ARC accuracy ({args.arch}) ===")
    print(f"  accuracy: {acc:.4f}  ({sum(r['correct'] for r in results)}/{len(results)})")

    out = Path(args.out) if args.out else (_DATA / f"arc_eval_arch_{args.arch}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
