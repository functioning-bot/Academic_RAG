"""
brain/arc/eval_arc.py
---------------------
Step 4 of the ARC evaluation track: evaluate the system on ARC multiple-choice
science questions.

This is a SELF-CONTAINED harness — it does not touch the free-form academic-QA
pipeline or its generators. ARC is multiple-choice, so the metric is accuracy
(did the model pick the correct letter), not Token F1.

Two modes:
  retrieval    — hybrid retrieval from the `arc_corpus` Qdrant collection, then
                 a multiple-choice prompt that asks the LLM to pick a letter.
  closed-book  — no retrieval; the LLM answers from parametric knowledge only.
                 A reference point: it isolates the contribution of retrieval.

Usage (from brain/):
    python arc/eval_arc.py --mode retrieval
    python arc/eval_arc.py --mode closed-book
    python arc/eval_arc.py --mode both
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_ARC_DIR = Path(__file__).resolve().parent
_DATA = _ARC_DIR / "data"
_BRAIN_ROOT = _ARC_DIR.parent
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

try:
    from dotenv import load_dotenv
    for _ep in (_BRAIN_ROOT / ".env", _BRAIN_ROOT.parent / ".env"):
        if _ep.exists():
            load_dotenv(_ep)
            break
except ImportError:
    pass

import os
from langchain_core.messages import HumanMessage
from llm_config import build_llm, current_model

ARC_COLLECTION = "arc_corpus"
RETRIEVE_TOP_K = 8


# ── Retrieval from the arc_corpus collection ─────────────────────────────────

_rs = None  # lazily-loaded retriever_shared module (provides models + client)


def _retrieve(query: str, top_k: int = RETRIEVE_TOP_K) -> list[dict]:
    """Hybrid dense+sparse RRF retrieval against the `arc_corpus` collection."""
    global _rs
    if _rs is None:
        import retriever_shared as rs   # loads BGE-M3 + BM25 + Qdrant client
        _rs = rs
    from qdrant_client import models

    dense_vec = _rs._get_dense_query_embedding(query)      # noqa: SLF001
    sparse_vec = _rs._get_sparse_query_embedding(query)    # noqa: SLF001
    res = _rs.client.query_points(
        collection_name=ARC_COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_vec, using="dense", limit=20),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_vec.indices.tolist(),
                                          values=sparse_vec.values.tolist()),
                using="sparse", limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k, with_payload=True,
    )
    return [{"text": p.payload.get("text", ""), "score": p.score} for p in res.points]


# ── Multiple-choice prompt (the ARC-specific generator prompt) ───────────────

def _mcq_prompt(question: str, choices: list[dict], context: str | None) -> str:
    choice_block = "\n".join(f"{c['label']}) {c['text']}" for c in choices)
    labels = ", ".join(c["label"] for c in choices)
    parts = ["You are answering a multiple-choice science exam question."]
    if context:
        parts.append(
            "Reference material retrieved from a science corpus is provided "
            "below. Use it when it is relevant; rely on your own knowledge "
            "when it is not.\n\nReference material:\n---\n" + context + "\n---")
    parts.append(f"Question: {question}\n\nChoices:\n{choice_block}")
    parts.append(
        f"Select the single best answer. Respond with ONLY the choice letter "
        f"({labels}). Do not explain, do not add any other text.")
    return "\n\n".join(parts)


def _parse_letter(reply: str, choices: list[dict]) -> str:
    """Extract the chosen choice label from the LLM reply, robustly."""
    valid = [c["label"] for c in choices]
    # 1. First standalone valid label token.
    for tok in re.findall(r"[A-Za-z0-9]+", reply or ""):
        if tok.upper() in {v.upper() for v in valid}:
            for v in valid:
                if v.upper() == tok.upper():
                    return v
    # 2. Fallback: match the reply text to a choice's text.
    low = (reply or "").lower()
    for c in choices:
        if c["text"] and c["text"].lower() in low:
            return c["label"]
    return ""   # unparseable -> counts as wrong


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate(questions: list[dict], mode: str) -> list[dict]:
    use_retrieval = (mode == "retrieval")
    llm = build_llm(temperature=0.0, max_tokens=8)
    results = []
    for i, q in enumerate(questions, 1):
        context, n_ctx = None, 0
        if use_retrieval:
            rq = q["question"] + " " + " ".join(c["text"] for c in q["choices"])
            try:
                chunks = _retrieve(rq)
            except Exception as exc:
                chunks = []
                print(f"  [retrieval-err] {q['id']}: {exc}")
            n_ctx = len(chunks)
            context = "\n".join(f"- {c['text']}" for c in chunks) or None

        prompt = _mcq_prompt(q["question"], q["choices"], context)
        t0 = time.time()
        try:
            reply = llm.invoke([HumanMessage(content=prompt)])
            reply_text = reply.content if hasattr(reply, "content") else str(reply)
        except Exception as exc:
            reply_text = ""
            print(f"  [llm-err] {q['id']}: {exc}")
        predicted = _parse_letter(reply_text, q["choices"])
        correct = int(predicted == q["answerKey"])
        results.append({
            "id": q["id"], "arc_split": q["arc_split"],
            "question": q["question"], "answerKey": q["answerKey"],
            "predicted": predicted, "correct": correct,
            "retrieved_count": n_ctx, "latency_seconds": round(time.time() - t0, 3),
        })
        mark = "OK " if correct else "XX "
        print(f"  [{i:3d}/{len(questions)}] {mark} {q['arc_split']:14s} "
              f"pred={predicted or '?'} gold={q['answerKey']}")
    return results


def _report(results: list[dict], mode: str) -> dict:
    n = len(results) or 1
    overall = sum(r["correct"] for r in results) / n
    by_split = {}
    for split in sorted({r["arc_split"] for r in results}):
        rs = [r for r in results if r["arc_split"] == split]
        by_split[split] = sum(r["correct"] for r in rs) / (len(rs) or 1)
    print(f"\n=== ARC accuracy ({mode}) ===")
    print(f"  overall: {overall:.4f}  ({sum(r['correct'] for r in results)}/{len(results)})")
    for split, acc in by_split.items():
        print(f"  {split:16s}: {acc:.4f}")
    return {"mode": mode, "n": len(results), "accuracy": overall,
            "accuracy_by_split": by_split}


def main() -> int:
    ap = argparse.ArgumentParser("Evaluate the system on ARC multiple-choice questions")
    ap.add_argument("--mode", default="both",
                    choices=["retrieval", "closed-book", "both"])
    ap.add_argument("--benchmark", default=str(_DATA / "arc_benchmark.jsonl"),
                    help="Path to the ARC benchmark .jsonl")
    ap.add_argument("--suffix", default="",
                    help="Suffix for output files (e.g. _60q) to avoid clobbering")
    args = ap.parse_args()

    bench = Path(args.benchmark)
    with open(bench, encoding="utf-8") as fh:   # line iteration, not splitlines()
        questions = [json.loads(l) for l in fh if l.strip()]
    print(f"[arc-eval] {len(questions)} questions  | model: {current_model()}")

    modes = ["closed-book", "retrieval"] if args.mode == "both" else [args.mode]
    summary = {}
    for mode in modes:
        print(f"\n{'-'*60}\n  Mode: {mode}\n{'-'*60}")
        results = _evaluate(questions, mode)
        summary[mode] = _report(results, mode)
        out = _DATA / f"arc_eval_{mode.replace('-', '_')}{args.suffix}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  -> {out}")

    with open(_DATA / f"arc_eval_summary{args.suffix}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
