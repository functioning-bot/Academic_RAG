"""
brain/eval_simple_hybrid_baseline.py
-------------------------------------
Batch-evaluate the `simple_hybrid_rag` architecture on a benchmark JSONL so it
can be compared apples-to-apples against the MADDPG-controlled RAG.

simple_hybrid_rag is the minimal RAG baseline:
    hybrid RRF retrieval -> take top-8 -> single generation call
    (no grader, no rewriter, no verifier, no recovery loop)

It shares the same Qdrant index, the same retriever_shared backend, and the
same provider-aware llm_config as the MADDPG system, so the only thing that
differs is the control policy. That makes the comparison fair.

Output: a JSONL with one row per question, in the same shape the MADDPG eval
emits (question, ground_truth, final_answer, final_status, latency, etc.).

Usage (from brain/):
    python eval_simple_hybrid_baseline.py \
        --benchmark-path maddpg/results/benchmark_splits/test.jsonl \
        --out maddpg/results/eval_200ep_v3/eval_simple_hybrid_rag.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ── sys.path: brain/ and brain/simple_hybrid_rag/ ────────────────────────────
_BRAIN_ROOT = Path(__file__).resolve().parent
for _p in (_BRAIN_ROOT, _BRAIN_ROOT / "simple_hybrid_rag"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── dotenv (local only) ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    for _ep in (_BRAIN_ROOT / ".env", _BRAIN_ROOT.parent / ".env"):
        if _ep.exists():
            load_dotenv(dotenv_path=_ep)
            break
except ImportError:
    pass


# ── Metrics (identical to the MADDPG live runner) ────────────────────────────

def _tok(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", (text or "").lower())

def token_f1(pred: str, gold: str) -> float:
    p, g = set(_tok(pred)), set(_tok(gold))
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)

def rouge_l(pred: str, gold: str) -> float:
    p, g = _tok(pred), _tok(gold)
    m, n = len(p), len(g)
    if not m or not n:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if p[i-1] == g[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    prec, rec = lcs / m, lcs / n
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

def _src_pdfs(chunks: List[Dict]) -> set:
    out = set()
    for c in chunks:
        meta = c.get("metadata", {}) if isinstance(c.get("metadata"), dict) else {}
        sf = meta.get("source_file") or c.get("source_file") or ""
        if sf:
            out.add(sf.split("_p")[0].strip())
    return out

def src_precision_recall(retrieved: List[Dict], expected: List[str]):
    ret = _src_pdfs(retrieved)
    exp = {s.split("_p")[0].strip() for s in expected if s}
    if not exp:
        return 0.0, 0.0
    tp = len(ret & exp)
    prec = tp / len(ret) if ret else 0.0
    rec = tp / len(exp)
    return prec, rec


# ── Benchmark loading ─────────────────────────────────────────────────────────

def load_benchmark(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Benchmark not found: {path}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser("Evaluate simple_hybrid_rag baseline")
    ap.add_argument("--benchmark-path",
                    default="maddpg/results/benchmark_splits/test.jsonl")
    ap.add_argument("--out",
                    default="maddpg/results/eval_200ep_v3/eval_simple_hybrid_rag.jsonl")
    ap.add_argument("--n-questions", type=int, default=0,
                    help="Limit to first N (0 = all)")
    args = ap.parse_args()

    benchmark = load_benchmark(args.benchmark_path)
    if args.n_questions > 0:
        benchmark = benchmark[:args.n_questions]

    # Build the simple_hybrid_rag LangGraph (imports the provider-aware LLM).
    from graph import build_graph
    app_graph = build_graph()

    print(f"[eval] simple_hybrid_rag on {len(benchmark)} questions")
    results: List[Dict[str, Any]] = []

    for idx, q in enumerate(benchmark):
        qid = q.get("question_id", f"Q{idx + 1:03d}")
        question = q.get("question", "")
        gold = q.get("ground_truth", "")
        exp_s = q.get("source_file", [])
        if isinstance(exp_s, str):
            exp_s = [exp_s]

        t0 = time.time()
        final_status = "accepted"
        answer = ""
        retrieved: List[Dict] = []
        graded: List[Dict] = []
        try:
            state = app_graph.invoke({
                "original_query": question,
                "search_query":   question,
                "retrieved_docs": [], "candidate_docs": [],
                "weak_signal_docs": [], "graded_docs": [],
                "generation": "", "crag_retries": 0, "verify_retries": 0,
                "citations_pass": True, "auditor_feedback": "",
            })
            answer = (state.get("generation") or "").strip()
            retrieved = state.get("retrieved_docs", []) or []
            graded = state.get("graded_docs", []) or []
            if not answer:
                final_status = "generation_failed"
        except Exception as exc:
            final_status = "error"
            print(f"  [err] {qid}: {exc}")
        latency = time.time() - t0

        src_p, src_r = src_precision_recall(retrieved, exp_s)
        tf1 = token_f1(answer, gold)
        rl  = rouge_l(answer, gold)

        results.append({
            "question_id":            qid,
            "question":               question,
            "ground_truth":           gold,
            "category":               q.get("category"),
            "difficulty":             q.get("difficulty"),
            "policy":                 "simple_hybrid_rag",
            "policy_mode":            "simple_hybrid_rag",
            "data_source":            "live_llm",
            "final_status":           final_status,
            "final_answer":           answer,
            "token_f1":               round(tf1, 4),
            "rouge_l":                round(rl, 4),
            "verification_pass":      int(final_status == "accepted"),
            # simple_hybrid_rag has no verifier/citation step:
            "citation_support":       0.0,
            "num_unsupported_claims": 0,
            "selected_evidence_count": len(graded),
            "source_precision":       round(src_p, 4),
            "source_recall":          round(src_r, 4),
            "num_steps":              3,   # retrieve -> select -> generate
            "num_llm_calls":          1,   # single generation call
            "latency_seconds":        round(latency, 3),
            "token_usage":            0,   # baseline graph does not track tokens
        })
        print(f"  {qid}  status={final_status:16s}  tf1={tf1:.3f}  lat={latency:.1f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[eval] wrote {len(results)} rows -> {out_path}")

    n = len(results) or 1
    mean_tf1 = sum(r["token_f1"] for r in results) / n
    mean_rl  = sum(r["rouge_l"]  for r in results) / n
    fails = sum(1 for r in results if r["final_status"] != "accepted")
    print(f"[eval] mean token_f1={mean_tf1:.4f}  mean rouge_l={mean_rl:.4f}  failures={fails}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
