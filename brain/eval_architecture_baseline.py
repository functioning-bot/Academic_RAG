"""
brain/eval_architecture_baseline.py
-----------------------------------
Batch-evaluate ANY LangGraph RAG architecture in brain/<arch>/ on a benchmark,
so it can be compared apples-to-apples against the MADDPG-controlled RAG.

Works for any architecture folder that exposes `graph.build_graph()` returning a
compiled LangGraph invoked with an `original_query` / `search_query` state and
producing a `generation` field — i.e. simple_hybrid_rag, final_arch,
crag_rewrite, crag_vericite, self_rag_grader.

Run ONE architecture per process (each folder has same-named modules — config,
node_generator, ... — so importing two in one process would collide).

Usage (from brain/):
  python eval_architecture_baseline.py --arch final_arch \
      --benchmark-path maddpg/results/benchmark_splits/test.jsonl \
      --out maddpg/results/eval_expanded/eval_final_arch.jsonl
"""
from __future__ import annotations

import argparse
import json
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

_BRAIN_ROOT = Path(__file__).resolve().parent

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
    return (tp / len(ret) if ret else 0.0), tp / len(exp)


def load_benchmark(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Benchmark not found: {path}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# Superset initial state covering every architecture's GraphState fields.
def _initial_state(question: str) -> Dict[str, Any]:
    return {
        "original_query": question,
        "search_query": question,
        "retrieved_docs": [], "candidate_docs": [],
        "weak_signal_docs": [], "graded_docs": [],
        "generation": "", "crag_retries": 0, "verify_retries": 0,
        "citations_pass": True, "auditor_feedback": "",
        "claim_verification": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser("Evaluate a brain/ RAG architecture on a benchmark")
    ap.add_argument("--arch", required=True,
                    help="Architecture folder under brain/ (e.g. final_arch, crag_rewrite)")
    ap.add_argument("--benchmark-path",
                    default="maddpg/results/benchmark_splits/test.jsonl")
    ap.add_argument("--out", default="")
    ap.add_argument("--n-questions", type=int, default=0)
    args = ap.parse_args()

    arch_dir = _BRAIN_ROOT / args.arch
    if not arch_dir.is_dir():
        print(f"[eval] architecture folder not found: {arch_dir}")
        return 1

    # brain/ first, then brain/<arch>/ so the arch's local modules resolve.
    for p in (_BRAIN_ROOT, arch_dir):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    out_path = Path(args.out) if args.out else (
        _BRAIN_ROOT / "maddpg" / "results" / "eval_expanded" / f"eval_{args.arch}.jsonl"
    )

    benchmark = load_benchmark(args.benchmark_path)
    if args.n_questions > 0:
        benchmark = benchmark[:args.n_questions]

    from graph import build_graph
    app_graph = build_graph()
    print(f"[eval] arch={args.arch}  on {len(benchmark)} questions")

    results: List[Dict[str, Any]] = []
    for idx, q in enumerate(benchmark):
        qid = q.get("question_id", f"Q{idx + 1:03d}")
        question = q.get("question", "")
        gold = q.get("ground_truth", "")
        exp_s = q.get("source_file", [])
        if isinstance(exp_s, str):
            exp_s = [exp_s]

        t0 = time.time()
        final_status, answer = "accepted", ""
        retrieved, graded = [], []
        try:
            state = app_graph.invoke(_initial_state(question))
            answer = (state.get("generation") or "").strip()
            retrieved = state.get("retrieved_docs", []) or []
            graded = state.get("graded_docs", []) or retrieved
            if not answer:
                final_status = "generation_failed"
        except Exception as exc:
            final_status = "error"
            print(f"  [err] {qid}: {exc}")
        latency = time.time() - t0

        src_p, src_r = src_precision_recall(retrieved, exp_s)
        tf1, rl = token_f1(answer, gold), rouge_l(answer, gold)
        results.append({
            "question_id": qid, "question": question, "ground_truth": gold,
            "category": q.get("category"), "difficulty": q.get("difficulty"),
            "policy": args.arch, "policy_mode": args.arch, "data_source": "live_llm",
            "final_status": final_status, "final_answer": answer,
            "token_f1": round(tf1, 4), "rouge_l": round(rl, 4),
            "verification_pass": int(final_status == "accepted"),
            "citation_support": 0.0, "num_unsupported_claims": 0,
            "selected_evidence_count": len(graded),
            "source_precision": round(src_p, 4), "source_recall": round(src_r, 4),
            "latency_seconds": round(latency, 3), "token_usage": 0,
        })
        print(f"  {qid}  {final_status:16s}  tf1={tf1:.3f}  lat={latency:.1f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(results) or 1
    mtf1 = sum(r["token_f1"] for r in results) / n
    mrl = sum(r["rouge_l"] for r in results) / n
    fails = sum(1 for r in results if r["final_status"] != "accepted")
    print(f"\n[eval] {args.arch}: wrote {len(results)} rows -> {out_path}")
    print(f"[eval] mean token_f1={mtf1:.4f}  mean rouge_l={mrl:.4f}  failures={fails}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
