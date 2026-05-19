"""
brain/maddpg/compute_extended_metrics.py
-----------------------------------------
Offline post-processor: reads an existing evaluate_maddpg output JSONL
(which already has final_answer + ground_truth) and the original benchmark
JSONL (which has source_file ground truth), then computes and prints:

    token_f1       – word-overlap F1 between final_answer and ground_truth
    rouge_l_f1     – LCS-based Rouge-L F1
    source_precision – fraction of retrieved source PDFs that were expected
    source_recall    – fraction of expected source PDFs that were retrieved

Source precision/recall are computed from the evidence chunk metadata stored
in the trace (discrete_action="keep_all" / "strict_filter" etc. steps show
what grader kept, but source is only in the live state).  Because the JSONL
does NOT persist chunk metadata, source P/R is approximated from the
benchmark's expected source_file list against the generator's citation
references embedded in final_answer when available.

NOTE: For full source P/R you need a fresh run with --save-sources.
      This script emits what is computable from saved data only.

Usage (from brain/):
    python -m maddpg.compute_extended_metrics \
        --eval-jsonl  maddpg/results/eval60_ceb_run1/eval_maddpg_ceb.jsonl \
        --benchmark   maddpg/results/benchmark_splits/eval60.jsonl \
        --out         maddpg/results/eval60_ceb_run1/extended_metrics.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Metric implementations (identical to eval_architecture_baseline.py) ──────

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


def rouge_l_f1(pred: str, gold: str) -> float:
    p, g = _tok(pred), _tok(gold)
    m, n = len(p), len(g)
    if not m or not n:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if p[i - 1] == g[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec, rec = lcs / m, lcs / n
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _stem(fname: str) -> str:
    """Strip page suffix and lower-case for fuzzy matching."""
    return fname.split("_p")[0].strip().lower()


def _cited_pdfs_from_answer(answer: str) -> set:
    """
    Heuristic: extract bare PDF-like filenames mentioned in the answer text.
    E.g. 'According to BERT.pdf ...' → {'bert.pdf'}
    Only useful when the generator cites sources inline.
    """
    hits = re.findall(r"[\w\-. ]+\.pdf", answer or "", re.IGNORECASE)
    return {h.strip().lower() for h in hits}


def src_precision_recall(
    retrieved_stems: set,
    expected: List[str],
) -> Tuple[float, float]:
    exp_stems = {_stem(s) for s in expected if s}
    if not exp_stems:
        return 0.0, 0.0
    tp = len(retrieved_stems & exp_stems)
    prec = tp / len(retrieved_stems) if retrieved_stems else 0.0
    rec  = tp / len(exp_stems)
    return prec, rec


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_benchmark_index(path: Path) -> Dict[str, Dict[str, Any]]:
    """Return a dict keyed by question_id (synthesised as Q001..QN if absent)
    AND by question text as a fallback."""
    rows = _load_jsonl(path)
    idx: Dict[str, Dict] = {}
    for i, r in enumerate(rows):
        # Assign stable Q001-style IDs matching how the MADDPG eval assigns them
        qid = r.get("question_id") or f"Q{i + 1:03d}"
        r.setdefault("_qid", qid)  # store back for debugging
        idx[qid] = r
        # Also index by question text for fall-through matching
        q = r.get("question", "")
        if q:
            idx[q] = r
    return idx


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser("Compute extended metrics for an existing eval JSONL")
    ap.add_argument("--eval-jsonl",  required=True,
                    help="Path to evaluate_maddpg output JSONL")
    ap.add_argument("--benchmark",   required=True,
                    help="Path to eval60.jsonl (has source_file ground truth)")
    ap.add_argument("--out",         default="",
                    help="Where to write extended per-question JSON (optional)")
    args = ap.parse_args()

    eval_path  = Path(args.eval_jsonl)
    bench_path = Path(args.benchmark)

    if not eval_path.exists():
        print(f"[ERR] eval JSONL not found: {eval_path}"); return 1
    if not bench_path.exists():
        print(f"[ERR] benchmark not found: {bench_path}"); return 1

    eval_rows  = _load_jsonl(eval_path)
    bench_idx  = _load_benchmark_index(bench_path)

    extended: List[Dict] = []
    for r in eval_rows:
        qid      = r.get("question_id", "")
        answer   = r.get("final_answer", "") or ""
        gold     = r.get("ground_truth", "") or ""
        question = r.get("question", "")

        # token_f1 and rouge_l_f1
        tf1 = token_f1(answer, gold)
        rl  = rouge_l_f1(answer, gold)

        # source P/R: look up benchmark by question_id, fall back to question text
        brow = bench_idx.get(qid) or bench_idx.get(question) or {}
        expected_sources: List[str] = brow.get("source_file", [])
        if isinstance(expected_sources, str):
            expected_sources = [expected_sources]

        # Best-effort retrieved stems: use cited PDF names from the answer
        cited = _cited_pdfs_from_answer(answer)
        cited_stems = {_stem(c) for c in cited}
        src_p, src_r = src_precision_recall(cited_stems, expected_sources)

        rec = {
            "question_id":     qid,
            "question":        question,
            "final_status":    r.get("final_status", ""),
            "verification_pass": r.get("verification_pass", 0),
            "token_f1":        round(tf1, 4),
            "rouge_l_f1":      round(rl,  4),
            "source_precision": round(src_p, 4),
            "source_recall":    round(src_r, 4),
            "citation_support": r.get("citation_support", 0.0),
            "expected_sources": expected_sources,
            "cited_in_answer":  sorted(cited),
            "_note": ("source_precision/recall based on PDF names cited in answer text; "
                      "re-run with --save-sources for exact chunk-level source tracking."),
        }
        extended.append(rec)

        print(f"  {qid}  tf1={tf1:.3f}  rl={rl:.3f}  "
              f"src_p={src_p:.3f}  src_r={src_r:.3f}  "
              f"[{r.get('final_status','?')}]")

    # Aggregate
    n = len(extended) or 1
    agg = {
        "n_questions":        n,
        "mean_token_f1":      round(sum(e["token_f1"]        for e in extended) / n, 4),
        "mean_rouge_l_f1":    round(sum(e["rouge_l_f1"]      for e in extended) / n, 4),
        "mean_source_precision": round(sum(e["source_precision"] for e in extended) / n, 4),
        "mean_source_recall":    round(sum(e["source_recall"]    for e in extended) / n, 4),
        "mean_citation_support": round(sum(e["citation_support"]  for e in extended) / n, 4),
    }

    print("\n" + "=" * 60)
    print(f"  mean token_f1        : {agg['mean_token_f1']}")
    print(f"  mean rouge_l_f1      : {agg['mean_rouge_l_f1']}")
    print(f"  mean source_precision: {agg['mean_source_precision']}"
          " (heuristic – inline PDF citations only)")
    print(f"  mean source_recall   : {agg['mean_source_recall']}"
          " (heuristic – inline PDF citations only)")
    print(f"  mean citation_support: {agg['mean_citation_support']}")
    print("=" * 60)

    # Write output
    out = {
        "source_eval_jsonl": str(eval_path),
        "source_benchmark":  str(bench_path),
        "aggregate":         agg,
        "per_question":      extended,
    }
    out_path_str = args.out or str(eval_path.parent / "extended_metrics.json")
    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nExtended metrics written -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
