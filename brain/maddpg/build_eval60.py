"""
brain/maddpg/build_eval60.py
----------------------------
Build a 60-question held-out evaluation set for the 4-model comparison
(simple_hybrid_rag, final_arch, MADDPG v4 CEB, MADDPG v4 no-CEB).

The set is drawn ONLY from questions the MADDPG controller never saw in
training — the union of:
    val.jsonl (28)  +  test.jsonl (29)  +  validated candidates never split (15)
= 72 held-out questions. 60 are sampled, stratified by category: the four small
categories are kept whole (they cannot spare questions), the four large ones are
proportionally down-sampled. Seed is fixed so the set is reproducible.

Output: brain/maddpg/results/benchmark_splits/eval60.jsonl

Usage (from brain/):
    python -m maddpg.build_eval60
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_SPLITS = Path(__file__).resolve().parent / "results" / "benchmark_splits"
_KEYS = ("question", "ground_truth", "source_file", "category", "difficulty")
_SEED = 42
_TARGET = 60

# Per-category quota. Small categories are kept whole; large ones are sampled.
# Sum = 60. (See module docstring for the proportional derivation.)
_QUOTA = {
    "definition_explanation":         14,   # of 17
    "cross_paper_comparison":         13,   # of 16
    "multi_chunk_synthesis":          11,   # of 14
    "paraphrase_hard_retrieval":      10,   # of 13
    "direct_fact_lookup":              4,   # of 4   (kept whole)
    "figure_table_diagram_grounded":   3,   # of 3   (kept whole)
    "intra_paper_comparison":          3,   # of 3   (kept whole)
    "distractor_edge_case":            2,   # of 2   (kept whole)
}


def _rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> int:
    train = _rows(_SPLITS / "train.jsonl")
    val = _rows(_SPLITS / "val.jsonl")
    test = _rows(_SPLITS / "test.jsonl")
    validated = _rows(_SPLITS / "validated_candidates.jsonl")

    split_qs = {r["question"] for r in (train + val + test)}
    extra = [r for r in validated if r["question"] not in split_qs]
    heldout = val + test + extra
    print(f"[eval60] held-out pool: {len(heldout)} "
          f"(val {len(val)} + test {len(test)} + extra {len(extra)})")

    # Strip auxiliary keys (the 15 extras carry _gen_type / _validation).
    heldout = [{k: r[k] for k in _KEYS} for r in heldout]

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in heldout:
        by_cat[r["category"]].append(r)

    rng = random.Random(_SEED)
    selected: list[dict] = []
    for cat, quota in _QUOTA.items():
        pool = by_cat.get(cat, [])
        if len(pool) < quota:
            raise SystemExit(f"[eval60] category '{cat}' has {len(pool)} < quota {quota}")
        picked = pool if quota == len(pool) else rng.sample(pool, quota)
        selected.extend(picked)
        print(f"  {cat:32s} {quota:>2d}/{len(pool)}")

    if len(selected) != _TARGET:
        raise SystemExit(f"[eval60] selected {len(selected)} != {_TARGET}")

    rng.shuffle(selected)
    out = _SPLITS / "eval60.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for r in selected:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[eval60] wrote {len(selected)} questions -> {out}")
    print(f"[eval60] difficulty: {dict(Counter(r['difficulty'] for r in selected))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
