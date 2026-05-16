"""
brain/resplit_benchmark.py
--------------------------
Phase 5 of benchmark expansion: pool the validated new questions with the
existing benchmark and produce fresh stratified train / val / test splits.

Steps:
  1. Back up the current train/test/val splits to benchmark_splits_backup_v1/.
  2. Pool: existing 60 questions + the validated-PASS new questions.
  3. Normalize each row to the canonical 5-field schema
     (question, ground_truth, source_file[list], category, difficulty);
     strip generator/validation bookkeeping fields.
  4. Stratified split by (category, difficulty) bucket, ~60/20/20.
  5. Write new train.jsonl / val.jsonl / test.jsonl.

The document corpus is shared across all splits (all 12 papers live in one
Qdrant collection), so there is no source-leakage concern — only the questions
are partitioned.

Usage (from brain/):
  python resplit_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BRAIN_ROOT = Path(__file__).resolve().parent
_SPLITS = _BRAIN_ROOT / "maddpg" / "results" / "benchmark_splits"

CANON_FIELDS = ("question", "ground_truth", "source_file", "category", "difficulty")


def _load(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _canonical(row: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a row to the canonical 5-field schema; source_file -> list."""
    sf = row.get("source_file", [])
    if isinstance(sf, str):
        sf = [sf]
    return {
        "question":     (row.get("question") or "").strip(),
        "ground_truth": (row.get("ground_truth") or "").strip(),
        "source_file":  sf,
        "category":     row.get("category") or "uncategorized",
        "difficulty":   row.get("difficulty") or "medium",
    }


def _stratified_split(pool: List[Dict[str, Any]], ratios=(0.6, 0.2, 0.2), seed=42):
    """Split pool into (train, val, test), stratified by (category, difficulty)."""
    rng = random.Random(seed)
    buckets: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for q in pool:
        buckets[(q["category"], q["difficulty"])].append(q)

    train, val, test = [], [], []
    for _, items in sorted(buckets.items()):
        rng.shuffle(items)
        n = len(items)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        # guarantee tiny buckets still contribute to val/test where possible
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def main() -> int:
    ap = argparse.ArgumentParser("Re-split the expanded benchmark")
    ap.add_argument("--validated",
                    default=str(_SPLITS / "validated_candidates.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ── 1. Back up current splits ────────────────────────────────────────
    backup = _SPLITS.parent / "benchmark_splits_backup_v1"
    backup.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        src = _SPLITS / name
        if src.exists():
            shutil.copy2(src, backup / name)
    print(f"[resplit] backed up old splits -> {backup}")

    # ── 2. Pool existing + validated-pass new ────────────────────────────
    existing = (_load(_SPLITS / "train.jsonl")
                + _load(_SPLITS / "val.jsonl")
                + _load(_SPLITS / "test.jsonl"))
    print(f"[resplit] existing questions: {len(existing)}")

    validated = _load(Path(args.validated))
    new_pass = [r for r in validated
                if r.get("_validation", {}).get("status") == "pass"]
    print(f"[resplit] validated new questions kept (status=pass): {len(new_pass)}")

    pool = [_canonical(r) for r in existing] + [_canonical(r) for r in new_pass]

    # dedupe on question text
    seen, deduped = set(), []
    for q in pool:
        key = q["question"].lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(q)
    if len(deduped) != len(pool):
        print(f"[resplit] removed {len(pool) - len(deduped)} duplicate question(s)")
    pool = deduped
    print(f"[resplit] total pooled questions: {len(pool)}")

    # ── 3. Stratified split ──────────────────────────────────────────────
    train, val, test = _stratified_split(pool, seed=args.seed)

    for name, rows in (("train.jsonl", train), ("val.jsonl", val), ("test.jsonl", test)):
        with open(_SPLITS / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 4. Summary ───────────────────────────────────────────────────────
    print(f"\n[resplit] NEW splits: train={len(train)}  val={len(val)}  test={len(test)}")
    for name, rows in (("train", train), ("val", val), ("test", test)):
        cats = Counter(r["category"] for r in rows)
        diffs = Counter(r["difficulty"] for r in rows)
        print(f"  {name:6s} difficulties={dict(diffs)}")
    print()
    print("  category distribution (whole benchmark):")
    for cat, n in sorted(Counter(q["category"] for q in pool).items(), key=lambda x: -x[1]):
        print(f"    {cat:28s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
