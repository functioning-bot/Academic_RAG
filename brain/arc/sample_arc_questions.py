"""
brain/arc/sample_arc_questions.py
---------------------------------
Step 1 of the ARC evaluation track: sample a fixed benchmark of ARC questions.

ARC (AI2 Reasoning Challenge) is a public dataset of grade-school multiple-choice
science questions. This script downloads it via the `datasets` library and
samples a fixed subset (stratified across ARC-Challenge and ARC-Easy) to serve
as a standard public benchmark, separate from the project's free-form
academic-QA benchmark.

Output: brain/arc/data/arc_benchmark.jsonl — one question per line:
    {id, question, choices:[{label,text}...], answerKey, arc_split}

Usage (from brain/):
    python arc/sample_arc_questions.py --n-challenge 40 --n-easy 40 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_ARC_DIR = Path(__file__).resolve().parent
_DATA = _ARC_DIR / "data"


def _norm_choices(choices: dict) -> list[dict]:
    """ARC choices come as {'text':[...], 'label':[...]}; flatten to a list."""
    return [{"label": l, "text": t}
            for l, t in zip(choices["label"], choices["text"])]


def main() -> int:
    ap = argparse.ArgumentParser("Sample an ARC benchmark subset")
    ap.add_argument("--n-challenge", type=int, default=40)
    ap.add_argument("--n-easy", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test", choices=["test", "validation", "train"])
    ap.add_argument("--all-choices", action="store_true",
                    help="Keep questions with any number of choices "
                         "(default keeps only standard 4-choice questions)")
    args = ap.parse_args()

    from datasets import load_dataset
    rng = random.Random(args.seed)
    _DATA.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for cfg, n in (("ARC-Challenge", args.n_challenge), ("ARC-Easy", args.n_easy)):
        if n <= 0:
            print(f"[arc] {cfg}: skipped (requested 0)")
            continue
        ds = load_dataset("allenai/ai2_arc", cfg, split=args.split)
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        # ARC has a few questions with 3 or 5 choices; keep standard 4-choice ones.
        kept = 0
        for i in idxs:
            ex = ds[i]
            choices = _norm_choices(ex["choices"])
            if not args.all_choices and len(choices) != 4:
                continue
            if ex["answerKey"] not in {c["label"] for c in choices}:
                continue
            rows.append({
                "id":        ex["id"],
                "question":  ex["question"].strip(),
                "choices":   choices,
                "answerKey": ex["answerKey"],
                "arc_split": cfg,
            })
            kept += 1
            if kept >= n:
                break
        print(f"[arc] {cfg}: sampled {kept} questions")

    rng.shuffle(rows)
    out = _DATA / "arc_benchmark.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[arc] wrote {len(rows)} questions -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
