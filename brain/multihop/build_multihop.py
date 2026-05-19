"""
brain/multihop/build_multihop.py
--------------------------------
Build a focused evaluation track for a multi-hop QA dataset (HotpotQA or
2WikiMultihopQA).

Both datasets ship, per question, a set of context paragraphs (the gold
supporting paragraphs mixed with distractors). That makes the retrieval corpus
trivial to build — no giant corpus scan (unlike ARC): the corpus is simply the
union of every sampled question's context paragraphs.

Outputs (under multihop/data/):
    <ds>_benchmark.jsonl  — {question, ground_truth, source_file, category, difficulty}
    <ds>_corpus.jsonl     — {text, source_file} one paragraph per line

Usage (from brain/):
    python multihop/build_multihop.py --dataset hotpotqa --n-questions 30
    python multihop/build_multihop.py --dataset 2wiki    --n-questions 30
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

_DATA = Path(__file__).resolve().parent / "data"
_SEED = 42

_SOURCES = {
    "hotpotqa": ("hotpot_qa", "distractor"),
    "2wiki": ("voidful/2WikiMultihopQA", None),
}


def _paragraphs(dataset: str, q: dict) -> list[tuple[str, list[str]]]:
    """Return [(title, sentences), ...] for a question's context."""
    ctx = q["context"]
    if dataset == "hotpotqa":               # dict: {'title':[...], 'sentences':[[...]]}
        return list(zip(ctx["title"], ctx["sentences"]))
    return [(item[0], item[1]) for item in ctx]   # 2wiki: [[title, [sents]], ...]


def _gold_titles(dataset: str, q: dict) -> list[str]:
    sf = q["supporting_facts"]
    if dataset == "hotpotqa":               # dict: {'title':[...], 'sent_id':[...]}
        return sorted(set(sf["title"]))
    return sorted({x[0] for x in sf})       # 2wiki: [[title, sent_id], ...]


def main() -> int:
    ap = argparse.ArgumentParser("Build a multi-hop QA evaluation track")
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "2wiki"])
    ap.add_argument("--n-questions", type=int, default=30)
    args = ap.parse_args()

    from datasets import load_dataset
    hid, cfg = _SOURCES[args.dataset]
    ds = load_dataset(hid, cfg, split="validation") if cfg else \
        load_dataset(hid, split="validation")
    print(f"[multihop] {args.dataset}: {len(ds)} validation questions loaded")

    idx = random.Random(_SEED).sample(range(len(ds)), args.n_questions)
    sampled = [ds[i] for i in idx]

    bench, corpus = [], {}
    for q in sampled:
        paras = _paragraphs(args.dataset, q)
        gold = _gold_titles(args.dataset, q)
        bench.append({
            "question": q["question"],
            "ground_truth": str(q["answer"]),
            "source_file": gold,
            "category": q.get("type", "multihop"),
            "difficulty": q.get("level", "") or "medium",
        })
        for title, sents in paras:
            text = " ".join(s.strip() for s in sents if s.strip())
            if text and title not in corpus:
                corpus[title] = {"text": text, "source_file": title}

    _DATA.mkdir(parents=True, exist_ok=True)
    bpath = _DATA / f"{args.dataset}_benchmark.jsonl"
    cpath = _DATA / f"{args.dataset}_corpus.jsonl"
    with open(bpath, "w", encoding="utf-8") as f:
        for r in bench:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(cpath, "w", encoding="utf-8") as f:
        for r in corpus.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[multihop] {len(bench)} questions  -> {bpath}")
    print(f"[multihop] {len(corpus)} unique paragraphs -> {cpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
