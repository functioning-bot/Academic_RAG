"""
brain/arc/build_arc_corpus.py
-----------------------------
Step 2 of the ARC evaluation track: build a focused retrieval corpus.

The full ARC Corpus is ~14M science sentences (1.48 GB) — far too large to
embed on CPU. ARC questions have no per-question source documents, so this
script builds a *small sampled subset*: for every sampled benchmark question it
scans the ARC Corpus and keeps the sentences most relevant to that question
(by keyword overlap with the question and all four answer choices). The union
of those sentences becomes the retrieval corpus.

This makes retrieval a fair test — relevant evidence exists in the corpus,
mixed with distractor sentences from the wrong answer choices — while keeping
the corpus small enough to embed quickly.

It streams ARC_Corpus.txt directly out of the downloaded zip; the 1.48 GB file
is never written to disk.

Output: brain/arc/data/arc_corpus_subset.jsonl — one sentence per line:
    {text, matched_question_ids:[...]}

Usage (from brain/):
    python arc/build_arc_corpus.py --per-question 80
"""
from __future__ import annotations

import argparse
import heapq
import io
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_DATA = Path(__file__).resolve().parent / "data"
_ZIP = _DATA / "ARC-V1-Feb2018.zip"
_CORPUS_MEMBER = "ARC-V1-Feb2018-2/ARC_Corpus.txt"

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "be", "been", "being", "for", "on", "at", "by", "with", "as", "that", "this",
    "these", "those", "it", "its", "from", "which", "what", "when", "where",
    "how", "why", "who", "will", "would", "can", "could", "should", "may",
    "might", "do", "does", "did", "has", "have", "had", "not", "but", "if",
    "than", "then", "into", "out", "up", "down", "more", "most", "some", "such",
    "best", "following", "question", "answer", "above", "below", "between",
}

_WORD = re.compile(r"\b[a-z]{3,}\b")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOPWORDS}


def main() -> int:
    ap = argparse.ArgumentParser("Build a focused ARC retrieval corpus")
    ap.add_argument("--per-question", type=int, default=80,
                    help="Sentences to keep per question")
    ap.add_argument("--min-overlap", type=int, default=2,
                    help="Minimum keyword overlap for a sentence to count")
    ap.add_argument("--min-chars", type=int, default=30)
    ap.add_argument("--max-chars", type=int, default=400)
    args = ap.parse_args()

    if not _ZIP.exists():
        print(f"[arc] ARC Corpus zip not found: {_ZIP}")
        return 1

    with open(_DATA / "arc_benchmark.jsonl", encoding="utf-8") as fh:
        questions = [json.loads(l) for l in fh if l.strip()]
    print(f"[arc] {len(questions)} benchmark questions loaded")

    # Per-question keyword sets = content words of question + all choice texts.
    q_keywords: list[set[str]] = []
    for q in questions:
        kw = _content_words(q["question"])
        for c in q["choices"]:
            kw |= _content_words(c["text"])
        q_keywords.append(kw)

    # Inverted index: content word -> set of question indices that contain it.
    word2q: dict[str, set[int]] = defaultdict(set)
    for qi, kw in enumerate(q_keywords):
        for w in kw:
            word2q[w].add(qi)
    print(f"[arc] inverted index: {len(word2q)} distinct keywords")

    # Per-question min-heaps of (score, counter, sentence), size <= per_question.
    heaps: list[list] = [[] for _ in questions]
    counter = 0

    print("[arc] streaming ARC Corpus (this is one pass over ~14M lines)...")
    with zipfile.ZipFile(_ZIP) as z:
        raw = z.open(_CORPUS_MEMBER)
        stream = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
        for ln, line in enumerate(stream, 1):
            if ln % 2_000_000 == 0:
                print(f"  ...{ln:,} lines")
            sent = line.strip()
            if not (args.min_chars <= len(sent) <= args.max_chars):
                continue
            toks = _content_words(sent)
            if not toks:
                continue
            # Candidate questions: any question sharing >=1 keyword.
            cand: set[int] = set()
            for t in toks:
                qs = word2q.get(t)
                if qs:
                    cand |= qs
            if not cand:
                continue
            for qi in cand:
                score = len(toks & q_keywords[qi])
                if score < args.min_overlap:
                    continue
                h = heaps[qi]
                counter += 1
                if len(h) < args.per_question:
                    heapq.heappush(h, (score, counter, sent))
                elif score > h[0][0]:
                    heapq.heapreplace(h, (score, counter, sent))

    # Union the kept sentences, deduplicate, record which questions matched.
    sent_to_qids: dict[str, set[str]] = defaultdict(set)
    for qi, h in enumerate(heaps):
        qid = questions[qi]["id"]
        for _score, _c, sent in h:
            sent_to_qids[sent].add(qid)

    out = _DATA / "arc_corpus_subset.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for sent, qids in sent_to_qids.items():
            f.write(json.dumps({"text": sent,
                                "matched_question_ids": sorted(qids)},
                               ensure_ascii=False) + "\n")

    empties = sum(1 for h in heaps if not h)
    print(f"[arc] kept {len(sent_to_qids)} unique sentences -> {out}")
    print(f"[arc] questions with zero matched sentences: {empties}/{len(questions)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
