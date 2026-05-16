"""
brain/generate_benchmark_questions.py
-------------------------------------
Phase 2 of benchmark expansion: LLM-assisted generation of hard RAG questions.

Design goal: generate questions a *fixed* RAG pipeline (retrieve top-k, generate
once) struggles with — multi-chunk synthesis and cross-paper comparison — with
gold answers long enough (target 50-110 words) that token-overlap metrics can
actually distinguish an evidence-rich answer from a 1-chunk answer.

Pipeline:
  1. Scroll all chunks from Qdrant, group by source_file (keep point IDs).
  2. For each requested question TYPE, sample chunks and prompt the LLM.
  3. The LLM returns {question, ground_truth} JSON.
  4. Write candidates to a JSONL for validation (Phase 3) + human review (Phase 4).

Question types:
  multi_chunk_synthesis   — 3 chunks, same paper, different sections
  cross_paper_comparison  — chunk from paper A + its dense-nearest chunk in a
                            DIFFERENT paper (guarantees a real shared theme)
  paraphrase_hard         — 1 chunk; question deliberately avoids the chunk's wording
  deep_single_source      — 1 rich chunk; multi-fact explanatory answer

Quality guards:
  - cross-paper pairing is theme-aware (dense similarity), not random
  - candidates containing the unicode replacement char are dropped
  - questions must be about research content, not authorship/metadata

Usage (from brain/):
  python generate_benchmark_questions.py --per-type 3            # small review batch
  python generate_benchmark_questions.py --per-type 18 --out ... # full run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Force UTF-8 stdout so legitimate scientific characters (CO₂, 32×32, accented
# author names) do not crash printing under the Windows cp1252 console codec.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BRAIN_ROOT = Path(__file__).resolve().parent
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

from qdrant_client import QdrantClient, models
from langchain_core.messages import HumanMessage

from llm_config import build_llm, current_model


COLLECTION = os.getenv("QDRANT_COLLECTION", "academic_papers")
MIN_CHUNK_CHARS = 250
QUESTION_TYPES = [
    "multi_chunk_synthesis",
    "cross_paper_comparison",
    "paraphrase_hard",
    "deep_single_source",
]

_CLIENT: Optional[QdrantClient] = None   # set in main()


# ── Quality guards ────────────────────────────────────────────────────────────

def _is_clean(*texts: str) -> bool:
    """Reject candidates with the unicode replacement char (PDF-parse mojibake)."""
    for t in texts:
        if not t:
            return False
        if "�" in t or "�" in t:
            return False
    return True


# ── Corpus loading ────────────────────────────────────────────────────────────

def load_corpus(client: QdrantClient) -> Dict[str, List[Dict[str, Any]]]:
    """Return {source_file: [chunk, ...]} for substantial text chunks.

    Each chunk dict is the Qdrant payload plus an injected '_id' (point id),
    needed to look the dense vector back up for theme-aware cross-paper pairing.
    """
    by_paper: Dict[str, List[Dict[str, Any]]] = {}
    offset = None
    while True:
        recs, offset = client.scroll(
            COLLECTION, limit=256, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for r in recs:
            p = dict(r.payload or {})
            text = (p.get("text") or "").strip()
            if len(text) < MIN_CHUNK_CHARS:
                continue
            if p.get("content_type") not in (None, "text", "table"):
                continue
            p["_id"] = r.id
            by_paper.setdefault(p.get("source_file", "?"), []).append(p)
        if offset is None:
            break
    return by_paper


def _nearest_chunk_in_other_paper(
    client: QdrantClient, chunk: Dict[str, Any], same_paper: str,
) -> Optional[Dict[str, Any]]:
    """Find the dense-nearest chunk that belongs to a DIFFERENT paper."""
    got = client.retrieve(COLLECTION, ids=[chunk["_id"]],
                           with_vectors=True, with_payload=False)
    if not got:
        return None
    vec = got[0].vector
    dense_vec = vec.get("dense") if isinstance(vec, dict) else vec
    if dense_vec is None:
        return None
    hits = client.query_points(
        COLLECTION, query=dense_vec, using="dense", limit=8,
        with_payload=True,
        query_filter=models.Filter(must_not=[
            models.FieldCondition(key="source_file",
                                  match=models.MatchValue(value=same_paper))
        ]),
    ).points
    for h in hits:
        pay = h.payload or {}
        if len((pay.get("text") or "").strip()) >= MIN_CHUNK_CHARS:
            return pay
    return None


# ── LLM JSON helper ───────────────────────────────────────────────────────────

def _ask_json(llm, prompt: str, retries: int = 2) -> Optional[Dict[str, Any]]:
    for _ in range(retries):
        resp = llm.invoke([HumanMessage(content=prompt)])
        txt = resp.content if hasattr(resp, "content") else str(resp)
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return None


def _chunk_block(chunk: Dict[str, Any], label: str) -> str:
    return (f"{label} (paper: {chunk.get('source_file','?')}, "
            f"section: {chunk.get('section_header','?')}, "
            f"page: {chunk.get('page_number','?')}):\n"
            f"{chunk.get('text', '').strip()}")


# ── Shared prompt rules ───────────────────────────────────────────────────────

_CONTENT_RULE = (
    "- The question MUST be about the paper's research content (methods, "
    "results, concepts, mechanisms, findings). It must NOT be about authorship, "
    "acknowledgements, funding, author contributions, references, or any "
    "document metadata.\n"
)

_RULES = (
    "Rules for the gold answer:\n"
    "- It MUST be fully supported by the passages provided — invent nothing.\n"
    "- It MUST synthesize/combine facts; it cannot be lifted from a single sentence.\n"
    "- Length: 50-110 words. Specific, factual, no hedging, no citations.\n"
    "Rules for the question:\n"
    "- A reader must consult ALL provided passages to answer it fully.\n"
    "- Self-contained: do not say 'the passage' or 'the figure'.\n"
    f"{_CONTENT_RULE}"
    'Return ONLY JSON: {"question": "...", "ground_truth": "..."}'
)


# ── Per-type generators ───────────────────────────────────────────────────────

def gen_multi_chunk(llm, corpus) -> Optional[Dict[str, Any]]:
    papers = [p for p, cs in corpus.items() if len(cs) >= 3]
    if not papers:
        return None
    paper = random.choice(papers)
    chunks = random.sample(corpus[paper], 3)
    blocks = "\n\n".join(_chunk_block(c, f"PASSAGE {i+1}") for i, c in enumerate(chunks))
    prompt = (
        "You are building a hard benchmark for a retrieval-augmented QA system.\n\n"
        "Below are THREE passages from the SAME academic paper.\n\n"
        f"{blocks}\n\n"
        "Write ONE question whose complete, correct answer requires synthesizing "
        f"information from all three passages.\n\n{_RULES}"
    )
    out = _ask_json(llm, prompt)
    if not out or not _is_clean(out.get("question", ""), out.get("ground_truth", "")):
        return None
    return {
        "question":     out["question"].strip(),
        "ground_truth": out["ground_truth"].strip(),
        "source_file":  [paper],
        "category":     "multi_chunk_synthesis",
        "difficulty":   "hard",
        "_gen_type":    "multi_chunk_synthesis",
    }


def gen_cross_paper(llm, corpus) -> Optional[Dict[str, Any]]:
    papers = [p for p, cs in corpus.items() if cs]
    if len(papers) < 2:
        return None
    p1 = random.choice(papers)
    c1 = random.choice(corpus[p1])
    # Theme-aware pairing: nearest chunk in a different paper, not random.
    c2 = _nearest_chunk_in_other_paper(_CLIENT, c1, p1)
    if not c2:
        return None
    p2 = c2.get("source_file", "?")
    blocks = _chunk_block(c1, "PASSAGE A") + "\n\n" + _chunk_block(c2, "PASSAGE B")
    prompt = (
        "You are building a hard benchmark for a retrieval-augmented QA system.\n\n"
        "Below are passages from TWO DIFFERENT academic papers that were matched "
        "because they discuss a related theme.\n\n"
        f"{blocks}\n\n"
        "Write ONE comparison/contrast question whose complete answer genuinely "
        "requires BOTH passages. If, despite the matching, the two passages do "
        "NOT share a substantive comparable theme, instead return "
        '{"question": "SKIP", "ground_truth": "SKIP"}.\n\n'
        f"{_RULES}"
    )
    out = _ask_json(llm, prompt)
    if not out or out.get("question") in (None, "", "SKIP"):
        return None
    if not _is_clean(out.get("question", ""), out.get("ground_truth", "")):
        return None
    return {
        "question":     out["question"].strip(),
        "ground_truth": out["ground_truth"].strip(),
        "source_file":  [p1, p2],
        "category":     "cross_paper_comparison",
        "difficulty":   "hard",
        "_gen_type":    "cross_paper_comparison",
    }


def gen_paraphrase_hard(llm, corpus) -> Optional[Dict[str, Any]]:
    papers = [p for p, cs in corpus.items() if cs]
    if not papers:
        return None
    paper = random.choice(papers)
    chunk = random.choice(corpus[paper])
    prompt = (
        "You are building a hard retrieval benchmark.\n\n"
        f"{_chunk_block(chunk, 'PASSAGE')}\n\n"
        "Write ONE question answerable from this passage, BUT phrase the question "
        "using synonyms and paraphrase so it shares as few content words with the "
        "passage as possible (this tests semantic, not keyword, retrieval). "
        "The gold answer must be 40-90 words, fully supported, synthesizing the "
        "passage's key facts.\n"
        f"{_CONTENT_RULE}"
        'Return ONLY JSON: {"question": "...", "ground_truth": "..."}'
    )
    out = _ask_json(llm, prompt)
    if not out or not _is_clean(out.get("question", ""), out.get("ground_truth", "")):
        return None
    return {
        "question":     out["question"].strip(),
        "ground_truth": out["ground_truth"].strip(),
        "source_file":  [paper],
        "category":     "paraphrase_hard_retrieval",
        "difficulty":   "medium",
        "_gen_type":    "paraphrase_hard",
    }


def gen_deep_single(llm, corpus) -> Optional[Dict[str, Any]]:
    papers = [p for p, cs in corpus.items() if cs]
    if not papers:
        return None
    paper = random.choice(papers)
    chunk = max(random.sample(corpus[paper], min(4, len(corpus[paper]))),
                key=lambda c: len(c.get("text", "")))
    prompt = (
        "You are building a benchmark for a retrieval-augmented QA system.\n\n"
        f"{_chunk_block(chunk, 'PASSAGE')}\n\n"
        "Write ONE explanatory question (how/why/what-mechanism) whose correct "
        "answer is a rich, multi-fact explanation of 50-110 words, fully supported "
        "by the passage.\n"
        f"{_CONTENT_RULE}"
        'Return ONLY JSON: {"question": "...", "ground_truth": "..."}'
    )
    out = _ask_json(llm, prompt)
    if not out or not _is_clean(out.get("question", ""), out.get("ground_truth", "")):
        return None
    return {
        "question":     out["question"].strip(),
        "ground_truth": out["ground_truth"].strip(),
        "source_file":  [paper],
        "category":     "definition_explanation",
        "difficulty":   "medium",
        "_gen_type":    "deep_single_source",
    }


_GENERATORS = {
    "multi_chunk_synthesis":  gen_multi_chunk,
    "cross_paper_comparison": gen_cross_paper,
    "paraphrase_hard":        gen_paraphrase_hard,
    "deep_single_source":     gen_deep_single,
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    global _CLIENT
    ap = argparse.ArgumentParser("Generate hard benchmark questions")
    ap.add_argument("--per-type", type=int, default=3)
    ap.add_argument("--out", default="maddpg/results/benchmark_splits/generated_candidates.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--types", nargs="*", default=QUESTION_TYPES, choices=QUESTION_TYPES)
    args = ap.parse_args()

    random.seed(args.seed)
    _CLIENT = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    corpus = load_corpus(_CLIENT)
    print(f"[gen] corpus: {len(corpus)} papers, "
          f"{sum(len(v) for v in corpus.values())} usable chunks")
    print(f"[gen] model: {current_model()}  | per-type: {args.per_type}")

    llm = build_llm(temperature=0.7)
    candidates: List[Dict[str, Any]] = []

    for qtype in args.types:
        gen_fn = _GENERATORS[qtype]
        made, attempts = 0, 0
        while made < args.per_type and attempts < args.per_type * 5:
            attempts += 1
            q = gen_fn(llm, corpus)
            if q:
                candidates.append(q)
                made += 1
                print(f"  [{qtype}] {made}/{args.per_type}: {q['question'][:80]}")
        if made < args.per_type:
            print(f"  [{qtype}] WARNING: only generated {made}/{args.per_type}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n[gen] wrote {len(candidates)} candidate questions -> {out_path}")
    print("\n=== REVIEW SAMPLE ===")
    for c in candidates[:12]:
        words = len(c["ground_truth"].split())
        print(f"\n[{c['category']} / {c['difficulty']}]  sources={c['source_file']}")
        print(f"  Q: {c['question']}")
        print(f"  A ({words}w): {c['ground_truth']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
