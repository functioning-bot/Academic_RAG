"""
brain/validate_benchmark_questions.py
-------------------------------------
Phase 3 of benchmark expansion: validate LLM-generated candidate questions.

Two independent checks per candidate:

  1. RETRIEVAL CHECK — run hybrid retrieval for the question and verify the
     gold source paper(s) actually appear in the retrieved chunks. If the RAG
     system cannot even surface the source, the question is unanswerable in
     practice and must be dropped.

  2. LLM-JUDGE — given the question, the gold answer, and the actually-retrieved
     chunks, an LLM judges: is the question clear/answerable, is the gold answer
     accurate w.r.t. retrievable evidence, and does answering genuinely require
     more than one chunk (so synthesis questions are not secretly trivial).

Status assigned per candidate:
  pass    — retrievable + answerable + gold accurate (+ multi-chunk for synthesis)
  review  — usable but flagged for human eyes (partial source recall, or a
            synthesis question the judge thinks is single-chunk answerable)
  fail    — source not retrievable, or judge says unanswerable / gold wrong

Output: a JSONL with a `_validation` block added to every row, plus a printed
summary. Phase 4 (human review) then keeps pass + vetted review rows.

Usage (from brain/):
  python validate_benchmark_questions.py \
      --in  maddpg/results/benchmark_splits/generated_candidates.jsonl \
      --out maddpg/results/benchmark_splits/validated_candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

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

from langchain_core.messages import HumanMessage
from llm_config import build_llm, current_model
import retriever_shared

SYNTHESIS_CATEGORIES = {"multi_chunk_synthesis", "cross_paper_comparison"}
RETRIEVAL_TOP_K = 12


def _src_base(s: str) -> str:
    """Normalize a source name so 'X.pdf' and 'X_p3' compare equal."""
    return s.split("_p")[0].strip().lower()


def _retrieved_sources(chunks: List[Dict[str, Any]]) -> set:
    out = set()
    for c in chunks:
        meta = c.get("metadata", {}) if isinstance(c.get("metadata"), dict) else {}
        sf = meta.get("source_file") or c.get("source_file") or ""
        if sf:
            out.add(_src_base(sf))
    return out


def _ask_judge(llm, question: str, gold: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    ctx = "\n\n".join(
        f"CHUNK {i+1} (paper: {c.get('metadata',{}).get('source_file','?')}):\n"
        f"{(c.get('text') or '')[:900]}"
        for i, c in enumerate(chunks[:8])
    )
    prompt = (
        "You are validating a candidate question for a retrieval-augmented QA "
        "benchmark. You are given the question, its proposed gold answer, and "
        "the chunks a retriever actually returned for this question.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"PROPOSED GOLD ANSWER:\n{gold}\n\n"
        f"RETRIEVED CHUNKS:\n{ctx}\n\n"
        "Judge strictly and return ONLY JSON with these boolean/string fields:\n"
        '{\n'
        '  "answerable": true if the question is clear, well-formed and answerable,\n'
        '  "gold_accurate": true if the gold answer is consistent with and '
        'supported by the retrieved chunks (no invented facts),\n'
        '  "needs_multiple_chunks": true if fully answering requires combining '
        'information from more than one chunk,\n'
        '  "issue": short string describing any problem, or "" if none\n'
        '}'
    )
    for _ in range(2):
        resp = llm.invoke([HumanMessage(content=prompt)])
        txt = resp.content if hasattr(resp, "content") else str(resp)
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return {"answerable": False, "gold_accurate": False,
            "needs_multiple_chunks": False, "issue": "judge_parse_failed"}


def validate_one(llm, cand: Dict[str, Any]) -> Dict[str, Any]:
    question = cand["question"]
    gold = cand["ground_truth"]
    category = cand.get("category", "")
    gold_sources = cand.get("source_file", [])
    if isinstance(gold_sources, str):
        gold_sources = [gold_sources]
    gold_bases = {_src_base(s) for s in gold_sources}

    # ── Check 1: retrieval ────────────────────────────────────────────────
    try:
        chunks = retriever_shared.retrieve_docs(question)[:RETRIEVAL_TOP_K]
    except Exception as exc:
        return {"status": "fail", "source_recall": 0.0,
                "judge": {}, "issue": f"retrieval_error: {exc}"}

    ret_bases = _retrieved_sources(chunks)
    hit = gold_bases & ret_bases
    source_recall = len(hit) / len(gold_bases) if gold_bases else 0.0

    # ── Check 2: LLM judge ────────────────────────────────────────────────
    judge = _ask_judge(llm, question, gold, chunks)

    # ── Decision ──────────────────────────────────────────────────────────
    if source_recall == 0.0:
        status, issue = "fail", "gold source(s) not retrievable"
    elif not judge.get("answerable", False):
        status, issue = "fail", judge.get("issue") or "judge: not answerable"
    elif not judge.get("gold_accurate", False):
        status, issue = "fail", judge.get("issue") or "judge: gold not accurate"
    elif category in SYNTHESIS_CATEGORIES and not judge.get("needs_multiple_chunks", False):
        status, issue = "review", "synthesis question judged single-chunk answerable"
    elif source_recall < 1.0:
        status, issue = "review", f"partial source recall ({source_recall:.2f})"
    else:
        status, issue = "pass", judge.get("issue", "")

    return {"status": status, "source_recall": round(source_recall, 3),
            "judge": judge, "issue": issue}


def main() -> int:
    ap = argparse.ArgumentParser("Validate candidate benchmark questions")
    ap.add_argument("--in", dest="inp",
                    default="maddpg/results/benchmark_splits/generated_candidates.jsonl")
    ap.add_argument("--out",
                    default="maddpg/results/benchmark_splits/validated_candidates.jsonl")
    args = ap.parse_args()

    cands = [json.loads(l) for l in Path(args.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[validate] {len(cands)} candidates  | judge model: {current_model()}")

    llm = build_llm(temperature=0.0)
    counts = {"pass": 0, "review": 0, "fail": 0}
    out_rows: List[Dict[str, Any]] = []

    for i, cand in enumerate(cands, 1):
        v = validate_one(llm, cand)
        counts[v["status"]] += 1
        cand["_validation"] = v
        out_rows.append(cand)
        print(f"  [{i:3d}/{len(cands)}] {v['status']:7s} "
              f"src_recall={v['source_recall']:.2f}  {cand['category']:24s} "
              f"{('— ' + v['issue']) if v['issue'] else ''}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[validate] wrote {len(out_rows)} rows -> {out_path}")
    print(f"[validate] pass={counts['pass']}  review={counts['review']}  fail={counts['fail']}")
    print(f"[validate] usable (pass + review) = {counts['pass'] + counts['review']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
