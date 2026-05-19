"""
brain/maddpg/serve_maddpg.py
----------------------------
FastAPI server that exposes the trained MADDPG controller to the web frontend.

It implements the SAME HTTP contract as the other architectures' `main.py`
servers (simple_hybrid_rag, final_arch, ...), so `web_app/` works against it
unchanged:

    GET  /                     — info
    GET  /health               — {status: "ok"}  (frontend polls this)
    GET  /health/dependencies  — Qdrant + LLM provider status
    POST /ask  {query}         — {answer, context_used, citations[]}

On `/ask` it runs one greedy MADDPG episode: the trained per-agent actors drive
the stage-gated RAG pipeline (retriever -> grader -> generator -> verifier),
choosing continuous parameters at each stage, and the generated answer plus its
cited evidence are returned.

Activate (from brain/):
    python -m maddpg.serve_maddpg
    python -m maddpg.serve_maddpg --checkpoint <path.pt> --port 8000

Then open web_app/index.html — the frontend talks to port 8000 by default.
"""
from __future__ import annotations

import argparse
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List

# ── sys.path: brain/ importable ───────────────────────────────────────────────
_BRAIN_ROOT = Path(__file__).resolve().parent.parent
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

# ── dotenv (local paths only — never print API keys) ─────────────────────────
try:
    from dotenv import load_dotenv
    for _ep in (_BRAIN_ROOT / ".env", _BRAIN_ROOT.parent / ".env"):
        if _ep.exists():
            load_dotenv(dotenv_path=_ep)
            break
except ImportError:
    pass

import os

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import context_marl_ac.config as cfg
from api_models import QueryRequest, QueryResponse, CitationItem
from context_marl_ac.marl.marl_env import MARLEnv

from .context_engineering_block import CEB_STATE_DIM, build_ceb_features
from .stage_utils import find_active_agent_and_valid_actions
from .trainer import StageConditionedMADDPGTrainer, TrainerConfig

import numpy as np

ARCHITECTURE_NAME = "maddpg_stage_conditioned"
# Use the FINAL periodic checkpoint (fully trained), NOT best_reward.pt.
# best_reward.pt is saved at the highest single-episode *training reward*, which
# is noisy and can land on an early, barely-trained (even untrained) episode.
_DEFAULT_CKPT = _BRAIN_ROOT / "maddpg" / "results" / "maddpg_v4" / "checkpoints" / "ep_0200.pt"

# Serve-time generator token floor. continuous_action_mapper caps the generator
# at 384 tokens during *training* (to protect the Groq free-tier TPM budget);
# that ceiling truncates served answers mid-sentence. We lift max_tokens to this
# floor for inference only — the training mapper and checkpoint are untouched.
_SERVE_MIN_TOKENS = int(os.getenv("MADDPG_SERVE_MIN_TOKENS", "1024"))

# ── Global state (loaded once at startup) ────────────────────────────────────
_trainer: StageConditionedMADDPGTrainer | None = None
_env: MARLEnv | None = None
_use_ceb: bool = True
_ckpt_path: str = ""
_lock = threading.Lock()   # the env + actors are single-instance; serialize /ask


def _state_features(env: MARLEnv) -> np.ndarray:
    if _use_ceb:
        return np.array(build_ceb_features(env.state), dtype=np.float32)
    return np.array(env.get_global_features(), dtype=np.float32)


def _load_controller(ckpt_path: str) -> None:
    """Load the trained MADDPG checkpoint once, at server startup."""
    global _trainer, _env, _use_ceb, _ckpt_path
    cfg.DRY_RUN = False
    _ckpt_path = ckpt_path

    if not os.path.exists(ckpt_path):
        raise RuntimeError(
            f"MADDPG checkpoint not found: {ckpt_path}\n"
            f"Pass --checkpoint <path>, or train a model first."
        )

    # The checkpoint records its state_dim; 20 => Context Engineering Block.
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dim = int(ckpt.get("state_dim", CEB_STATE_DIM))
    _use_ceb = (state_dim == CEB_STATE_DIM)
    hidden_dim = int(ckpt.get("hidden_dim", 128))

    tcfg = TrainerConfig(state_dim=state_dim, hidden_dim=hidden_dim, device="cpu")
    _trainer = StageConditionedMADDPGTrainer(tcfg)
    _trainer.load_checkpoint(Path(ckpt_path))
    _env = MARLEnv()

    print(f"[serve_maddpg] loaded {ckpt_path}")
    print(f"[serve_maddpg]   state_dim={state_dim} (CEB={_use_ceb})  "
          f"gradient_updates={_trainer.total_gradient_updates}  "
          f"trained={_trainer.total_gradient_updates > 0}")


def _run_episode(query: str) -> Dict[str, Any]:
    """Run one greedy MADDPG episode for a user query; return answer + evidence."""
    assert _trainer is not None and _env is not None
    state = _env.reset({"question": query, "ground_truth": "", "source_file": []}, index=1)
    done = False
    steps = 0

    while not done and steps < cfg.MAX_STEPS_PER_EPISODE + 2:
        active_agent, valid_actions = find_active_agent_and_valid_actions(_env)
        if active_agent is None:
            if state.final_status == "pending":
                state.final_status = "abstained"
            state.done = True
            break

        obs = _state_features(_env)
        raw, params, discrete = _trainer.select_action(
            active_agent, obs, valid_actions, explore=False)   # greedy: no noise

        # Serve-time only: lift the generator's training-time 384-token cap so
        # answers are not truncated mid-sentence. Floored, not fixed, so the
        # actor can still ask for more than the floor if it ever learns to.
        if active_agent == "generator" and isinstance(params, dict):
            floored = max(int(params.get("max_tokens", 0)), _SERVE_MIN_TOKENS)
            params = {**params, "max_tokens": floored}

        try:
            new_state, _reward, done, _info = _env.step(active_agent, discrete, params=params)
            state = new_state
        except Exception as exc:
            state.final_status = "error"
            state.done = True
            raise RuntimeError(f"pipeline error at stage '{active_agent}': {exc}") from exc
        steps += 1

    return {
        "answer":            state.generated_answer or "",
        "final_status":      state.final_status,
        "selected_evidence": state.selected_evidence or [],
    }


# Words too generic to signal that an evidence chunk actually grounds the answer.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "be", "been", "being", "for", "on", "at", "by", "with", "as", "that", "this",
    "these", "those", "it", "its", "from", "which", "what", "when", "where", "how",
    "why", "who", "will", "would", "can", "could", "should", "may", "might", "do",
    "does", "did", "has", "have", "had", "not", "but", "if", "than", "then", "into",
    "out", "also", "such", "more", "most", "other", "based", "using", "used",
    "their", "there", "they", "them", "while", "between", "both", "each", "any",
}
_WORD_RE = re.compile(r"[a-z][a-z-]{2,}")


def _content_words(text: str) -> set:
    return {w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOPWORDS}


def _relevant_evidence(query: str, evidence: List[Dict[str, Any]],
                       max_keep: int = 6) -> List[Dict[str, Any]]:
    """Keep only the evidence chunks actually relevant to the question.

    The MADDPG retriever pulls a broad top-k across every indexed paper, and
    the grader keeps a *ratio* of them — so the kept pack can carry chunks from
    unrelated papers that the generator never used. Those would otherwise show
    up as nonsensical citations.

    Two signals, combined:
      1. `retrieval_score` (hybrid dense+sparse RRF) — a genuinely relevant
         chunk matches the query well in both retrievers and scores high;
         off-topic filler that only made the top-k scores low. This is the
         primary ranking, and we cap at `max_keep`.
      2. Question-keyword overlap — a guard that drops any surviving chunk that
         shares fewer than two of the question's content words (clearly a
         different topic), in case scores are unavailable or anomalous.
    """
    if not evidence:
        return evidence
    q = _content_words(query)
    scores = [float(e.get("retrieval_score", 0.0)) for e in evidence]

    if any(s > 0 for s in scores):
        ranked = sorted(evidence,
                        key=lambda e: float(e.get("retrieval_score", 0.0)),
                        reverse=True)
    elif q:   # no usable scores — rank by question-keyword overlap instead
        ranked = sorted(evidence,
                        key=lambda e: len(q & _content_words(e.get("text", ""))),
                        reverse=True)
    else:
        ranked = list(evidence)

    top = ranked[:max_keep]
    if not q:
        return top
    kept = [e for e in top if len(q & _content_words(e.get("text", ""))) >= 2]
    return kept or ranked[:1]   # never strip everything


def _attribute_citations(answer: str,
                         evidence: List[Dict[str, Any]]) -> tuple:
    """Insert inline [n] markers into the answer and return the numbered sources.

    The generator is prompted to produce no inline citations (so answer text
    stays clean for evaluation). Here, post-hoc, each answer sentence is matched
    to the evidence chunk(s) it overlaps most, and a [n] marker is appended.
    `n` is the 1-based index into the returned (de-duplicated) source list, so
    the markers line up with the "Sources" footer the frontend renders.

    Returns (marked_answer, ordered_unique_evidence).
    """
    if not evidence or not answer.strip():
        return answer, evidence

    # De-duplicate by (source, page, section) -> the numbered source list.
    seen: Dict[tuple, Dict[str, Any]] = {}
    ordered: List[Dict[str, Any]] = []
    for e in evidence:
        key = (e.get("source"), str(e.get("page")), e.get("section"))
        if key in seen:
            seen[key]["text"] = (seen[key].get("text", "") + " "
                                 + e.get("text", "")).strip()
            continue
        item = dict(e)
        seen[key] = item
        ordered.append(item)

    cit_words = [_content_words(e.get("text", "")) for e in ordered]
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())

    marked: List[str] = []
    for sent in sentences:
        sw = _content_words(sent)
        picks: List[int] = []
        if sw:
            scores = sorted(((len(sw & cw), i) for i, cw in enumerate(cit_words)),
                            reverse=True)
            if scores and scores[0][0] >= 2:
                picks.append(scores[0][1])
                # include a close second source if it also clearly supports it
                for ov, i in scores[1:]:
                    if ov >= 2 and ov >= scores[0][0] - 1:
                        picks.append(i)
                        break
        marker = "".join(f"[{i + 1}]" for i in sorted(picks))
        marked.append(f"{sent} {marker}".strip() if marker else sent)

    return " ".join(marked), ordered


def _citations(evidence: List[Dict[str, Any]]) -> List[CitationItem]:
    """Convert the evidence pack the answer was generated from into citations."""
    out: List[CitationItem] = []
    for e in evidence:
        meta = e.get("metadata") if isinstance(e.get("metadata"), dict) else {}
        out.append(CitationItem(
            source_file=str(e.get("source") or meta.get("source_file") or "Unknown Source"),
            page_number=e.get("page") or meta.get("page_number") or "Unknown Page",
            section_header=str(e.get("section") or meta.get("section_header") or "Unknown Section"),
            excerpt=str(e.get("text", "")),
            content_type=str(e.get("content_type") or meta.get("content_type") or "text"),
        ))
    return out


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Academic RAG Brain API — MADDPG Controller")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Brain API is running.",
        "architecture": ARCHITECTURE_NAME,
        "checkpoint": _ckpt_path,
        "trained": bool(_trainer and _trainer.total_gradient_updates > 0),
        "endpoints": {"ask": "POST /ask", "health": "GET /health"},
    }


@app.get("/health")
def health():
    return {"status": "ok", "architecture": ARCHITECTURE_NAME}


@app.get("/health/dependencies")
def dependencies_health():
    qdrant_status, qdrant_error = "ok", None
    try:
        from qdrant_client import QdrantClient
        QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333")).get_collections()
    except Exception as e:
        qdrant_status, qdrant_error = "error", str(e)
    provider = (os.getenv("LLM_PROVIDER") or "groq").lower()
    key_env = "OPENAI_API_KEY" if provider == "openai" else "GROQ_API_KEY"
    return {
        "architecture": ARCHITECTURE_NAME,
        "qdrant": {"status": qdrant_status, "error": qdrant_error},
        "llm": {"provider": provider, "status": "ok" if os.getenv(key_env) else "missing_api_key"},
        "controller": {"checkpoint": _ckpt_path,
                        "trained": bool(_trainer and _trainer.total_gradient_updates > 0)},
    }


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    """Run the MADDPG-controlled RAG pipeline for one query (sync; blocking)."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if _trainer is None:
        raise HTTPException(status_code=503, detail="Controller not loaded.")

    try:
        with _lock:   # single shared env + actors — serialize requests
            result = _run_episode(request.query.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail={
            "message": "MADDPG pipeline failed.",
            "architecture": ARCHITECTURE_NAME,
            "error": str(exc),
        })

    answer = result["answer"] or "I could not produce a grounded answer for this question."
    # Show only the evidence relevant to the question — not the whole broad
    # retrieved pack, which can include chunks from unrelated papers.
    evidence = _relevant_evidence(request.query.strip(), result["selected_evidence"])
    # Insert inline [n] markers tying each answer sentence to its source(s).
    answer, evidence = _attribute_citations(answer, evidence)
    return QueryResponse(
        answer=answer,
        context_used=[f"[EVIDENCE] {e.get('text', '')}" for e in evidence],
        citations=_citations(evidence),
    )


def main() -> int:
    ap = argparse.ArgumentParser("Serve the MADDPG controller to the web frontend")
    ap.add_argument("--checkpoint", default=str(_DEFAULT_CKPT),
                    help="Path to the trained MADDPG .pt checkpoint")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    _load_controller(args.checkpoint)
    print(f"[serve_maddpg] starting on http://{args.host}:{args.port}  "
          f"(frontend: open web_app/index.html)")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
