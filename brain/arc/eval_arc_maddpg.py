"""
brain/arc/eval_arc_maddpg.py
----------------------------
Evaluate the trained MADDPG controller on ARC multiple-choice questions.

ARC is multiple-choice, so the metric is accuracy (did the controller pick the
correct letter), not Token F1. This harness:

  1. Points retrieval at the `arc_corpus` Qdrant collection (via QDRANT_COLLECTION).
  2. Runs one greedy MADDPG episode per question on an MCQ-formatted query.
  3. Parses the chosen letter out of the controller's free-form answer.
  4. Scores accuracy against answerKey.

The checkpoint's state_dim selects CEB (20) vs base (14) automatically — so the
same harness evaluates both the CEB and no-CEB controllers.

Usage (from brain/):
    python arc/eval_arc_maddpg.py --checkpoint maddpg/results/maddpg_v4/checkpoints/ep_0200.pt \\
        --benchmark arc/data/arc_benchmark_60q.jsonl --tag maddpg_ceb
"""
from __future__ import annotations

import argparse
import json
import os
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

_ARC_DIR = Path(__file__).resolve().parent
_DATA = _ARC_DIR / "data"
_BRAIN_ROOT = _ARC_DIR.parent
if str(_BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRAIN_ROOT))

# Retrieval collection — set before any retriever import.
os.environ["QDRANT_COLLECTION"] = os.environ.get("QDRANT_COLLECTION", "arc_corpus")

try:
    from dotenv import load_dotenv
    for _ep in (_BRAIN_ROOT / ".env", _BRAIN_ROOT.parent / ".env"):
        if _ep.exists():
            load_dotenv(_ep)            # override=False: keeps QDRANT_COLLECTION
            break
except ImportError:
    pass

import numpy as np
import torch

import context_marl_ac.config as cfg
from context_marl_ac.marl.marl_env import MARLEnv
from maddpg.context_engineering_block import CEB_STATE_DIM, build_ceb_features
from maddpg.stage_utils import find_active_agent_and_valid_actions
from maddpg.trainer import StageConditionedMADDPGTrainer, TrainerConfig


def _mcq_query(question: str, choices: List[dict]) -> str:
    block = "\n".join(f"{c['label']}) {c['text']}" for c in choices)
    labels = ", ".join(c["label"] for c in choices)
    return (f"{question}\n\nChoices:\n{block}\n\n"
            f"Select the single best answer. Respond with only the choice "
            f"letter ({labels}).")


def _parse_letter(reply: str, choices: List[dict]) -> str:
    valid = [c["label"] for c in choices]
    upper = {v.upper(): v for v in valid}
    for tok in re.findall(r"[A-Za-z0-9]+", reply or ""):
        if tok.upper() in upper:
            return upper[tok.upper()]
    low = (reply or "").lower()
    for c in choices:
        if c["text"] and c["text"].lower() in low:
            return c["label"]
    return ""


def main() -> int:
    ap = argparse.ArgumentParser("Evaluate the MADDPG controller on ARC MCQ")
    ap.add_argument("--checkpoint", required=True, help="Path to the trained .pt checkpoint")
    ap.add_argument("--benchmark", default=str(_DATA / "arc_benchmark_60q.jsonl"))
    ap.add_argument("--tag", default="maddpg", help="Label for the output file")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg.DRY_RUN = False
    if not os.path.exists(args.checkpoint):
        print(f"[arc-maddpg] checkpoint not found: {args.checkpoint}")
        return 1

    # state_dim in the checkpoint => CEB (20) vs base (14).
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dim = int(ckpt.get("state_dim", CEB_STATE_DIM))
    use_ceb = (state_dim == CEB_STATE_DIM)
    hidden_dim = int(ckpt.get("hidden_dim", args.hidden_dim))

    trainer = StageConditionedMADDPGTrainer(
        TrainerConfig(state_dim=state_dim, hidden_dim=hidden_dim, device="cpu"))
    trainer.load_checkpoint(Path(args.checkpoint))
    env = MARLEnv()
    print(f"[arc-maddpg] tag={args.tag}  state_dim={state_dim} (CEB={use_ceb})  "
          f"updates={trainer.total_gradient_updates}  collection={os.environ['QDRANT_COLLECTION']}")

    def state_features() -> np.ndarray:
        if use_ceb:
            return np.array(build_ceb_features(env.state), dtype=np.float32)
        return np.array(env.get_global_features(), dtype=np.float32)

    with open(args.benchmark, encoding="utf-8") as fh:
        questions = [json.loads(l) for l in fh if l.strip()]
    print(f"[arc-maddpg] {len(questions)} questions")

    results: List[Dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        query = _mcq_query(q["question"], q["choices"])
        t0 = time.time()
        status, answer = "ok", ""
        final_status, visited, steps = "", [], 0
        try:
            state = env.reset({"question": query, "ground_truth": "", "source_file": []}, index=1)
            done = False
            # The verifier-driven retry loop can time out and clear the answer,
            # so capture each generator output as it is produced.
            last_answer = ""
            while not done and steps < cfg.MAX_STEPS_PER_EPISODE + 2:
                active_agent, valid_actions = find_active_agent_and_valid_actions(env)
                if active_agent is None:
                    break
                visited.append(active_agent)
                obs = state_features()
                _raw, params, discrete = trainer.select_action(
                    active_agent, obs, valid_actions, explore=False)   # greedy
                state, _r, done, _info = env.step(active_agent, discrete, params=params)
                steps += 1
                if (state.generated_answer or "").strip():
                    last_answer = state.generated_answer.strip()
            answer = (state.generated_answer or "").strip() or last_answer
            final_status = getattr(state, "final_status", "")
            if not answer:
                status = "no_answer"
        except Exception as exc:
            status = "error"
            print(f"  [err] {q['id']}: {exc}")
        predicted = _parse_letter(answer, q["choices"])
        correct = int(predicted == q["answerKey"])
        results.append({
            "id": q["id"], "arc_split": q.get("arc_split", "ARC-Challenge"),
            "question": q["question"], "answerKey": q["answerKey"],
            "predicted": predicted, "correct": correct, "status": status,
            "final_status": final_status, "agents_visited": visited, "num_steps": steps,
            "reply": answer, "latency_seconds": round(time.time() - t0, 3),
        })
        mark = "OK " if correct else "XX "
        print(f"  [{i:3d}/{len(questions)}] {mark} pred={predicted or '?'} "
              f"gold={q['answerKey']}  ({time.time()-t0:.1f}s)")

    n = len(results) or 1
    acc = sum(r["correct"] for r in results) / n
    print(f"\n=== ARC accuracy ({args.tag}) ===")
    print(f"  accuracy: {acc:.4f}  ({sum(r['correct'] for r in results)}/{len(results)})")

    out = Path(args.out) if args.out else (_DATA / f"arc_eval_{args.tag}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
