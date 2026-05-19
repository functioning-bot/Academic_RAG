"""
brain/maddpg/plot_training.py
-----------------------------
Plot MADDPG training curves from a run's episode_metrics CSV.

Reinforcement learning has no "accuracy" — the learning signal is the episode
reward. This figure shows four panels:

  1. Episode reward            — the RL learning curve (raw + rolling mean)
  2. Answer-quality signals    — verification-pass rate + citation support
                                 (rolling) — the closest analogue to "accuracy"
  3. Critic loss               — temporal-difference loss of the centralized critic
  4. Actor losses              — per-agent deterministic policy-gradient loss

Usage (from brain/):
    python -m maddpg.plot_training --run-dir maddpg/results/maddpg_v4
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _rolling(xs: list[float], w: int) -> list[float]:
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w + 1)
        seg = xs[lo:i + 1]
        out.append(sum(seg) / len(seg))
    return out


def _col(rows: list[dict], name: str):
    """Return (episodes, values) for rows where the column is numeric."""
    ep, val = [], []
    for r in rows:
        v = r.get(name, "")
        if v not in ("", "None", None):
            ep.append(int(r["episode"]))
            val.append(float(v))
    return ep, val


def main() -> int:
    ap = argparse.ArgumentParser("Plot MADDPG training curves")
    ap.add_argument("--run-dir", required=True, help="Run directory under maddpg/results/")
    ap.add_argument("--window", type=int, default=15, help="Rolling-mean window")
    ap.add_argument("--out", default="", help="Output PNG path")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    csvs = list((run_dir / "metrics").glob("episode_metrics_*.csv"))
    if not csvs:
        print(f"[plot] no episode_metrics CSV in {run_dir / 'metrics'}")
        return 1
    rows = list(csv.DictReader(open(csvs[0], encoding="utf-8")))
    print(f"[plot] {csvs[0].name}: {len(rows)} episodes")

    ep_r, reward = _col(rows, "total_reward")
    ep_c, critic = _col(rows, "critic_loss")
    ep_vp, vpass = _col(rows, "verification_pass")
    ep_cs, csup = _col(rows, "citation_support")
    actors = ["retriever", "rewriter", "grader", "generator", "verifier"]
    w = args.window

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"MADDPG training curves — {run_dir.name}", fontsize=14, fontweight="bold")

    # 1. Episode reward
    ax[0, 0].plot(ep_r, reward, color="#bbb", lw=0.8, label="per episode")
    ax[0, 0].plot(ep_r, _rolling(reward, w), color="#1f77b4", lw=2.2,
                  label=f"rolling mean (w={w})")
    ax[0, 0].set_title("Episode reward (RL learning curve)")
    ax[0, 0].set_xlabel("episode"); ax[0, 0].set_ylabel("total reward")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

    # 2. Answer-quality signals (the "accuracy"-like panel)
    ax[0, 1].plot(ep_vp, _rolling(vpass, w), color="#2ca02c", lw=2,
                  label="verification-pass rate")
    ax[0, 1].plot(ep_cs, _rolling(csup, w), color="#ff7f0e", lw=2,
                  label="citation support")
    ax[0, 1].set_title(f"Answer-quality signals (rolling, w={w})")
    ax[0, 1].set_xlabel("episode"); ax[0, 1].set_ylabel("rate / score")
    ax[0, 1].set_ylim(0, 1.05); ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

    # 3. Critic loss
    ax[1, 0].plot(ep_c, critic, color="#d62728", lw=1.4)
    ax[1, 0].set_title("Critic loss (TD error)")
    ax[1, 0].set_xlabel("episode"); ax[1, 0].set_ylabel("MSE loss")
    ax[1, 0].grid(alpha=0.3)
    if critic and max(critic) > 0:
        ax[1, 0].set_yscale("log")

    # 4. Actor losses
    for agent in actors:
        ep_a, loss = _col(rows, f"actor_loss_{agent}")
        if ep_a:
            ax[1, 1].plot(ep_a, _rolling(loss, w), lw=1.6, label=agent)
    ax[1, 1].set_title(f"Actor losses, per agent (rolling, w={w})")
    ax[1, 1].set_xlabel("episode"); ax[1, 1].set_ylabel("policy-gradient loss")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.out) if args.out else (run_dir / "training_curves.png")
    fig.savefig(out, dpi=130)
    print(f"[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
