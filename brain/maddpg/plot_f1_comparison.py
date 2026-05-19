"""
brain/maddpg/plot_f1_comparison.py
----------------------------------
Token-F1 comparison: MADDPG v4 CEB vs no-CEB on eval60.

Each variant was evaluated 3x on the same 60 held-out questions. This plots the
per-run means as points, the 3-run mean as a bar, and a 95% confidence interval
(from the per-question standard error of the 3-run-averaged scores).

Usage (from brain/):
    python -m maddpg.plot_f1_comparison
"""
from __future__ import annotations

import json
import math
import re
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_RESULTS = Path(__file__).resolve().parent / "results"
_T59 = 2.0010   # t(0.975, df=59)

_RUNS = {
    "CEB": [
        "eval60_compare/maddpg_ceb/eval_maddpg_ceb.jsonl",
        "eval60_ceb_run1/eval_maddpg_ceb.jsonl",
        "eval60_ceb_run2/eval_maddpg_ceb.jsonl",
    ],
    "no-CEB": [
        "eval60_compare/maddpg_no_ceb/eval_maddpg_no_ceb.jsonl",
        "eval60_noceb_run2/eval_maddpg_no_ceb.jsonl",
        "eval60_noceb_run3/eval_maddpg_no_ceb.jsonl",
    ],
}


def _f1(pred: str, gold: str) -> float:
    p = set(re.findall(r"\b\w+\b", (pred or "").lower()))
    g = set(re.findall(r"\b\w+\b", (gold or "").lower()))
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if not tp:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def _run_f1s(path: Path) -> list[float]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [_f1(r.get("final_answer", ""), r.get("ground_truth", "")) for r in rows]


def main() -> int:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    labels = list(_RUNS)
    colors = {"CEB": "#1f77b4", "no-CEB": "#ff7f0e"}

    for x, variant in enumerate(labels):
        per_run = [_run_f1s(_RESULTS / p) for p in _RUNS[variant]]
        run_means = [st.mean(v) for v in per_run]
        # 3-run-averaged per-question scores -> mean + 95% CI
        n = len(per_run[0])
        avg_q = [st.mean([per_run[r][i] for r in range(3)]) for i in range(n)]
        mean = st.mean(avg_q)
        se = st.stdev(avg_q) / math.sqrt(n)
        ci = _T59 * se

        ax.bar(x, mean, width=0.5, color=colors[variant], alpha=0.55,
               yerr=ci, capsize=8, ecolor="black",
               label=f"{variant}: {mean:.4f} ± {ci:.4f}")
        ax.scatter([x] * 3, run_means, color=colors[variant], edgecolor="black",
                   zorder=5, s=70)
        for rm in run_means:
            ax.annotate(f"{rm:.4f}", (x + 0.07, rm), fontsize=8, va="center")
        ax.annotate(f"mean {mean:.4f}", (x, mean), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"MADDPG v4\n{l}" for l in labels])
    ax.set_ylabel("Token F1 (eval60, n = 60)")
    ax.set_title("MADDPG v4 — Token F1: CEB vs no-CEB\n"
                 "(3 eval runs each; bar = 3-run mean, error bar = 95% CI)")
    ax.set_ylim(0.35, 0.50)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    out = _RESULTS / "f1_comparison_ceb_vs_noceb.png"
    fig.savefig(out, dpi=130)
    print(f"[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
