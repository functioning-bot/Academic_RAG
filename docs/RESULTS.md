# Results — Stage-Conditioned MADDPG-style RL for Academic RAG

Full experimental record, honestly reported. Read alongside `PROJECT_REVIEW.md`
(architecture) and `MADDPG_EXTENSION.md` (the MADDPG adaptation).

> **Correction notice (this revision).** Every MADDPG number in earlier drafts
> of this document was read from `best_reward.pt`. That checkpoint is saved at
> the highest *single-episode training reward*, which is noisy — for the v4
> 200-episode run it landed on episode ~9, with **0 gradient updates**: an
> untrained, randomly-initialised policy. The previously-reported "0.474 Token
> F1 / parity with the baseline" was therefore the score of an *untrained*
> controller, not the trained one. This revision reports the genuinely trained
> checkpoints (`ep_0200.pt`, 392 gradient updates, and the intermediate
> checkpoints). The headline finding changed as a result — see §1.

---

## 1. Headline

The genuinely **trained** stage-conditioned MADDPG controller (v4, `ep_0200.pt`,
392 gradient updates) scores **0.434 ± 0.031 Token F1** on the 29-question test
set. This is **below** all three working fixed/heuristic baselines, and — the
key finding — **no better than the untrained policy it started from** (0.474).

| System | Token F1 (n=29) | ROUGE-L | Pass rate | Failure rate | Mean latency |
|---|---:|---:|---:|---:|---:|
| `self_rag_grader` | 0.490 ± 0.028 | 0.390 | 1.00 | 0.00 | 4.3 s |
| `simple_hybrid_rag` | 0.487 ± 0.025 | 0.367 | 1.00 | 0.00 | 1.7 s |
| `final_arch` | 0.482 ± 0.027 | 0.354 | 1.00 | 0.00 | 32.8 s |
| `crag_rewrite` | 0.425 ± 0.039 † | 0.314 | 0.86 | 0.14 | 2.0 s |
| **MADDPG v4 (trained)** | **0.434 ± 0.031** | 0.310 | 0.97 | 0.03 | 22.7 s |

**This is a negative result, and it is the honest one.** Training the
stage-conditioned MADDPG controller on the v4 proxy reward did **not** improve
held-out answer quality. The trained controller lands last among the working
systems (tied with the crashed `crag_rewrite`), ~0.05 Token F1 (≈1.3 SE) below
`simple_hybrid_rag`. It is also slow (22.7 s) and the only learned system that
ever fails (1/29).

† `crag_rewrite` crashed on 4 / 29 questions due to a bug in its own node code
(`rewritten_query` referenced before assignment). On the 25 it completed it
scores 0.493; its 0.425 aggregate is dragged down by its own crashes.

**The architectures compared:**
- `simple_hybrid_rag` — hybrid retrieval → top-8 → one generation call.
- `self_rag_grader` — Self-RAG style: retrieval + LLM relevance grading.
- `crag_rewrite` — Corrective-RAG: retrieval evaluation + query rewrite loop.
- `final_arch` — the full pipeline: grade + rerank + generate + claim-verify + retry.
- `MADDPG v4 (trained)` — the learned stage-conditioned continuous-control controller.

**Reading this honestly:** the contribution of this project is *not* a
controller that beats fixed RAG pipelines — it does not. The contributions are
(a) a working stage-conditioned MADDPG-style architecture that trains stably
end-to-end on live LLM calls, and (b) a cautionary, well-diagnosed negative
result: a hand-designed proxy reward (embedding similarity + Token F1 blend +
evidence-utilization term) does not transfer to true held-out answer quality.
See §3 and §6.

---

## 2. Experimental setup

- **Corpus:** 12 academic PDFs (ML/NLP, medical, finance), 1,226 chunks, indexed
  in Qdrant with hybrid dense (BGE-M3) + sparse (BM25) vectors.
- **Benchmark:** 145 questions (expanded from 60) — 88 train / 28 val / 29 test,
  stratified by category and difficulty. Hard synthesis categories dominate.
- **LLM:** OpenAI `gpt-4o-mini` (switched from Groq after free-tier rate limits).
- **Primary metric:** Token F1 of the generated answer vs the gold answer
  (set-overlap F1, identical function across every system; see
  `brain/maddpg/live_maddpg_runner.py:token_f1`).
- **Baseline:** `simple_hybrid_rag` — hybrid RRF retrieval → top-8 → single
  generation call. No grader, verifier, or recovery loop. A lean, strong
  fixed-pipeline baseline.

### 2.1 Checkpoint selection — the bug that invalidated the earlier draft

`train_maddpg.py` saves two kinds of checkpoint: periodic ones (`ep_0050.pt`,
`ep_0100.pt`, …, `ep_0200.pt`) and `best_reward.pt`. **`best_reward.pt` tracks
the highest single-episode *training reward*** — a noisy quantity that, early in
training, can be a lucky random episode. For the v4 200-episode run it was
written at episode ~9 with **0 gradient updates**: the policy was still at its
random initialisation. Every MADDPG evaluation in earlier drafts used
`best_reward.pt`, so every reported MADDPG number described an untrained policy.

The fix: evaluate the **periodic final checkpoint** instead. `ep_0200.pt` has
392 gradient updates and is the policy at the end of training. A `final.pt`
alias is now also written at the end of every run (`train_maddpg.py`), and
`serve_maddpg.py` defaults to `ep_0200.pt`, never `best_reward.pt`.

> **Cross-benchmark caveat.** The benchmark was expanded mid-project, and the
> new gold answers are longer (≈51 words vs ≈27). Longer gold answers inflate
> Token F1 for *every* system. **Absolute Token F1 is only comparable within the
> same benchmark.** Below, every head-to-head compares systems on the *same*
> test set; the gap (Δ vs baseline) is the comparable quantity.

---

## 3. The v4 training curve — Token F1 does not improve with training

The decisive evidence. Five checkpoints from the v4 200-episode run plus the
500-episode run, every one evaluated on the **same 29-question test set** with
the **same** Token F1 function:

| Checkpoint | Gradient updates | Token F1 (n=29) | SE |
|---|---:|---:|---:|
| `best_reward.pt` (untrained, ep ~9) | 0 | 0.474 | 0.028 |
| `ep_0050.pt` | 69 | 0.439 | 0.032 |
| `ep_0100.pt` | 170 | 0.432 | 0.032 |
| `ep_0150.pt` | 286 | 0.445 | 0.024 |
| `ep_0200.pt` (final, 200-ep run) | 392 | 0.434 | 0.031 |
| `best_reward.pt` (500-ep run) | 986 | 0.419 | 0.027 |

**The curve is flat-to-slightly-declining.** From 0 to 986 gradient updates the
held-out Token F1 moves 0.474 → 0.434 → 0.419. Every trained checkpoint sits
within ≈1–1.5 SE of every other, so no pairwise difference is statistically
significant — but the *direction* is unambiguous: training never lifts the
metric above its untrained starting point, and the trend is gently downward.

Plain reading: **for this architecture, reward, and data, training the
controller does not produce a better controller.** The answer quality the system
delivers is set by the underlying RAG pipeline and the LLM; the learned
continuous-control policy does not add measurable value on top.

---

## 4. The iterative experiment log

Each step was a targeted fix for a diagnosed failure mode. The comparable
quantity is the gap to `simple_hybrid_rag` on the same test set.

| Step | Change | Benchmark | MADDPG Token F1 | Baseline | Gap |
|---|---|---|---:|---:|---:|
| v1 | First trained policy (30 ep) | old (18 test) | 0.340 | 0.383 | −11.3% |
| v2 | Raised `evidence_keep_ratio` floor 0.1→0.5 (200 ep) | old | 0.331 | 0.383 | −13.6% |
| v3 | Reward rebalance + `source_diversity` hard-floor (200 ep) | old | 0.340 | 0.383 | −11.3% |
| v3-exp | Un-throttled `top_k` 3-12→5-20; retrained on expanded benchmark | expanded (29 test) | 0.445 | 0.487 | −8.6% |
| **v4** | **Reward redesign (200 ep), trained `ep_0200.pt` — see §5** | expanded | **0.434** | 0.487 | **−10.9%** |
| v4-500 | Same as v4 but 500 ep (`best_reward.pt`, 986 updates) | expanded | 0.419 | 0.487 | −14.0% |

> **Note on v1–v3.** These earlier rows were also read from `best_reward.pt`.
> Their update counts (v1≈49, v3≈211, v3-exp≈283) show they were at least
> partially trained, unlike the v4 200-ep `best_reward.pt`. They are kept here
> for the diagnostic narrative (the reward-hacking story in §4.1) but should be
> read as approximate; the v4 row is the one evaluated on a verified-trained
> periodic checkpoint.

### 4.1 The recurring failure mode (still a valid finding)

Across v1–v3-exp the trained policy converged on the same **minimal-evidence
reward-hack**: it used ~1–2 evidence chunks per question. This satisfied the
verifier (few claims, all trivially supported) and was cheap, but produced thin
answers. Raising floors and rebalancing weights did not fix it — the policy kept
finding a different lever to strip evidence (`source_diversity`, then
`relevance_threshold`, then `evidence_keep_ratio`).

The diagnosis: **the reward did not actually reward using evidence.** Token F1 +
verifier-pass + citation-support are all satisfiable with few chunks. The v4
reward redesign (§5) fixed *that specific behaviour* — evidence count rose from
~2 to 5.3 — but, as §3 shows, fixing the reward-hack did not translate into
better answers on held-out questions.

---

## 5. The v4 reward redesign

Three changes, all targeting the minimal-evidence root cause:

1. **Answer quality → semantic blend.** Replaced pure Token F1 with
   `0.5 × embedding-similarity(answer, gold) + 0.5 × Token F1`.
2. **New evidence-utilization reward term.**
   `W_EVIDENCE_UTILIZATION × min(evidence_count, 6)/6` — the ~2-chunk strategy
   now loses reward outright. Capped at 6 chunks to prevent over-retrieval gaming.
3. **`evidence_keep_ratio` floor 0.5 → 0.7.**

Reward weights: `W_ANSWER_QUALITY=0.45, W_CITATION_SUPPORT=0.10,
W_VERIFICATION_PASS=0.10, W_RETRIEVAL_F1=0.20, W_EVIDENCE_UTILIZATION=0.15`.

**Effect:** the redesign did what it was designed to do at the *behaviour*
level — evidence count moved 2.07 → 5.28 chunks, breaking the minimal-evidence
local optimum. **But it did not improve the held-out objective.** This is the
core lesson: the reward and the true metric (Token F1 / answer quality) are only
loosely coupled. Optimising the proxy harder changed the policy's behaviour
without improving its answers.

---

## 6. Head-to-head — trained v4 (`ep_0200.pt`) vs the baseline

Checkpoint: `brain/maddpg/results/maddpg_v4/checkpoints/ep_0200.pt`
(state_dim 20, CEB, **392 gradient updates** — verified trained).

| Metric | MADDPG v4 (trained) | `simple_hybrid_rag` | Winner |
|---|---:|---:|---|
| Token F1 | 0.434 ± 0.031 | 0.487 ± 0.025 | **baseline** (≈1.3 SE) |
| ROUGE-L | 0.310 | 0.367 | **baseline** |
| Verification pass rate | 0.97 | 1.00 | baseline |
| Failure rate | 0.03 | 0.00 | baseline |
| Mean latency | 22.7 s | 1.7 s | **baseline** |
| Mean LLM calls | 2.69 | 1.0 | **baseline** |
| Mean evidence count | 5.28 | 8.0 | — |

The baseline is ahead or tied on every metric. The MADDPG controller's only
distinctive behaviour is *adaptivity* — it varies parameters per query — but
that adaptivity does not convert into a quality gain here.

### Per-category Token F1 (trained v4 vs baseline)

| Category | MADDPG v4 (trained) | baseline | n | Winner |
|---|---:|---:|---:|---|
| multi_chunk_synthesis | 0.463 | 0.473 | 7 | tie |
| definition_explanation | **0.530** | 0.494 | 6 | **MADDPG** |
| paraphrase_hard_retrieval | 0.367 | 0.375 | 6 | tie |
| cross_paper_comparison | 0.378 | 0.496 | 4 | baseline |
| figure_table_diagram_grounded | 0.190 | 0.653 | 2 | baseline |
| direct_fact_lookup | 0.543 | 0.608 | 2 | baseline |
| intra_paper_comparison | 0.404 | 0.442 | 1 | baseline |
| distractor_edge_case | 0.571 | 0.667 | 1 | baseline |

The trained controller wins exactly one category (`definition_explanation`,
n=6), ties two, and loses the rest. `figure_table_diagram_grounded` collapses to
0.190. The earlier draft's claim that "MADDPG wins `multi_chunk_synthesis`" does
not survive the corrected checkpoint — it is now a tie (0.463 vs 0.473).

---

## 7. The 200-vs-500-episode comparison

Both v4 runs, evaluated on the same test set:

| | v4 200 ep (`ep_0200.pt`) | v4 500 ep (`best_reward.pt`) |
|---|---:|---:|
| Gradient updates | 392 | 986 |
| Token F1 | 0.434 | 0.419 |
| Evidence count | 5.28 | 5.55 |

More training (392 → 986 updates) is associated with a small drop in held-out
Token F1 (0.434 → 0.419), within ≈1 SE — consistent with mild proxy-reward
over-optimisation, but **not** the clean "early stopping wins" story the earlier
draft told. With the curve in §3, the honest summary is: training does not help,
and pushing it further does not obviously help either. There is no "optimal
budget" to claim — the policy simply does not learn to outperform its
initialisation on this objective.

---

## 8. Honest limitations

- **The trained controller does not reach parity.** It scores below every
  working baseline and no better than its own untrained initialisation. The
  honest claim is a *negative result*, not parity and certainly not superiority.
- **The reward is a proxy, and it does not transfer.** This is the central
  finding. Token F1 / embedding similarity reward shaped the policy's behaviour
  (evidence count) without improving answer quality. An LLM-as-judge reward is
  the recommended next step.
- **The controller is slow and occasionally fails** (22.7 s; 1/29 failures) vs
  `simple_hybrid_rag` (1.7 s; 0 failures).
- **Small test set (n=29).** Per-category n is as low as 1. Differences of
  ≈1 SE are not statistically conclusive — which is also why the project does
  *not* claim training significantly *hurt*, only that it did not help.
- **The benchmark was generated by an LLM** (validated, but not human-authored).
- **Discrete action selection is non-differentiable** — the actor learns
  continuous parameters; the discrete action is selected by a fixed mapper.

---

## 9. Defensible claims for the thesis

1. **The stage-conditioned MADDPG-style architecture trains stably end-to-end on
   live LLM calls** — 392–986 gradient updates, 0 rate-limit failures, clean
   convergence of critic and actor losses. The engineering works.

2. **Training the controller on a hand-designed proxy reward did not improve
   held-out answer quality.** Across 0 → 986 gradient updates, Token F1 stayed
   flat-to-slightly-declining (0.474 → 0.434 → 0.419) and never exceeded the
   untrained policy. This is a clean, well-instrumented negative result about
   proxy-reward RL for RAG control.

3. **The iterative diagnosis is itself a contribution** — the project identifies
   and fixes a reward-hacking failure mode (minimal-evidence collapse), and then
   shows that even after the fix the proxy reward does not transfer. It also
   documents a checkpoint-selection pitfall (`best_reward.pt` on noisy
   single-episode training reward) that silently produced an untrained policy.

4. **What would be needed to push past the baselines** (future work): an
   LLM-as-judge reward to align the training signal with true answer quality, a
   larger human-authored benchmark to reduce variance, and differentiable
   discrete action selection (e.g. Gumbel-softmax).

---

## 10. Artifact index

| Artifact | Path |
|---|---|
| Trained checkpoint (v4, 200 ep, **recommended**) | `brain/maddpg/results/maddpg_v4/checkpoints/ep_0200.pt` |
| v4 intermediate checkpoints | `brain/maddpg/results/maddpg_v4/checkpoints/ep_{0050,0100,0150}.pt` |
| v4 training aggregate | `brain/maddpg/results/maddpg_v4/aggregate_metrics_maddpg_ceb_openai_200ep_v3.json` |
| Trained v4 eval (29-q test) | `brain/maddpg/results/eval_v4_trained/eval_maddpg_ceb.jsonl` |
| v4 training-curve evals | `brain/maddpg/results/eval_v4_ep{0050,0100,0150}/` |
| Untrained `best_reward.pt` eval (for the curve) | `brain/maddpg/results/eval_v4/eval_maddpg_ceb.jsonl` |
| v4-500ep eval | `brain/maddpg/results/eval_v4_500ep/eval_maddpg_ceb.jsonl` |
| `simple_hybrid_rag` eval | `brain/maddpg/results/eval_expanded/eval_simple_hybrid_rag.jsonl` |
| Expanded benchmark | `brain/maddpg/results/benchmark_splits/{train,val,test}.jsonl` |
| Old benchmark backup | `brain/maddpg/results/benchmark_splits_backup_v1/` |
