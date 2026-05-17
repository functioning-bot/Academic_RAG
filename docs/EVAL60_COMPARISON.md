# 4-Model Comparison on the 60-Question Held-Out Set

A controlled comparison of four RAG systems on a single 60-question evaluation
set, including a Context Engineering Block (CEB) ablation of the MADDPG
controller. Read alongside `RESULTS.md`.

---

## 1. Setup

- **Evaluation set:** `brain/maddpg/results/benchmark_splits/eval60.jsonl` — 60
  questions, **all held out from MADDPG training**. Drawn from the 72 questions
  the controller never trained on (val 28 + test 29 + 15 validated candidates
  never assigned to a split), stratified by category (built by
  `brain/maddpg/build_eval60.py`, seed 42). Difficulty mix: 26 hard / 29 medium
  / 5 easy.
- **Same** corpus (12 papers, ≈1,226 chunks), **same** hybrid retrieval backbone,
  **same** LLM (`gpt-4o-mini`) for every system.
- **Metrics** computed with one shared function set for all four systems
  (set-overlap Token F1; LCS-based ROUGE-L), so the numbers are directly
  comparable.
- **Checkpoints:** both MADDPG variants are the **periodic final** checkpoint
  `ep_0200.pt` (verified trained — never `best_reward.pt`; see `RESULTS.md` §2.1).

The two MADDPG variants:
- **MADDPG v4 CEB** — 20-dim state (Context Engineering Block: adds source
  diversity, evidence coverage, budget fractions, query length, multi-source
  flag). Checkpoint `maddpg_v4/checkpoints/ep_0200.pt`, 392 gradient updates.
- **MADDPG v4 no-CEB** — 14-dim base state. Trained fresh for this comparison,
  identical v4 reward and config, 200 episodes. Checkpoint
  `maddpg_v4_noceb/checkpoints/ep_0200.pt`, 233 gradient updates.

---

## 2. Headline results (n = 60)

| System | Token F1 | ROUGE-L | Pass rate | Failures | Mean latency | Mean evidence |
|---|---:|---:|---:|---:|---:|---:|
| `simple_hybrid_rag` | **0.460 ± 0.020** | 0.335 | 1.00 | 0 / 60 | 1.9 s | 8.0 |
| `final_arch` | 0.449 ± 0.020 | 0.325 | 1.00 | 0 / 60 | 36.8 s | 4.6 |
| MADDPG v4 **no-CEB** | 0.426 ± 0.016 | 0.295 | 0.98 | 0 / 60 | 28.1 s | 5.7 |
| MADDPG v4 **CEB** | 0.418 ± 0.021 | 0.279 | 0.95 | 0 / 60 | 26.8 s | 5.7 |

**Reading it:**

1. **The two fixed baselines lead; both learned controllers trail.** The best
   MADDPG variant (no-CEB, 0.426) is 0.034 below `simple_hybrid_rag` (0.460) —
   roughly 1.3 standard errors. Not formally significant at the 60-question
   scale, but the direction is consistent with the corrected `RESULTS.md`
   finding: training the controller on the v4 proxy reward does not lift answer
   quality above a strong fixed pipeline.

2. **The Context Engineering Block did not help — it slightly hurt.** no-CEB
   (0.426) edges out CEB (0.418). The 0.008 gap is well inside one standard
   error, so the honest statement is a **tie**: the six extra cross-stage
   features bought no measurable improvement. The no-CEB model is in fact
   marginally *more* stable — lower standard error (0.016 vs 0.021) and higher
   verification pass rate (0.98 vs 0.95). This is a clean negative ablation:
   the added state representation was not worth its complexity here.

3. **`simple_hybrid_rag` remains the efficiency winner** — 1.9 s/query against
   27–37 s for every other system, and it never fails. `final_arch` is the
   slowest by far (36.8 s) for no quality gain over the simple baseline.

---

## 3. Per-category Token F1

| Category | n | `simple_hybrid_rag` | `final_arch` | MADDPG CEB | MADDPG no-CEB |
|---|--:|---:|---:|---:|---:|
| definition_explanation | 14 | 0.411 | 0.401 | **0.432** | **0.432** |
| cross_paper_comparison | 13 | **0.430** | 0.417 | 0.407 | 0.403 |
| multi_chunk_synthesis | 11 | **0.476** | 0.447 | 0.445 | 0.422 |
| paraphrase_hard_retrieval | 10 | 0.402 | 0.415 | 0.329 | **0.424** |
| direct_fact_lookup | 4 | **0.697** | 0.687 | 0.665 | 0.490 |
| figure_table_diagram_grounded | 3 | 0.575 | **0.600** | 0.282 | 0.437 |
| intra_paper_comparison | 3 | **0.415** | 0.372 | 0.346 | 0.350 |
| distractor_edge_case | 2 | **0.626** | 0.602 | 0.501 | 0.530 |

Observations (category n is small — these are indicative, not conclusive):

- **The only category either MADDPG variant wins is `definition_explanation`**
  (0.432 vs 0.411 baseline) — and both variants win it identically.
- **CEB collapses on `figure_table_diagram_grounded` (0.282) and
  `paraphrase_hard_retrieval` (0.329).** The no-CEB model is more balanced on
  exactly those two (0.437, 0.424) — the main reason its aggregate is higher.
- **no-CEB collapses on `direct_fact_lookup` (0.490 vs ~0.69 for everything
  else).** The CEB model handles short factual lookups much better. So the two
  variants trade categories rather than one dominating.

---

## 4. The CEB ablation — interpretation

The Context Engineering Block was introduced to give the controller cross-stage
context (e.g. whether an earlier verification attempt failed). On this 60-question
held-out set it produced **no aggregate benefit**: 0.418 with CEB vs 0.426
without, a within-noise tie, with the no-CEB model slightly more stable.

Caveat — this is not a perfectly controlled ablation. Both runs used 200
episodes of the identical v4 reward and configuration, but episode lengths
differ, so the CEB run accumulated 392 gradient updates and the no-CEB run 233.
The comparison is therefore "same training budget (200 episodes)", not "same
number of gradient updates". Even so, the more-updated CEB model did not come
out ahead, which makes the no-help conclusion if anything more robust.

The practical takeaway: the richer 20-dim state did not earn its place. If the
MADDPG controller is carried forward, the 14-dim base state is the simpler and
equally-good (marginally better) default.

---

## 5. Artifact index

| Artifact | Path |
|---|---|
| 60-question eval set | `brain/maddpg/results/benchmark_splits/eval60.jsonl` |
| Eval-set builder | `brain/maddpg/build_eval60.py` |
| `simple_hybrid_rag` eval | `brain/maddpg/results/eval60_compare/eval_simple_hybrid_rag.jsonl` |
| `final_arch` eval | `brain/maddpg/results/eval60_compare/eval_final_arch.jsonl` |
| MADDPG CEB eval | `brain/maddpg/results/eval60_compare/maddpg_ceb/eval_maddpg_ceb.jsonl` |
| MADDPG no-CEB eval | `brain/maddpg/results/eval60_compare/maddpg_no_ceb/eval_maddpg_no_ceb.jsonl` |
| MADDPG CEB checkpoint | `brain/maddpg/results/maddpg_v4/checkpoints/ep_0200.pt` |
| MADDPG no-CEB checkpoint | `brain/maddpg/results/maddpg_v4_noceb/checkpoints/ep_0200.pt` |
| no-CEB training log | `brain/maddpg/results/train_noceb.log` |
