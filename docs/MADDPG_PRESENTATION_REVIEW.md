# MADDPG Architecture — Presentation Review

A self-contained study guide for presenting the reinforcement-learning controller
of this project. Everything here is what you need to explain it, defend it, and
answer questions. Read top to bottom once; the last two sections (anticipated
questions, glossary) are for the night before.

---

## 0. The 30-second pitch

> "I built a reinforcement-learning controller that *learns* how to configure a
> retrieval-augmented-generation pipeline, instead of configuring it by hand.
> It is a **stage-conditioned, MADDPG-style continuous-control policy**: each of
> the five RAG stages has its own neural-network 'actor' that learns the numeric
> parameters for that stage — how many chunks to retrieve, how strict to grade,
> what temperature to generate at, and so on. It is trained off-policy against
> the live pipeline with a centralized critic. The honest finding is a
> well-diagnosed **negative result**: the controller trains stably, but the
> hand-designed proxy reward does not transfer to held-out answer quality — it
> reaches parity with, and does not beat, a strong fixed baseline."

If you remember nothing else: **per-stage learned parameter control; trains
cleanly; does not beat the baseline; the contribution is the architecture + the
honest diagnosis, not a win.**

---

## 1. The problem — why reinforcement learning for RAG

A RAG pipeline has many **numeric knobs**: retrieval breadth (`top_k`), dense vs
sparse weighting, rerank threshold, grading strictness, how much graded evidence
to keep, generation temperature, citation strictness, verification threshold.

In every other architecture in this project these knobs are **fixed by hand** —
"retrieve top-8, keep top-5, one rewrite allowed." But the *optimal* setting
depends on the query:

- A **direct factual lookup** needs little evidence and a short answer.
- A **multi-paper synthesis** question needs broad retrieval and a long answer.

A fixed configuration cannot be optimal for both. The hypothesis: a controller
that **learns** to set these knobs *per query* should beat any single fixed
configuration.

That hypothesis is what the MADDPG controller tests. (Spoiler: the answer turned
out to be "not with this reward" — see §10. That is still a publishable result.)

---

## 2. Algorithmic background — go deep here

This is the section examiners probe. Know it cold.

### 2.1 Reinforcement learning / the MDP

RL frames a problem as a **Markov Decision Process (MDP)**: at each timestep the
agent sees a **state** `s`, takes an **action** `a`, receives a **reward** `r`,
and moves to a next state `s'`. The goal is a **policy** `π(a|s)` that maximizes
expected cumulative discounted reward `Σ γ^t r_t`. `γ` (gamma, here 0.99) is the
**discount factor** — how much future reward counts vs immediate reward.

Two function families matter:
- **Value function** `Q(s,a)` — expected return from taking `a` in `s` then
  following the policy. "How good is this action here?"
- **Policy** `π` — the thing that actually chooses actions.

**Actor-critic** methods learn both: the *actor* is the policy, the *critic* is
the value function that tells the actor how good its actions are.

### 2.2 DDPG — Deep Deterministic Policy Gradient (Lillicrap et al., 2016)

DDPG is the single-agent algorithm this project builds on. It is **off-policy,
actor-critic, for continuous action spaces**. Five ingredients:

**(a) A deterministic actor** `μ(s|θ^μ)`.
Unlike a stochastic policy that outputs a *distribution* over actions, a
deterministic actor maps a state directly to *one* action vector. This is the
right choice when actions are continuous (a temperature of 0.37, a `top_k` of
12) — you do not want to sample, you want to *output the number*.

**(b) A critic** `Q(s,a|θ^Q)` — a neural net estimating the action-value.

**(c) A replay buffer.**
Every transition `(s, a, r, s', done)` the agent experiences is stored. Training
samples **random minibatches** from this buffer. Two reasons:
- It **breaks temporal correlation** — consecutive transitions are highly
  correlated; random sampling de-correlates the gradient updates.
- It makes learning **off-policy** and **sample-efficient** — old experience is
  reused many times, not thrown away.

**(d) Target networks** `μ'`, `Q'` — slowly-updated copies of the actor and
critic. They exist to **stabilize training**. The critic is trained toward a
target that itself depends on the critic; if you used the *live* critic in that
target, you would be chasing a moving target and training would oscillate or
diverge. Slow target copies make the target quasi-stationary.

**(e) Exploration noise.**
A deterministic actor always outputs the same action for the same state — it
cannot explore on its own. So during *training* you add noise to the actor's
output. DDPG uses **Ornstein-Uhlenbeck (OU) noise** — temporally-correlated
noise that produces smooth, drifting exploration (good for control). At
*evaluation* the noise is switched off → the policy is "greedy."

**The two update rules** (this is the heart of it):

*Critic update* — fit the critic to the **temporal-difference (TD) target**:
```
y = r + γ · (1 − done) · Q'(s', μ'(s'))
L_critic = mean( ( y − Q(s, a) )² )          ← mean-squared TD error
```
The critic is doing regression toward "immediate reward + discounted value of
where you landed."

*Actor update* — the **deterministic policy gradient**. The actor's job is to
output the action the critic values most, so you do gradient *ascent* on the
critic's value of the actor's action:
```
∇_θμ J ≈ mean( ∇_a Q(s, a)|_{a=μ(s)} · ∇_θμ μ(s) )
```
In code this is just: `actor_loss = −Q(s, μ(s))`, then minimize it. (That is why
"actor loss" is negative and why it is *not* a normal loss — see §10.)

*Soft (Polyak) update* — after every step, nudge the targets toward the live
networks:
```
θ' ← τ·θ + (1 − τ)·θ'         with τ small (here 0.005)
```

### 2.3 MADDPG — Multi-Agent DDPG (Lowe et al., 2017)

MADDPG extends DDPG to **multiple agents**. The key problem it solves:
**non-stationarity**. If several agents learn at once, then from any one agent's
viewpoint the environment keeps changing (because the *other* agents' policies
change). A normal single-agent critic would be chasing a target that moves for
reasons it cannot see.

MADDPG's fix: **a centralized critic**. Each agent keeps its own actor, but the
critic is trained with access to the **joint** state and the **joint actions of
all agents**. Because the critic conditions on what everyone did, the
environment looks stationary *to the critic* — and a stable critic gives the
actors a stable learning signal.

### 2.4 CTDE — Centralized Training, Decentralized Execution

This is MADDPG's paradigm and a phrase examiners love:

- **Centralized training:** the critic — used *only* during training — may see
  more than any single actor (the joint state/actions).
- **Decentralized execution:** at deployment the critic is thrown away; each
  actor runs **independently**, using only its own observation. No agent needs
  to see the others to act.

You get the stability benefit of global information during training without
requiring global information at run time.

---

## 3. The RAG pipeline as a reinforcement-learning environment

To apply MADDPG you must cast the RAG pipeline as an MDP. This is the modeling
contribution — be ready to explain every choice.

### 3.1 The five agents (one per RAG stage)

| Agent | RAG stage | What it does |
|---|---|---|
| `retriever` | retrieval | hybrid dense+sparse retrieval from Qdrant |
| `rewriter` | query rewrite | reformulate the query on a recovery branch |
| `grader` | evidence grading | score and filter retrieved chunks |
| `generator` | answer generation | produce the answer from kept evidence |
| `verifier` | claim verification | check the answer's claims against evidence |

### 3.2 Stage-gating — only one agent acts per step

The RAG workflow is **causal and sequential**: the grader cannot run before
retrieval; the verifier cannot run before generation. So at every timestep
**exactly one agent is "active,"** decided by **action-masking rules** (the
`stage_utils` / `action_masking` logic). The episode flows
`retriever → grader → generator → verifier`, with recovery branches
(re-retrieve, rewrite, regenerate) when verification fails.

### 3.3 The state

A fixed-length numeric feature vector describing the current condition of the
pipeline: query type/complexity, retrieval confidence, number of retrieved and
graded chunks, citation-support rate, count of unsupported claims, verification
status, retry count, latency, step counts.

Two encodings:
- **Base state — 14 dimensions** (`FEATURE_DIM = 14`).
- **CEB state — 20 dimensions** — the base 14 plus 6 **Context Engineering
  Block** features (see §7).

### 3.4 The actions — continuous parameters

Each agent's actor outputs a continuous vector in `[−1, 1]` (a `tanh` output
layer). A deterministic **mapper** converts that vector into (i) the numeric RAG
parameters for the stage and (ii) a discrete action *name* chosen from the
agent's currently legal (masked) action set.

There are **14 continuous action dimensions** in total:

| Agent | Continuous parameters |
|---|---|
| Retriever | dense/sparse weight, `top_k`, rerank threshold, source diversity |
| Rewriter | rewrite strength, query-expansion weight |
| Grader | relevance threshold, evidence-keep ratio, strictness |
| Generator | temperature, citation strictness, `max_tokens`, detail level |
| Verifier | support threshold |

**Important nuance for questions:** the actor learns the *continuous numbers*.
The mapping from those numbers to a discrete action name (e.g. "use
`hybrid_rerank` when the rerank parameter is high") is a **fixed,
non-differentiable function**. The discrete choice is *derived*, not learned —
this is a known limitation (see §11).

### 3.5 The transition and the reward

One `env.step()` = one agent acts: the active agent executes, the shared context
state is mutated, a scalar reward is returned. A transition records
`(state, active_agent, action, reward, next_state, next_active_agent, done)`.

The reward is a **single, shared, cooperative scalar** — all five agents receive
the *same* reward, computed mostly at episode end (see §6). This is a fully
**cooperative** multi-agent setting: the agents are a team optimizing one
objective (a good grounded answer), not competing.

---

## 4. The architecture as built — "stage-conditioned MADDPG-style"

Call it **"MADDPG-style"**, not "MADDPG" — and be ready to say *why*. It keeps
MADDPG's core (per-agent deterministic actors, a centralized critic, off-policy
replay, target networks, OU exploration) but **deviates in three ways**, each
forced by the stage-gated RAG environment.

### Deviation 1 — sequential, not simultaneous, agents

Textbook MADDPG assumes **all** agents act every timestep. The RAG pipeline is
causal, so **exactly one** agent acts per step. The episode is a sequence of
single-agent decisions, not a vector of simultaneous ones.

### Deviation 2 — a stage-conditioned critic, not a joint-action critic

Textbook MADDPG's critic takes the concatenated joint action of *all* agents. If
you did that here, then at every step 4 of the 5 action slots would be zero
(only one agent acted) — the critic would train mostly on zero-padding and the
gradient signal would be diluted.

Instead there is **one shared centralized critic** conditioned on **which agent
is acting**:
```
Q( state, active-agent one-hot, discrete-action one-hot, padded continuous action )
```
- `active-agent one-hot` (5-dim) — which of the 5 stages is acting.
- `discrete-action one-hot` — which discrete action it took.
- `padded continuous action` — the continuous vector, zero-padded to a uniform
  width so the critic always sees a fixed-shape input regardless of which agent
  acted.

Every training sample is informative; the input shape is constant.

### Deviation 3 — shared global observation, not per-agent local observations

Textbook MADDPG gives each actor a *private* local observation. Here every actor
receives the **same global state**. Justification: the agents run sequentially
in one process, so there is no decentralized-deployment requirement that forces
local-only observations, and shared cross-stage context (e.g. "an earlier
verification attempt failed") is genuinely useful to every stage.

### The precise one-liner

> "A sequential-agent, parameterized-action, off-policy actor-critic with a
> stage-conditioned centralized critic — adapted from MADDPG/DDPG."

It still honors **CTDE**: the critic is used only in training; at evaluation
each actor runs alone with no critic and no inter-agent communication.

### The networks

- **Actors (×5):** a multilayer perceptron — **LayerNorm → ReLU → ... → `tanh`
  output**. Deterministic. Each agent also has a slow **target actor** copy.
  Hidden dimension 128.
- **Critic (×1, shared):** an MLP estimating the `Q(...)` above. Has a **target
  critic** copy. Hidden dimension 128.
- Exploration during training: **Ornstein-Uhlenbeck noise** (`σ = 0.15`) added
  to actor outputs; off at evaluation (greedy).

---

## 5. The training algorithm, step by step

Training is **off-policy**, against the **live** RAG pipeline (real Qdrant
retrieval, real `gpt-4o-mini` calls).

1. **Collect.** Run an episode: at each step the active agent's actor produces an
   action (+ OU noise), the env executes, the transition is stored in the
   **replay buffer**.
2. **Warm up.** For the first `WARMUP_STEPS` (~50) transitions, only collect — do
   not update — so the buffer has enough diverse data.
3. **Update** (every few env steps, on a sampled minibatch of 64):
   - **Critic update:** compute the TD target
     `y = r + γ·(1−done)·Q'(s', next stage's target action)` and minimize
     mean-squared error `(y − Q(s,a))²`. (The "next stage's action" comes from
     the *target actor* of whichever agent acts next.)
   - **Actor update — once per agent:** filter the minibatch to transitions
     where that agent was active, run its actor on those states, and do gradient
     ascent on the critic's value of the resulting action
     (`actor_loss = −Q(...)`). The discrete-action context is held fixed (it is
     not differentiable).
   - **Soft-update** all target networks: `θ' ← τθ + (1−τ)θ'`, `τ = 0.005`.
4. **Robustness:** if an env step throws (e.g. transient API error), a terminal
   transition with a negative reward (`error_penalty = −1.0`) is written so
   training continues and the failure still produces a learning signal.

Key hyperparameters to have memorized: `γ = 0.99`, `τ = 0.005`,
actor/critic LR `1e-3`, batch size 64, OU `σ = 0.15`, gradient clip 1.0,
200 training episodes, checkpoint every 50.

---

## 6. The reward function (and the reward-hacking story)

The reward is the **most consequential and most error-prone** part of the
design. Examiners *will* ask about it.

### The v4 reward (the final one)

A weighted sum, computed mostly at episode termination:

| Term | Weight | What it measures |
|---|---|---|
| Answer quality | **0.45** | `0.5 × embedding-similarity + 0.5 × token-F1` vs the gold answer |
| Retrieval F1 | 0.20 | did retrieval fetch the correct source documents |
| Evidence utilization | 0.15 | `min(evidence_count, 6) / 6` — rewards using a substantial evidence pack |
| Citation support | 0.10 | fraction of answer claims supported by evidence |
| Verification pass | 0.10 | did the answer pass claim verification |

Plus small negative terms for latency and step count, and fixed penalties for an
empty answer, unsupported claims, hallucination, and timeout.

### The reward-hacking story — tell this, it is a real contribution

Earlier reward versions produced a consistent **failure mode**: the trained
policy learned to **minimize evidence use** — it answered with only ~1–2 chunks.
Why? A short answer makes few claims, and few claims are trivially easy for the
verifier to support — so the policy *maximized the reward* (verification +
citation terms) while producing **thin, low-quality answers**.

This is textbook **reward hacking** / proxy-reward misalignment: *the policy was
optimizing the reward correctly; the reward was the problem.* Tightening
parameter ranges did not fix it — the policy just found a different lever to
strip evidence.

The **v4 fix**: (1) replace pure token-F1 with a semantic blend, and (2) add the
explicit **evidence-utilization** term so the 2-chunk strategy *loses* reward
outright (capped at 6 so it cannot be gamed by over-retrieval). Effect: evidence
use roughly doubled (~2 → ~5 chunks). The behavior was fixed — but, crucially,
fixing the *behavior* did **not** improve held-out answer quality (§10).

---

## 7. The Context Engineering Block (CEB)

The **CEB** is the 20-dimensional state encoding: the 14 base features plus 6
extra cross-stage features — source diversity, evidence coverage, step-budget
fraction, LLM-call-budget fraction, query length, and a multi-source flag.

Its purpose: give the controller **cross-stage context** the base 14-dim state
misses (e.g. how much of the step budget is spent, whether the query needs
multiple sources).

**The honest result on CEB:** an ablation (CEB vs no-CEB, both trained 200
episodes) found **no measurable benefit**. On the 60-question held-out set,
Token F1 was 0.416 (CEB) vs 0.421 (no-CEB) — a statistical tie (`p = 0.64`).
The one place CEB helped was the *critic's* training loss (it converged 2–3×
lower), but that better value estimate did **not** produce a better policy. So:
**CEB helps value estimation, not answer quality.**

---

## 8. Results — the honest story

Be honest and confident; a well-diagnosed negative result is a strong thesis.

### 8.1 The checkpoint-selection bug — disclose this proactively

Training saves `best_reward.pt` at the **highest single-episode training
reward** — a noisy quantity. For the 200-episode run it landed on an early
episode with **0 gradient updates** — an *untrained* policy. Early result drafts
unknowingly evaluated this untrained checkpoint. The fix: evaluate the
**periodic final checkpoint** `ep_0200.pt` (392 gradient updates). All current
numbers use the genuinely-trained checkpoint.

### 8.2 The headline numbers (60-question held-out set, Token F1)

| System | Token F1 |
|---|---:|
| `self_rag_grader` / `simple_hybrid_rag` / `final_arch` (fixed baselines) | ~0.45 – 0.49 |
| **MADDPG v4, trained (CEB, `ep_0200`, 3-run mean)** | **~0.416** |
| MADDPG v4, trained (no-CEB) | ~0.421 |
| MADDPG, *untrained* random policy | ~0.47 |

Three things this says:
1. The trained controller scores **below the fixed baselines**.
2. It scores **no higher than its own untrained initialization** — training did
   not raise held-out quality.
3. The differences are small and **not statistically significant** after
   correcting for multiple comparisons — best read as "**statistically
   indistinguishable**, with MADDPG trending lowest."

### 8.3 The training curves — the mechanism

- **Critic loss converges cleanly** (≈0.19 → 0.003). The training *infrastructure
  works*.
- **Episode reward is flat** (slope ≈ 0). The policy does *not* learn to earn
  more reward.
- Why the contradiction: the proxy reward is **near-saturated from episode 1**
  (verification ≈ 1.0, citation ≈ 0.9) — there is no headroom for the actors to
  climb. The critic learns to *predict* the reward; the actors cannot *raise* it.
- One real defect: the **`rewriter` actor never trained** (`nan` loss) — the
  rewrite stage was never triggered during training, so that actor stayed at
  random initialization.

### 8.4 Cross-benchmark consistency

Across **four** benchmarks — academic-QA, ARC-Challenge, HotpotQA, 2Wiki — the
pattern repeats: the **simple hybrid baseline is at or near the top, and the
learned MADDPG controller never beats it.** Consistency across benchmarks is
what makes the conclusion defensible.

---

## 9. What to claim — the defensible contributions

1. **A working stage-conditioned MADDPG-style architecture** that trains stably
   end-to-end on live LLM calls — clean critic convergence, no divergence, no
   rate-limit collapse. The engineering and the adaptation are real.
2. **A clean, well-instrumented negative result:** training a continuous-control
   RL policy on a hand-designed proxy reward does **not** improve held-out
   answer quality over a strong fixed pipeline. The proxy reward does not
   transfer to the true objective.
3. **The iterative diagnosis is itself a contribution** — identifying and fixing
   a reward-hacking failure mode (minimal-evidence collapse), and documenting a
   checkpoint-selection pitfall that silently produced an untrained policy.
4. **An honest ablation** showing the Context Engineering Block adds no
   measurable policy gain.

Do **not** claim the controller beats or matches the baselines, or that CEB
helps. The strength of the work is its rigor and honesty.

---

## 10. Limitations (state them before you are asked)

- The reward is a **proxy** (embedding similarity + token-F1), not a measure of
  correctness — and it does not transfer. An **LLM-as-judge** reward is the
  recommended fix.
- **Discrete action selection is non-differentiable** — the actor learns only
  the continuous parameters; the discrete choice comes from a fixed mapper. A
  Gumbel-softmax relaxation would let gradients flow through it.
- **Small benchmark** (60 held-out questions; LLM-generated, validated but not
  human-authored) → wide confidence intervals.
- The **`rewriter` agent never trained** — part of the policy is untrained.
- The controller is **slower** than the lightweight baseline (~25 s vs ~2 s) for
  no quality gain.

---

## 11. Anticipated questions — defense prep

**Q: Why is it "MADDPG-style" and not MADDPG?**
A: Three deviations forced by the stage-gated environment — sequential (not
simultaneous) agents, a stage-conditioned (not joint-action) critic, and a
shared global (not per-agent local) observation. It keeps the MADDPG core:
per-agent actors, centralized critic, off-policy replay, target nets, OU noise.

**Q: Is it centralized or decentralized?**
A: CTDE — centralized training (the critic sees the global state + the active
agent + its action), decentralized execution (at eval each actor runs alone, no
critic, no communication).

**Q: Discrete or continuous actions?**
A: Both, in a specific way. The actor learns **continuous** parameters; a fixed
non-differentiable mapper turns them into a **discrete** action name. So it is a
*parameterized-action* policy. The continuous part is learned, the discrete part
is derived.

**Q: Why a deterministic policy?**
A: The quantities being controlled (temperature, `top_k`, thresholds) are
continuous values you want to *output*, not sample. DDPG's deterministic actor
fits continuous control; exploration is handled by added OU noise during
training.

**Q: Why a replay buffer / what does off-policy mean?**
A: Transitions are stored and reused in random minibatches — this de-correlates
updates and makes learning sample-efficient. Off-policy = it learns from
experience collected by an older version of the policy, not only the current one.

**Q: What do the target networks do?**
A: They stabilize training. The critic's TD target depends on the critic itself;
using slow target copies keeps that target quasi-stationary so training does not
oscillate.

**Q: Your reward graph is flat — did training fail?**
A: No — training *worked* (the critic converged cleanly). The reward is flat
because the proxy reward was near-saturated from the start; there was no
headroom. That is the finding: the bottleneck is the reward, not the optimizer.

**Q: So the project failed?**
A: No. The hypothesis — that learned per-query control beats fixed
configuration — was tested rigorously and **not supported**. That is a valid,
publishable negative result, plus a working architecture and a documented
reward-hacking diagnosis. A negative result honestly reported is a contribution.

**Q: Why does the controller not beat the baseline?**
A: The proxy reward (embedding similarity + token-F1) is only loosely coupled to
true answer quality, and it is near-saturated — so optimizing it harder changes
the policy's *behavior* (evidence count) without improving its *answers*.

**Q: What would make it work?**
A: An LLM-as-judge reward aligned with true quality; a larger human-authored
benchmark; and a differentiable discrete-action relaxation (Gumbel-softmax).

**Q: Why MADDPG and not single-agent RL or PPO?**
A: The five stages have distinct action spaces and roles → natural multi-agent
decomposition. The actions are continuous → the deterministic-policy family
(DDPG/MADDPG) fits better than discrete-action methods; off-policy replay is far
more sample-efficient than on-policy PPO, which matters when every sample costs
a live LLM call.

---

## 12. Glossary — one-liners

- **MDP** — state → action → reward → next state; maximize discounted return.
- **Actor** — the policy network; state → action.
- **Critic** — value network; estimates `Q(s,a)`, how good an action is.
- **DDPG** — off-policy actor-critic for continuous actions; deterministic actor.
- **MADDPG** — multi-agent DDPG; per-agent actors + a centralized critic.
- **CTDE** — centralized training, decentralized execution.
- **Replay buffer** — stored past transitions, sampled randomly for updates.
- **Target network** — slow copy of actor/critic for stable TD targets.
- **TD target** — `r + γ·Q'(s', μ'(s'))`; what the critic regresses toward.
- **Soft update** — `θ' ← τθ + (1−τ)θ'`; slow blending of target nets.
- **OU noise** — temporally-correlated exploration noise on the actor output.
- **γ (gamma)** — discount factor (0.99). **τ (tau)** — soft-update rate (0.005).
- **Stage-gating** — exactly one agent is legal to act per step.
- **CEB** — Context Engineering Block; the 20-dim state with 6 extra features.
- **Reward hacking** — policy maximizes the stated reward in a way that defeats
  the intended objective.
- **Proxy reward** — a measurable stand-in (token-F1) for the true goal
  (a correct answer); they can diverge.
