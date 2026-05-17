# Thesis Update — Reinforcement Learning Controller (MADDPG)

Draft thesis sections integrating the implemented reinforcement-learning
controller. Written in the report's academic style, ready to adapt into the
document. Read together with `RESULTS.md`, `PROJECT_REVIEW.md`, and
`MADDPG_EXTENSION.md`, which contain the underlying detail.

---

## How to integrate this (read first)

The original report (Chapters 5–8) describes reinforcement learning as a
*planned direction* — Section 4.8.6 and Section 2.8 present it conceptually, and
Table 5.1 compares six fixed/agentic architectures on the original 60-question
benchmark. The RL controller has since been **implemented and evaluated**. This
document supplies the sections that change.

**Three facts the report must state honestly:**

1. **The benchmark differs.** The RL controller required a larger training set,
   so the corpus was expanded from 6 to 12 papers and the benchmark from 60 to
   145 questions. The RL evaluation therefore uses a *different* benchmark than
   Table 5.1, and a different generation model (OpenAI `gpt-4o-mini`). The two
   result sets are **not directly comparable**. Per the chosen structure,
   Table 5.1 is kept as the original study, and the RL controller is presented
   as a **separate study on the expanded benchmark** (new Section 5.7).

2. **The implemented controller differs from the one originally envisioned.**
   Section 4.8.6 envisioned an *offline* controller, trained from logged
   trajectories, that learns *discrete orchestration* (which agent to run next,
   whether to retrieve again). The implemented controller is an *online*,
   *continuous-control* controller: each agent's RAG parameters are tuned by a
   deterministic actor, trained off-policy against the live environment. The
   new Section 4.12 describes what was actually built and explains the
   evolution.

3. **The trained controller is evaluated on a verified-trained checkpoint.**
   Training saves a `best_reward.pt` checkpoint at the highest single-episode
   *training reward*, which is noisy; for the 200-episode run it landed on an
   early episode with **0 gradient updates** — an untrained policy. All RL
   numbers below are from the periodic final checkpoint `ep_0200.pt`
   (392 gradient updates), the genuinely trained policy.

The honest headline result: the **trained** RL controller scores **below** the
fixed architectures on answer quality (Token F1 0.434 against 0.482–0.490), and
**no better than the untrained policy it started from** (0.474). The report's
contribution is not a controller that wins or matches the baselines — it does
neither — but (a) a working stage-conditioned MADDPG-style architecture that
trains stably on live LLM calls, and (b) a well-diagnosed *negative result*:
training on a hand-designed proxy reward did not transfer to held-out answer
quality. The report should present this honestly as a negative/cautionary
result, not as parity and not as a win. See Section D.

---

# A. Revised Abstract (final paragraph)

> Replace the sentence *"It also evaluates supervisor, multi agent, and
> reinforcement learning based designs…"* and the closing sentence with:

The project compares a simple hybrid baseline with more advanced architectures
that add reranking, retrieval sufficiency checks, query rewriting, context
selection, document grading, answer generation, and claim-level auditing. It
further implements a reinforcement-learning controller — a stage-conditioned,
MADDPG-style continuous-control policy that learns to tune the parameters of
each retrieval-augmented-generation stage rather than relying on fixed
configuration. To support reinforcement-learning training, the document corpus
and benchmark were expanded, and the controller was evaluated against the fixed
architectures on this expanded benchmark. The system is evaluated using metrics
such as correctness, faithfulness, context precision and recall, token F1,
ROUGE-L, source hit rate, latency, and failure rate. The results show that
training the learned controller on a hand-designed proxy reward did not improve
held-out answer quality: the trained controller scores below the strongest
fixed pipelines and no better than its untrained initialisation. This negative
result, together with a diagnosed reward-hacking failure mode in which the
policy minimises evidence use, indicates that the proxy reward — a blend of
embedding similarity and lexical overlap — does not transfer to the true
objective of answer quality. Overall, this thesis shows that academic RAG
systems can become more reliable by moving beyond fixed retrieve-and-generate
pipelines toward adaptive, agentic, and verification-aware architectures, and
that, while a learned continuous-control architecture can be trained stably
end-to-end, aligning its reward with true answer quality remains the central
unsolved problem before it can rival hand-designed configuration.

---

# B. Addition to Section 2.8 (Literature — Reinforcement Learning)

> Append the following to Section 2.8, after the existing discussion of
> RL as a supervisory mechanism.

The reinforcement-learning controller implemented in this work draws on the
deterministic policy-gradient family rather than on value-based or
policy-gradient methods for discrete actions. Deep Deterministic Policy Gradient
(DDPG) trains a deterministic actor for continuous action spaces using an
off-policy critic, a replay buffer, and slowly-updated target networks. Its
multi-agent extension, Multi-Agent Deep Deterministic Policy Gradient (MADDPG),
gives each agent its own actor while training a centralised critic that can
observe the joint state and joint action. MADDPG follows the
centralised-training, decentralised-execution paradigm: the critic, used only
during training, may see more than any individual actor, while at deployment
each actor acts independently.

These methods are a natural fit for the present problem because the quantities
that most affect retrieval-augmented generation — the number of chunks to
retrieve, the grading strictness, the generation temperature, the verification
threshold — are *continuous* rather than discrete. A controller that outputs
continuous parameters can tune the pipeline more finely than one that selects
among a small set of discrete actions. The implementation described in
Section 4.12 adapts MADDPG to the stage-gated structure of the RAG environment.

> Suggested references to add: Lillicrap et al. (2016), *Continuous Control with
> Deep Reinforcement Learning*; Lowe et al. (2017), *Multi-Agent Actor-Critic
> for Mixed Cooperative-Competitive Environments*.

---

# C. New Section 4.12 — Reinforcement Learning Controller

> Insert as Section 4.12, after Section 4.11 (Multi-Agent Architecture). This
> replaces the conceptual treatment in Section 4.8.6; Section 4.8.6 may be
> shortened to a forward reference ("the reinforcement-learning layer is
> described as implemented in Section 4.12").

## 4.12 Reinforcement Learning Controller

### 4.12.1 Motivation

The supervisor and multi-agent architectures move beyond a fixed
retrieve-and-generate pipeline, but their decisions remain governed by
hand-written rules and prompt-based heuristics. A fixed rule such as "evaluate
the top five retrieved chunks, keep the top three, allow one rewrite" applies
the same policy to every query regardless of its difficulty. Optimal behaviour,
however, depends on the query: a direct factual lookup needs little evidence and
a short answer, whereas a multi-paper synthesis question needs broad retrieval
and a longer answer. Reinforcement learning offers a way to *learn* this
behaviour from experience rather than to specify it by hand.

This section describes the reinforcement-learning controller that was
implemented for the project. It is a **stage-conditioned, MADDPG-style
continuous-control controller**: each of the five agents has its own
deterministic actor network that outputs continuous parameters for that stage,
and a single centralised critic estimates the value of those parameters
conditioned on which stage is currently acting.

### 4.12.2 From Orchestration Control to Continuous Control

The reinforcement-learning layer was originally conceived (Section 4.8.6) as an
*offline* controller that learns *discrete orchestration* — which agent to run
next, whether to retrieve again, when to stop — from logged trajectories. During
implementation the design evolved in two ways.

First, the control problem was reframed from *discrete orchestration* to
*continuous parameterisation*. The stage order in the RAG workflow is already
fixed by the action-masking rules and is causally constrained (the grader cannot
run before retrieval). What is *not* fixed, and what most affects answer
quality, is the numeric configuration of each stage. The controller therefore
learns continuous parameters — retrieval breadth, grading strictness, generation
temperature, verification threshold — rather than a discrete routing policy.

Second, training was moved from offline replay of logged trajectories to
*online, off-policy* training against the live environment. The controller
interacts with the real retrieval and generation pipeline, stores each
transition in a replay buffer, and updates its networks from sampled
mini-batches. This is still off-policy — past experience is reused many times —
but the experience is collected by the controller itself rather than read from
a fixed log.

The remainder of this section describes the implemented controller.

### 4.12.3 Benchmark Expansion

Reinforcement learning requires substantially more interaction data than a
fixed-pipeline evaluation. The original 60-question benchmark, drawn from six
papers, was too small to train a controller and too narrow to exercise adaptive
behaviour — most of its questions are answerable from a single chunk, so a fixed
configuration is already near-optimal.

The corpus was therefore expanded from 6 to 12 academic papers (approximately
1,200 indexed chunks), spanning machine learning, medical, and finance domains.
The benchmark was expanded from 60 to 145 questions using an LLM-assisted
generation pipeline followed by automatic validation. Generation deliberately
targeted *synthesis-dependent* question types — multi-chunk synthesis,
cross-paper comparison, and paraphrase-hard retrieval — with longer reference
answers (a mean of roughly 50 words, against 27 in the original benchmark) so
that overlap metrics can distinguish an evidence-rich answer from a
single-chunk answer. Validation ran retrieval and an LLM judge on every
candidate, discarding questions whose gold source was not retrievable or whose
reference answer was not supported. The expanded benchmark was split, stratified
by category and difficulty, into 88 training, 28 validation, and 29 test
questions.

This expanded benchmark is used for all reinforcement-learning experiments. It
is distinct from the 60-question benchmark behind Table 5.1; the two are not
directly comparable.

### 4.12.4 Environment Formulation

The RAG workflow is formulated as a Markov Decision Process. Because the
workflow is stage-gated, exactly one of the five agents — retriever, rewriter,
grader, generator, verifier — is *active* at each step, determined by the same
action-masking rules used throughout the project. The episode proceeds
retriever → grader → generator → verifier, with the recovery branches
(regeneration, additional retrieval, query rewrite) described in earlier
sections.

The **state** is a fixed-length numeric feature vector encoding the current
condition of the workflow: query type and complexity, retrieval confidence,
number of retrieved and graded chunks, citation-support rate, count of
unsupported claims, verification status, retry count, latency, and step counts.
Two state representations are supported: a 14-dimensional base encoding, and a
20-dimensional *Context Engineering Block* encoding that adds six features
(source diversity, evidence coverage, step and LLM-call budget fractions,
query length, and a multi-source flag). The Context Engineering Block gives the
controller cross-stage context — for example, whether an earlier verification
attempt failed — that the base encoding does not capture.

A **transition** is one call to `env.step`: the active agent executes, the
shared context state is mutated, and a scalar reward is returned. Each
transition records the state, the active agent, the action taken, the reward,
the next state, the next active agent, and whether the episode terminated.

### 4.12.5 Continuous Action Space

Each agent's actor outputs a continuous vector in [−1, 1], produced by a `tanh`
output layer. A deterministic mapper converts this vector into (i) numeric RAG
parameters and (ii) a discrete action name selected from the agent's currently
valid (masked) action set. The discrete name is what the environment executes;
the numeric parameters tune *how* it executes.

| Agent | Continuous parameters |
|---|---|
| Retriever | dense/sparse weighting, `top_k` (retrieval breadth), rerank threshold, source diversity |
| Rewriter | rewrite strength, query-expansion weight |
| Grader | relevance threshold, evidence-keep ratio, strictness |
| Generator | temperature, citation strictness, `max_tokens`, answer detail level |
| Verifier | support threshold |

There are 14 continuous dimensions in total. The mapping from a continuous
vector to a discrete action name (for example, choosing `hybrid_rerank` when the
rerank parameter is high) is a fixed, non-differentiable function; the actor
learns the continuous parameters, and the discrete choice follows from them.

### 4.12.6 Actor and Critic Networks

Each of the five agents has a **deterministic actor** — a multilayer perceptron
with layer normalisation, ReLU activations, and a `tanh` output — and a
slow-moving **target actor** copy used to stabilise training. The actors are
deterministic rather than stochastic; exploration during training is provided by
Ornstein-Uhlenbeck noise added to the actor output.

The controller uses a single **stage-conditioned centralised critic**. It
estimates the action-value function

  Q( state, active-agent one-hot, discrete-action one-hot, padded continuous action )

where the continuous action is zero-padded to a uniform width so that the
critic accepts a fixed-shape input regardless of which agent is acting. A
target-critic copy is also maintained.

The critic is described as *stage-conditioned* rather than *joint-action*
because the environment is stage-gated: only one agent acts per step. A
textbook MADDPG critic over the concatenated joint action of all five agents
would, at every step, see four of five action slots as zero, diluting the
gradient signal. Conditioning the critic on the identity of the active agent and
on its action alone keeps every training sample informative and of identical
shape. Section 4.12.9 discusses this and the other deviations from textbook
MADDPG.

### 4.12.7 Training Algorithm

Training is **off-policy**. Every environment transition is stored in a replay
buffer; once the buffer has filled past a warm-up threshold, the controller
performs gradient updates on uniformly sampled mini-batches. Each update has two
parts.

The **critic update** fits the critic to the temporal-difference target

  target = reward + γ · (1 − done) · Q_target( next state, next stage's action )

where the next stage's action is produced by the target actor of whichever
agent is active next. The critic loss is the mean-squared error between this
target and the critic's current estimate.

The **actor update** is performed once per agent. For each agent, the batch is
filtered to the transitions in which that agent was active; the agent's actor is
run on those states, and its parameters are adjusted by gradient ascent on the
critic's value of the resulting action. The discrete-action context is held
fixed during this step, since the discrete selection is not differentiable.

After every update, the target networks are moved a small fraction τ toward the
live networks (a soft update). Slow targets prevent the actor–critic feedback
loop from oscillating.

The controller is trained online against the live RAG environment, using
OpenAI `gpt-4o-mini` for all language-model calls. If an environment step raises
an exception — for example, a transient API failure — a terminal transition
carrying a negative reward is written to the buffer so that training continues
and the failure still produces a learning signal.

### 4.12.8 Reward Function

The controller is trained on the same single, shared, cooperative scalar reward
used throughout the project — all five agents receive the same reward, computed
mostly at episode termination. The reward combines:

- **Answer quality** — a blend of embedding similarity and token-F1 against the
  reference answer (the dominant term);
- **Evidence utilisation** — a term rewarding the use of a substantial evidence
  pack, capped so it cannot be inflated by over-retrieval;
- **Retrieval F1** — whether the correct source documents were retrieved;
- **Citation support** and **verification pass** — smaller terms;
- **Latency and step costs** — small negative terms;
- **Penalties** — fixed negative terms for an empty answer, unsupported claims,
  hallucination, and timeout.

The answer-quality and evidence-utilisation terms were not part of the initial
reward design. They were introduced after an iterative diagnosis described in
Section 5.7.5, in which the controller was found to reward-hack an earlier
reward formulation by minimising evidence use.

### 4.12.9 Relationship to Textbook MADDPG

The controller is described as *MADDPG-style* rather than as MADDPG because it
deviates from the textbook formulation in three respects, each motivated by the
stage-gated RAG environment.

1. **Sequential, not simultaneous, agents.** Textbook MADDPG assumes all agents
   act at every timestep. The RAG workflow is causal — the grader cannot act
   before retrieval — so exactly one agent acts per step.

2. **Stage-conditioned, not joint-action, critic.** Because only one agent acts
   per step, a joint-action critic would be trained largely on zero-padded
   inputs. The critic is instead conditioned on the active agent and its action.

3. **Shared global observation, not per-agent local observations.** Textbook
   MADDPG gives each actor a private local observation. Here every actor
   receives the same global state. The agents run sequentially in one process,
   so there is no decentralised-deployment requirement, and shared cross-stage
   context is useful.

The controller retains the core of MADDPG and DDPG — per-agent deterministic
actors, a centralised critic, off-policy replay, target networks, and
Ornstein-Uhlenbeck exploration. It is most precisely described as a
sequential-agent, parameterised-action, off-policy actor-critic with a
stage-conditioned centralised critic. It still follows centralised-training,
decentralised-execution: the critic is used only in training, and at evaluation
each actor runs independently with no critic and no inter-agent communication.

### 4.12.10 Discussion and Limitations

The controller adds no language-model calls beyond those the underlying RAG
pipeline already makes; its overhead is one small neural-network forward pass
per step. Its main limitations are that the discrete action selection is not
differentiable — the actor learns only the continuous parameters — and that the
reward is a proxy for answer quality rather than a direct measure of it. The
consequences of this proxy are examined in Section 5.7.

---

# D. New Section 5.7 — Reinforcement Learning Controller Evaluation

> Insert as Section 5.7, after Section 5.6 (Summary of Findings), or move the
> existing 5.6 summary to follow it.

## 5.7 Reinforcement Learning Controller Evaluation

### 5.7.1 Experimental Setup

The reinforcement-learning controller was evaluated on the expanded
145-question benchmark (Section 4.12.3), using its 29-question held-out test
split. All systems in this section use the same expanded corpus, the same
hybrid-retrieval backbone, and OpenAI `gpt-4o-mini` for generation. Because the
benchmark, corpus, and generation model differ from those behind Table 5.1, the
results in this section are a **separate study** and are not directly comparable
to the architecture-level results in Section 5.1.

The controller was trained for 200 episodes on the 88-question training split
and evaluated from its final periodic checkpoint, `ep_0200.pt` (392 gradient
updates). Four of the fixed architectures — Simple Hybrid RAG, Self-RAG Grader,
CRAG Rewrite, and Final Hybrid RAG — were re-run on the same 29-question test
split to provide a contemporary comparison. The Supervisor and Multi-Agent
architectures were not re-run on the expanded benchmark.

### 5.7.2 Architecture-Level Results

Table 5.2 reports the results on the 29-question expanded test split. Token F1
is reported with its standard error.

**Table 5.2 — Expanded-benchmark evaluation (29-question test split)**

| System | Token F1 | ROUGE-L | Pass rate | Failure rate | Avg. latency (s) |
|---|---:|---:|---:|---:|---:|
| Self-RAG Grader | 0.490 ± 0.028 | 0.390 | 1.00 | 0.00 | 4.3 |
| Simple Hybrid RAG | 0.487 ± 0.025 | 0.367 | 1.00 | 0.00 | 1.7 |
| Final Hybrid RAG | 0.482 ± 0.027 | 0.354 | 1.00 | 0.00 | 32.8 |
| CRAG Rewrite † | 0.425 ± 0.039 | 0.314 | 0.86 | 0.14 | 2.0 |
| **RL Controller (MADDPG, trained)** | **0.434 ± 0.031** | 0.310 | 0.97 | 0.03 | 22.7 |

† On the expanded benchmark the CRAG Rewrite implementation failed on 4 of 29
questions because of an internal code defect (a variable referenced before
assignment), which is unrelated to the evaluation harness. On the 25 questions
it completed, its Token F1 is 0.493; the 0.425 aggregate is depressed by its own
crashes. The defect should be fixed and the architecture re-evaluated.

The trained reinforcement-learning controller scores 0.434 ± 0.031 Token F1.
This places it **below all three fixed architectures that completed every
question** (0.482–0.490) and roughly level with the crashed CRAG Rewrite. The
gap to the strongest baselines is approximately 0.05 Token F1, or about 1.3
standard errors — not a statistically conclusive difference at the n = 29 scale,
but a consistent shortfall on every quality metric (Token F1, ROUGE-L) rather
than an even split. The controller is also the only learned system that ever
fails (1 of 29) and is the second-slowest at 22.7 seconds per query.

The honest reading is that the trained controller **does not reach the answer
quality of the hand-configured pipelines**. As Section 5.7.4 shows, it is also
no better than its own untrained initialisation — training did not lift it.

### 5.7.3 Category-Wise Analysis

A category-wise comparison between the trained controller and the Simple Hybrid
RAG baseline confirms the aggregate shortfall. The controller wins exactly one
category — definition/explanation (Token F1 0.530 against 0.494, n = 6) — ties
multi-chunk synthesis (0.463 against 0.473, n = 7) and paraphrase-hard retrieval,
and loses every other category, including a sharp collapse on figure/table-
grounded questions (0.190 against 0.653, n = 2). Because several categories
contain only one or two test questions, the category-level differences are
indicative rather than conclusive, but no category shows the controller clearly
outperforming the baseline. The earlier expectation that learned per-query
control would help most on synthesis-oriented categories is not borne out: on
multi-chunk synthesis, the flagship category the expanded benchmark was designed
around, the controller merely matches the baseline.

### 5.7.4 The Training Curve — Training Does Not Improve Answer Quality

The decisive evidence for the controller's behaviour is the training curve.
Five checkpoints spanning the 200-episode run, plus the 500-episode run, were
each evaluated on the same 29-question test split:

**Table 5.3 — Held-out Token F1 against training progress**

| Checkpoint | Gradient updates | Token F1 (n = 29) |
|---|---:|---:|
| Untrained (initial policy) | 0 | 0.474 ± 0.028 |
| 50 episodes | 69 | 0.439 ± 0.032 |
| 100 episodes | 170 | 0.432 ± 0.032 |
| 150 episodes | 286 | 0.445 ± 0.024 |
| 200 episodes (final) | 392 | 0.434 ± 0.031 |
| 500-episode run | 986 | 0.419 ± 0.027 |

Across the entire training span — from 0 to 986 gradient updates — held-out
Token F1 does not rise. It moves from 0.474 at initialisation to 0.434 at 200
episodes and 0.419 at 500 episodes: flat, with a slight downward drift. Every
trained checkpoint lies within roughly one standard error of every other, so no
pairwise difference is statistically significant; but the curve never exceeds
its untrained starting point.

The interpretation is that **training the controller on the v4 reward did not
produce a better controller.** The reward is a proxy for answer quality — a
blend of embedding similarity and token overlap, plus an evidence-utilisation
term — and optimising that proxy changed the policy's *behaviour* (measured
evidence use roughly doubled, from about two chunks to five) without improving
the answers it produced on unseen questions. The proxy reward and the true
objective are only loosely coupled, and the coupling is too weak for the
learned policy to add measurable value over its random initialisation.

The mild decline at 500 episodes is consistent with light proxy-reward
over-optimisation, but it is within sampling noise and should not be reported
as a clean "early-stopping" result. The defensible claim is the stronger and
simpler one: on this architecture, reward, and benchmark, training the
controller does not improve held-out answer quality.

### 5.7.5 The Iterative Reward-Design Process

The reward function described in Section 4.12.8 is the outcome of an iterative
diagnosis. Earlier reward formulations produced a consistent failure mode: the
trained controller **minimised evidence use**, converging on policies that
passed verification with only one or two evidence chunks. This satisfied the
verification and citation terms of the reward — a short answer making few claims
is easy to support — while producing thin answers.

Successive attempts to remove this behaviour by adjusting parameter ranges and
rebalancing reward weights did not work; the controller repeatedly found an
alternative parameter through which to reduce evidence. The behaviour was only
removed by changing the reward itself: replacing a purely lexical answer-quality
term with a blend of embedding similarity and token F1, and adding an explicit
evidence-utilisation term that rewards using a substantial evidence pack. With
this reward the controller's evidence use roughly doubled, from about two chunks
to five.

It is important to be precise about what the redesign achieved. It fixed the
*behaviour* it targeted — the controller stopped minimising evidence — but, as
the training curve in Section 5.7.4 shows, it did **not** improve held-out
answer quality. The redesigned reward was a better-behaved proxy, not a
better-aligned one.

This diagnosis is reported because it is itself a contribution. It is a concrete
instance of reward hacking — a policy optimising the specified reward in a way
that does not serve the intended objective — and, taken together with the flat
training curve, it illustrates a single lesson: in an RL controller for RAG, the
reward function is the hardest part of the design, and a hand-crafted proxy
reward, even after careful iteration, may shape behaviour without improving the
true objective.

### 5.7.6 Summary

On the expanded benchmark, the trained reinforcement-learning controller scores
below the fixed and agentic architectures on answer quality (Token F1 0.434
against 0.482–0.490), wins only one of eight question categories, and is no
better than its untrained initialisation. Training on the v4 proxy reward did
not improve held-out performance at any budget from 50 to 500 episodes. The
controller is also slower than the lightweight baselines and is the only learned
system that fails on any question. This is a negative result, and it is reported
as one. Its value to the thesis is twofold: it confirms that the
stage-conditioned MADDPG-style architecture can be trained stably end-to-end on
live LLM calls, and it produces a well-instrumented cautionary finding — a
hand-designed proxy reward, even after a reward-hacking failure mode is
diagnosed and fixed, does not transfer to true answer quality. Both are relevant
to any future RL controller for retrieval-augmented generation.

---

# E. New Section 6.4 — Discussion of the Reinforcement Learning Controller

> Insert as Section 6.4, after Section 6.3 (Trade-Offs Observed).

## 6.4 The Reinforcement Learning Controller

The reinforcement-learning controller was the project's attempt to replace
hand-designed configuration with a learned policy. Its evaluation supports a
measured but unambiguous conclusion: as built, the learned continuous-control
policy **did not reach** the performance of carefully hand-configured RAG
pipelines, and training it did not improve its held-out answer quality.

This is a negative result, and it is reported as one — but it is an informative
negative result rather than a failed experiment. The fixed architectures in
this project are not weak baselines; they implement well-motivated mechanisms —
hybrid retrieval, reranking, corrective rewriting, document grading,
claim-level verification — and were tuned by hand. The controller, trained on
145 questions, scored about 0.05 Token F1 below the strongest of them and no
better than its own random initialisation. The architecture itself works: it
trains stably end-to-end on live language-model calls, with hundreds of
gradient updates and clean optimisation. What does not work, on this evidence,
is the assumption that a hand-designed proxy reward will guide that architecture
toward better answers.

The evaluation isolates *why*. The first issue is reward hacking: under earlier
reward formulations the controller minimised evidence use to satisfy the
verification reward, and the behaviour was removed only by redesigning the
reward (Section 5.7.5). The second, and more fundamental, is that even the
redesigned reward did not transfer: the training curve (Section 5.7.4) is flat
from 0 to nearly a thousand gradient updates. The reward shaped the policy's
behaviour without improving its answers. Both findings reinforce a theme already
present in the discussion of the fixed architectures — that added machinery must
be justified by measured benefit. For an RL controller, the most consequential
and most error-prone piece of machinery is the reward function, and here it was
not aligned closely enough with answer quality for learning to help.

The controller as built has clear limitations. The discrete action selection is
not differentiable, so the actor learns only continuous parameters. The reward
is a proxy: embedding similarity and token F1 measure resemblance to a reference
answer, not correctness. And the controller was trained and evaluated on an
LLM-generated benchmark of moderate size. These limitations frame the future
work in Section 7.

---

# F. Revised Section 7 — Future Work

> The original Section 7 lists "develop the reinforcement learning controller"
> as future work. The controller now exists, so the RL-related paragraphs
> should be revised. Replace the two paragraphs beginning *"The reinforcement
> learning layer should be developed further…"* with the following.

The reinforcement-learning controller has been implemented and evaluated
(Sections 4.12 and 5.7), but several extensions would strengthen it. The most
important is a better-aligned reward signal, and it is the direct response to
this work's central negative result. The current reward blends embedding
similarity and token F1, both of which measure resemblance to a reference answer
rather than correctness; the training curve shows that optimising this proxy
does not improve held-out answer quality. Replacing it with an LLM-as-judge
reward — a language model scoring each answer against the reference — would
align the training signal more closely with the true objective, and is the most
likely route to a controller that actually improves with training.

A second extension is a larger, human-authored benchmark. The controller was
trained on 145 LLM-generated and automatically-validated questions. A larger
benchmark, ideally with human-written questions and answers across more domains,
would reduce evaluation variance — the n = 29 test split admits standard errors
of roughly 0.03 — and would support stronger generalisation claims.

A third extension concerns the action space. The discrete action selection is
currently a fixed, non-differentiable function of the continuous parameters; a
differentiable relaxation, such as a Gumbel-softmax selector, would let gradient
information flow through the discrete choice and would let the controller learn
the discrete policy directly rather than only the continuous parameters.

Finally, the controller could be compared directly against the supervisor and
multi-agent architectures on the expanded benchmark, which would place all
adaptive approaches — heuristic and learned — under a single comparison.

> The remaining future-work paragraphs (benchmark expansion, citation
> verification, multimodal grounding, latency optimisation, adaptive retrieval,
> human evaluation, deployment) remain valid and need no change.

---

# G. Additions to Section 8 — Conclusion

> Insert the following paragraph after the paragraph describing the Multi-Agent
> architecture and before the paragraph beginning *"These findings show that
> academic RAG cannot be judged using a single metric."*

Beyond the fixed and agentic architectures, the project implemented a
reinforcement-learning controller: a stage-conditioned, MADDPG-style
continuous-control policy in which each agent's actor learns the numeric
parameters of its RAG stage, trained off-policy against the live environment.
The architecture trains stably end-to-end on live language-model calls.
Evaluated on an expanded 145-question benchmark, however, the trained controller
did not reach the answer quality of the strongest fixed pipelines, and training
it did not improve held-out performance over its untrained initialisation. Its
evaluation is therefore valuable as a negative result, supported by two
methodological findings — a reward-hacking failure mode, in which the policy
minimised evidence use to satisfy a misaligned reward, and a flat training curve
showing that even the redesigned reward did not transfer to true answer quality.
Both findings show that, for a learned RAG controller, the reward function is as
central to the design as the policy architecture, and that a hand-designed proxy
reward is not sufficient to make learned control competitive with hand-tuned
pipelines.

> Optionally, extend the final sentence of the conclusion to acknowledge the
> learned controller:

The system developed in this thesis provides a foundation for building academic
RAG systems that are more grounded, transparent, and trustworthy; and its
reinforcement-learning controller, while not yet competitive with hand-designed
pipeline configuration, establishes both a working architecture for learned
continuous control and a clear diagnosis of the reward-alignment problem that
must be solved before such control can deliver a measurable benefit.

---

# Appendix — Key numbers for the report (verified)

All figures below are from the expanded-benchmark evaluation; see `RESULTS.md`
for full detail and artifact paths.

| Quantity | Value |
|---|---|
| Corpus | 12 papers, ≈1,226 chunks |
| Benchmark | 145 questions (88 train / 28 val / 29 test) |
| Generation model | OpenAI `gpt-4o-mini` |
| RL controller — Token F1 (trained, `ep_0200.pt`) | 0.434 ± 0.031 (n = 29) |
| RL controller — gradient updates (200 ep) | 392 |
| RL controller — Token F1 (untrained initialisation) | 0.474 ± 0.028 |
| RL controller — Token F1 (500-episode run) | 0.419 ± 0.027 |
| RL controller — gradient updates (500 ep) | 986 |
| Self-RAG Grader — Token F1 | 0.490 ± 0.028 |
| Simple Hybrid RAG — Token F1 | 0.487 ± 0.025 |
| Final Hybrid RAG — Token F1 | 0.482 ± 0.027 |
| CRAG Rewrite — Token F1 | 0.425 (0.493 excluding its 4 crashes) |
| Continuous action dimensions | 14 |
| State dimension | 14 (base) or 20 (Context Engineering Block) |

Training curve (held-out Token F1 vs gradient updates, same 29-q test set):
0 → 0.474, 69 → 0.439, 170 → 0.432, 286 → 0.445, 392 → 0.434, 986 → 0.419.

**Do not state** that the controller "beats" or "matches" the baselines — the
trained controller scores below them. The correct framing is a *negative
result*: training on the proxy reward did not improve held-out answer quality.
**Always** evaluate the RL controller from a periodic checkpoint (`ep_0200.pt`),
never from `best_reward.pt` — see Fact 3 in the integration notes. **Do not
merge** Table 5.2 with Table 5.1 — they use different benchmarks and generation
models.
