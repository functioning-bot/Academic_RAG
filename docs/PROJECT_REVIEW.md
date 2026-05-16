# Project Review — Stage-Conditioned MADDPG-style RL for Academic RAG

A complete walkthrough for presenting and defending this project. Read top to
bottom: it goes from "what problem are we solving" → ingestion → retrieval →
the multi-agent RL controller → training → results → likely defense questions.

Every section explains **what** a component does and **why** it exists — because
"why" is what a committee asks.

---

## 0. One-paragraph summary

This project builds a Retrieval-Augmented Generation (RAG) system for academic
papers, and then puts a **reinforcement-learning controller** on top of it. The
RAG pipeline has five stages — retrieve, (rewrite), grade, generate, verify.
Instead of hard-coding the parameters of each stage (how many chunks to fetch,
generation temperature, how strict the verifier is), we train five cooperating
RL agents to choose those parameters per query. The RL method is a **MADDPG-style
continuous-control architecture adapted for a stage-gated environment**: each
agent has its own deterministic actor network, and a single centralized critic
is conditioned on which stage is currently acting.

---

## 1. The problem & motivation

**Plain RAG is statically configured.** A normal RAG system picks, once and
forever, "retrieve top-10 chunks, temperature 0.3, medium grading." But queries
differ:

- A simple factual lookup needs few chunks and a short answer.
- A multi-source comparison needs broad retrieval and careful grading.
- An ambiguous query may need rewriting before retrieval.

**The thesis question:** *Can an RL controller learn to tune RAG parameters
per-query better than a fixed configuration?*

We answer it by comparing a trained RL policy against a fixed-action baseline on
the same benchmark.

---

## 2. High-level architecture

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │                         OFFLINE — INGESTION                           │
   │                                                                        │
   │   PDF ──► LlamaParse ──► Markdown ──► Vision model ──► Chunker ──►     │
   │           (Agentic)      + figures    (figure desc.)   (semantic)      │
   │                                                            │           │
   │                                            ┌───────────────┘           │
   │                                            ▼                           │
   │                          BGE-M3 (dense) + BM25 (sparse) embeddings     │
   │                                            │                           │
   │                                            ▼                           │
   │                              Qdrant hybrid vector DB                   │
   └────────────────────────────────────────────┬───────────────────────────┘
                                                 │
   ┌─────────────────────────────────────────────┼───────────────────────────┐
   │                       ONLINE — QUERY TIME    │                           │
   │                                              ▼                           │
   │   Question ──► MARLEnv ──► [retrieve ─► grade ─► generate ─► verify]     │
   │                              │         │         │           │          │
   │                              ▼         ▼         ▼           ▼          │
   │                       each stage's parameters chosen by an RL actor     │
   │                                                                          │
   │   RL controller = 5 actors + 1 stage-conditioned critic + replay buffer  │
   └──────────────────────────────────────────────────────────────────────────┘
```

Two completely separate phases:
- **Ingestion** (offline, run once per document): turn PDFs into a searchable index.
- **Query time** (online): answer a question by running the RAG pipeline, with the RL controller picking parameters.

---

## 3. Stage 1 — Ingestion pipeline (offline)

Goal: turn a messy PDF into clean, searchable, richly-tagged chunks in a vector
database. Code: `ingestion/watcher.py`, `ingestion/chunker.py`, `ingestion/indexer.py`.

### 3.1 Parse — PDF → Markdown (`watcher.py`)

**What:** A `watchdog` file watcher monitors `ingest_folder/`. When a PDF lands,
it is uploaded to **LlamaParse** (LlamaCloud's "Agentic" parsing tier), which
returns clean **Markdown**, one block per page, with tables emitted as Markdown
tables.

**Why not just extract text with PyPDF2?**
Academic PDFs are visually complex: two-column layouts, dense tables, equations,
figures, footnotes. Naive text extraction reads columns in the wrong order,
turns tables into unreadable streams of numbers, and drops equation structure.
LlamaParse Agentic uses a vision-language model to *understand the page layout*
and emit structurally faithful Markdown. We pay for parse quality because
everything downstream (chunking, embedding, retrieval) inherits the parse's
errors.

**Figures:** if image description is enabled, each extracted figure is sent to a
local **vision model (Ollama `llava`)**, optionally grounded with OCR text and
the figure caption. The model writes a short text description of the figure.

**Why describe figures?** A figure is a binary image — a vector retriever cannot
match a question against pixels. By generating a *text description* of each
figure ("a diagram showing the transformer encoder-decoder stack..."), the
figure becomes retrievable like any other passage. This is the "multimodal"
part of the system.

Output: one enriched Markdown file per PDF, pages separated by `---`, with an
appended `## Extracted Figures & Visual Descriptions` section.

### 3.2 Chunk — Markdown → semantic chunks (`chunker.py`)

**What:** `chunk_markdown()` splits the Markdown into retrieval units ("chunks")
of ~1300 characters with 150-character overlap. The splitting is *hierarchical*:

1. **Split into pages** (on the `---` separator) — keeps citations page-accurate.
2. **Clean** each page: remove inline image markdown, fix hyphenation broken
   across lines, merge PDF hard-wrapped lines, strip HTML remnants.
3. **Extract special blocks first** — Markdown tables, mermaid diagrams, and
   display equations (`$$...$$`) become *standalone chunks* with their own
   `content_type`.
4. **Header-aware split** — split remaining prose on Markdown headers (`#`, `##`,
   `###`) so each chunk belongs to one section.
5. **Recursive character split** — within a section, split to the target chunk
   size on paragraph → line → sentence → word boundaries (in that priority).
6. **Tag metadata** — every chunk records `source_file`, `page_number`,
   `section_header`, `chunk_index`, `content_type`, etc.

**Why chunk at all?** Two reasons. (a) Embedding models have a token limit and
produce one vector per input — a whole 20-page paper would collapse to one
fuzzy vector. (b) Retrieval should return *focused passages*, not entire
documents, so the generator sees only relevant text.

**Why hierarchical / why split on headers?** A chunk that straddles a section
boundary or is half-table-half-prose produces a muddy embedding that matches
nothing well. Splitting on semantic boundaries (headers, then paragraphs) keeps
each chunk *about one thing*. Tables and equations are extracted whole because
chopping a table in half destroys its meaning.

**The `text` vs `embedding_text` distinction:** each chunk stores two strings.
`text` is the clean, page-local payload used for *display and citation*.
`embedding_text` is what actually gets embedded — it prepends the section header
and (for the first chunk on a page) a 300-character "boundary tail" from the
previous page. **Why:** retrieval quality improves when the embedding knows the
chunk's section context and what immediately preceded it across a page break;
but you don't want that extra context polluting the cited answer text.

### 3.3 Embed — chunks → vectors (`indexer.py`)

Each chunk's `embedding_text` is turned into **two** different vector types.

**Dense embedding — BGE-M3 (`BAAI/bge-m3`, 1024-dim).**
A dense embedding maps text to a fixed-length vector of floats such that texts
with *similar meaning* land near each other, even if they use different words.
"How does attention work" and "the self-attention mechanism computes..." are
far apart lexically but close in dense space. BGE-M3 ("M3" = Multilingual,
Multi-Function, Multi-Granularity) handles long inputs (up to 8192 tokens) and
is strong on technical text. **Why dense:** users phrase questions in their own
words; dense retrieval matches *concepts*, not *spelling*.

**Sparse embedding — BM25 (`Qdrant/bm25` via FastEmbed).**
BM25 ("Best Match 25") is the classic lexical-retrieval algorithm. It represents
text as a sparse bag-of-words weighted by term frequency and inverse document
frequency. It only matches *exact terms* (and their stems). **Why sparse:**
dense embeddings are weak on rare, out-of-vocabulary tokens — acronyms, author
surnames, exact equation symbols, dataset names. BM25 nails these. If a user
asks for "BLEU score in Table 3", BM25 finds the literal string; a dense model
might drift to "evaluation metrics" in general.

**Why keep both (hybrid)?** Dense and sparse have *complementary failure modes*.
Dense handles paraphrase but blurs rare terms; sparse handles rare terms but
fails on paraphrase. Indexing both means retrieval can use whichever (or a fused
combination) the query needs.

### 3.4 Index — store in Qdrant (`indexer.py`)

**What:** Both vectors per chunk are upserted into a **Qdrant** collection
(`academic_papers`) configured with a *named* dense vector ("dense", 1024-dim,
cosine distance) and a *named* sparse vector ("sparse", IDF modifier). Each
point's payload is the chunk text + all metadata. Point IDs are deterministic
(`uuid5(source_file + chunk_index)`) so re-indexing a paper *replaces* its old
chunks instead of duplicating them (upsert-new-then-delete-stale).

**Why Qdrant?** It is a vector database that natively stores dense and sparse
vectors *in the same collection* and can fuse them server-side (see §4.3). We
don't need a separate keyword search engine. It runs locally via Docker
(`docker-compose.yml` → port 6333).

**Why "cosine distance" for dense?** Cosine similarity measures the *angle*
between vectors, ignoring magnitude — standard for semantic embeddings, where
direction encodes meaning and length is noise.

---

## 4. Stage 2 — Retrieval (online)

Code: `brain/retriever_shared.py`, `brain/context_marl_ac/adapters/retriever_adapter.py`,
`brain/final_arch/reranker_shared.py`.

Given a query, retrieval finds the most relevant chunks. The retriever agent can
use four strategies:

### 4.1 Dense retrieval

Embed the query with BGE-M3 → Qdrant approximate-nearest-neighbour search over
the "dense" vectors → top-k chunks by cosine similarity. **Best for** conceptual
/ paraphrased questions.

### 4.2 Sparse retrieval

Embed the query with BM25 → Qdrant sparse-vector search → top-k by BM25 score.
**Best for** exact-term / keyword questions.

### 4.3 Hybrid retrieval (RRF fusion)

Run dense and sparse search *in parallel* (each fetches 20 candidates), then fuse
the two ranked lists with **Reciprocal Rank Fusion (RRF)**.

**What is RRF?** A rank-combination formula. Each document gets a score
`Σ 1/(k + rank_in_list)` summed over the lists it appears in. A document ranked
highly by *both* dense and sparse rises to the top; a document ranked highly by
only one still scores reasonably. **Why RRF and not score-averaging?** Dense
cosine scores and BM25 scores live on totally different scales — averaging them
is meaningless. RRF only uses *rank position*, so it is scale-invariant. Qdrant
computes this server-side via `FusionQuery(fusion=RRF)`.

### 4.4 Hybrid + rerank

Run hybrid retrieval to get ~20 candidates, then **rerank** them with a
**CrossEncoder** (`BAAI/bge-reranker-v2-m3`).

**Bi-encoder vs cross-encoder.** The dense retriever is a *bi-encoder*: it embeds
the query and the document *separately*, then compares vectors. Fast (documents
pre-embedded) but approximate — the two never "see" each other. A *cross-encoder*
feeds the `(query, document)` pair *together* through a transformer and outputs
a single relevance score. Much more accurate because it can model fine-grained
interaction — but slow, so it can't run over the whole corpus.

**Why both?** Standard two-stage retrieval: a cheap bi-encoder/​BM25 stage casts
a wide net (20 candidates), then an expensive cross-encoder precisely re-orders
just those 20. Best accuracy where it matters, affordable cost.

`retrieve_more` is a fifth action: hybrid retrieval that *excludes* chunks
already retrieved — used only on the recovery path when the verifier asks for
more evidence.

---

## 5. Stage 3 — The multi-agent RAG environment

Code: `brain/context_marl_ac/marl/`, `brain/context_marl_ac/agents/`.

The RAG pipeline is modelled as a **Markov Decision Process** with five agents.

### 5.1 The five agents

| Agent | Job | Real operation it performs |
|---|---|---|
| **retriever** | Fetch evidence | dense / sparse / hybrid / rerank search in Qdrant |
| **rewriter** | Reformulate the query | LLM rewrites the query (recovery only) |
| **grader** | Filter retrieved chunks | LLM relevance grading, or score-sort |
| **generator** | Write the answer | LLM generates a grounded answer |
| **verifier** | Check the answer | LLM verifies each claim against evidence |

### 5.2 ContextState — the shared state

`ContextState` is one mutable object per episode. It holds the query, retrieved
chunks, graded/selected evidence, generated answer, verification result, step
counts, latency, token usage, and `final_status`. Every agent reads and mutates
the *same* state object — that is how they cooperate.

### 5.3 Stage gating — who acts when (`action_masking.py`)

The agents do **not** act simultaneously. At any moment, exactly one agent is
"active", determined by the state via an **action mask**. The main path is:

```
retriever ─► grader ─► generator ─► verifier
                                       │
                  ┌────────────────────┤ (on verification FAIL, retry_count<2)
                  ▼                    ▼                    ▼
        request_regeneration   request_more_retrieval   request_rewrite
                  │                    │                    │
              generator         retriever.retrieve_more   rewriter
                  │                    │                    │
                  │                  grader              retriever
                  │                    │                    │
                  │                  generator            grader
                  │                    │                    │
                  └──────► verifier ◄───┴──► generator ─► verifier
```

The **action mask** is a binary vector per agent saying which of its discrete
actions are legal *right now*. `find_active_agent_and_valid_actions()` scans all
five masks and returns the one agent with a non-empty mask. **Why stage-gate?**
It keeps the workflow structurally valid (you cannot grade before retrieving)
and makes training stable — the policy never has to learn the workflow order,
only how to *parameterize each stage*.

After at most 2 recovery attempts, recovery actions become illegal and the
episode ends as `rejected`. Hard caps: 12 steps, 15 LLM calls per episode.

### 5.4 The reward function (`reward.py`, weights in `config.py`)

A single cooperative scalar reward, mostly paid at episode end:

| Term | Weight | Meaning |
|---|---|---|
| Answer quality | **0.45** | blend: 0.5 × embedding-similarity + 0.5 × token-F1 vs gold |
| Evidence utilization | **0.15** | `min(evidence_count, 6) / 6` — rewards using a real evidence pack |
| Retrieval F1 | 0.20 | did retrieval fetch the correct source documents |
| Citation support | 0.10 | fraction of answer claims supported by evidence |
| Verification pass | 0.10 | +bonus if the verifier accepted |
| Latency cost | −0.05 | penalize slow episodes |
| Step cost | −0.02 | small per-step penalty (discourages dithering) |

(These are the **v4** weights — see §8 / `docs/RESULTS.md`. The
evidence-utilization term and the embedding-similarity blend were added in v4
to break the minimal-evidence reward-hack that earlier weight sets could not.)

Plus fixed penalties (constant across v1/v2/v3 — only the positive weights above
were rebalanced):

| Penalty | Value | When applied |
|---|---|---|
| Empty answer (`PENALTY_NO_ANSWER`) | **−0.50** | Terminal, if the generated answer is blank |
| Unsupported claim (`PENALTY_UNSUPPORTED_CLAIM`) | **−0.10 per claim** | Terminal, scales with the number of unsupported claims |
| Timeout (`PENALTY_MAX_STEPS`) | **−0.20** | Terminal, if `final_status == "timeout"` |
| Hallucination / rejection (`PENALTY_HALLUCINATION`) | **−0.30** | Terminal, if `final_status == "rejected"` |
| Repeated action (`PENALTY_REPEATED_ACTION`) | **−0.05** | Mid-episode, if a non-retriever agent repeats its previous action |

(`PENALTY_INVALID_ACTION = −0.10` is defined in `config.py` but not currently
wired into `reward.py`.)

**Why mostly-terminal reward?** The thing we actually care about — answer
quality — is only measurable once an answer exists. Intermediate steps get only
the small step cost. This is a *sparse-reward* RL problem, which is why
sample-efficiency and the reward design matter so much.

---

## 6. Stage 4 — The MADDPG-style RL controller

Code: `brain/maddpg/`.

This is the contribution. The RL controller decides the *parameters* of each
RAG stage. It is an off-policy, continuous-control, multi-agent actor-critic.

### 6.1 What the controller controls

Each agent has an **actor network** that, given the current state, outputs a
continuous vector in [−1, 1]. A **mapper** converts that vector into real RAG
parameters *and* picks a discrete action name:

| Agent | Continuous params it sets |
|---|---|
| retriever | dense/sparse weight, `top_k` (5–20), rerank threshold, source diversity |
| rewriter | rewrite strength, query expansion weight |
| grader | relevance threshold, evidence keep ratio (0.7–1.0), strictness |
| generator | temperature, citation strictness, `max_tokens` (128–384), detail level |
| verifier | support threshold |

**14 continuous dimensions total.** The discrete action name (e.g.
`hybrid_rerank`) is *derived* from those continuous params by a selector
function — it is not chosen directly and is not differentiable.

### 6.2 The actors

Five `MADDPGActor` MLPs (one per agent). Each: `state → Linear → LayerNorm →
ReLU → Linear → LayerNorm → ReLU → Linear → tanh`. The `tanh` bounds outputs to
[−1, 1]. Each actor also has a **target actor** — a slow-moving copy used to
stabilize training.

**Why deterministic actors?** This is DDPG-family RL. Unlike a stochastic policy
that outputs a probability distribution, a deterministic actor outputs *the*
action directly. This works for continuous action spaces and enables the
off-policy training below. Exploration during training comes from added
**Ornstein-Uhlenbeck noise**, not from policy stochasticity.

### 6.3 The stage-conditioned critic

One `StageConditionedCritic`. It estimates the **Q-value** — the expected total
future reward — of taking an action in a state:

```
Q( state, active_agent_one_hot, discrete_action_one_hot, padded_continuous_action ) → scalar
```

**Why "stage-conditioned" and not textbook MADDPG?** Textbook MADDPG uses a
critic over the *joint action of all agents* `Q(s, a₁…a₅)`. But here only one
agent acts per step, so 4 of the 5 action slots would always be zero — a
diluted, wasteful signal. Instead the critic is told *which* agent is acting
(one-hot), *which* discrete action they took (one-hot), and *that agent's*
continuous action (zero-padded to a uniform width). Every training sample has
the same shape and every input dimension is informative.

The critic also has a **target critic** copy.

### 6.4 The replay buffer

Every environment step produces one **Transition** — (state, active agent,
action, reward, next state, next active agent, done, …) — stored in a
**replay buffer** (capacity 50k). **Why a buffer?** This is *off-policy* RL:
instead of learning only from the most recent episode, we sample random
mini-batches of *past* transitions. Each experience is reused many times →
far more sample-efficient, which matters with a small benchmark.

### 6.5 The training update (`trainer.py`)

Every few env steps, `trainer.update()` runs one gradient step:

1. **Sample** a mini-batch (64) of transitions from the buffer.
2. **Critic update.** For each transition compute the TD target:
   `target = reward + γ · (1−done) · target_critic(next_state, next_agent, …)`,
   where the next action comes from the *target actor* of whichever agent is
   active next. Critic loss = MSE(current Q, target). This teaches the critic to
   predict reward-to-go.
3. **Actor update.** For each agent, take the batch rows where it was active,
   run its actor to get fresh actions, and push the critic's Q *up* by gradient
   ascent: `actor_loss = −critic(state, …, actor(state)).mean()`. This teaches
   each actor to output actions the critic rates highly.
4. **Soft target update.** Nudge every target network a tiny fraction (τ=0.005)
   toward its live network. Slow targets prevent the feedback loop between actor
   and critic from oscillating.

**Centralized training, decentralized execution (CTDE) — partially.** The critic
(used only in training) sees more than any single actor. At evaluation only the
actors run, one per stage, no critic. (Caveat: all our actors share the global
state rather than private local observations — see the deviations doc.)

### 6.6 Why this is "MADDPG-*style*" not textbook MADDPG

Three deliberate deviations, each justified by the stage-gated environment:
1. **Sequential, not simultaneous** agents (the RAG pipeline is causal).
2. **Stage-conditioned critic, not joint-action critic** (avoids the zero-padding waste).
3. **Shared global observation, not per-agent local observations** (cross-stage
   context is useful and there is no deployment need for decentralization).

See `docs/MADDPG_EXTENSION.md` and the deviations discussion for the full
argument. The honest framing: *"a sequential-agent, parameterized-action,
off-policy actor-critic with a stage-conditioned centralized critic."*

---

## 7. End-to-end sequence diagram — answering one question

```
USER         MARLEnv        stage_utils       Actor[X]        Mapper        Agent[X]         LLM/Qdrant      Reward         Trainer
 │              │               │                │              │             │                │              │              │
 │ question ───►│               │                │              │             │                │              │              │
 │              │ reset()       │                │              │             │                │              │              │
 │              │ ContextState  │                │              │             │                │              │              │
 │              │               │                │              │             │                │              │              │
 │           ┌──┤ LOOP: while not done                                                                                          │
 │           │  │ find_active_agent_and_valid_actions() ─►│      │             │                │              │              │
 │           │  │◄── (agent X, valid_actions) ────────────│      │             │                │              │              │
 │           │  │               │                │       │      │             │                │              │              │
 │           │  │ encode state features (14 or 20-d) ─────►│     │             │                │              │              │
 │           │  │               │  actor[X].select_action(state)─►│            │                │              │              │
 │           │  │               │  raw continuous vector ◄─────────│           │                │              │              │
 │           │  │               │                │  map_agent_params(raw) ────►│                │              │              │
 │           │  │               │                │  params + discrete name ◄───│                │              │              │
 │           │  │ env.step(X, discrete, params) ──────────────────────────────►│                │              │              │
 │           │  │               │                │              │  agent.act(state, action)     │              │              │
 │           │  │               │                │              │  runs real RAG op ───────────►│              │              │
 │           │  │               │                │              │  mutated state ◄──────────────│              │              │
 │           │  │               │                │              │              calculate_reward(state) ──────►│              │
 │           │  │ (next_state, reward, done) ◄─────────────────────────────────────────────────│              │              │
 │           │  │               │                │              │             │                │              │              │
 │           │  │ push_transition(state, X, action, reward, next_state, …) ──────────────────────────────────►│              │
 │           │  │               │                │              │             │   if buffer ready: trainer.update() ─────────►│
 │           │  │               │                │              │             │                │   critic + actor gradient step│
 │           └──┤ END LOOP when state.done                                                                                      │
 │              │               │                │              │             │                │              │              │
 │◄─ answer ────│ episode summary: total_reward, final_status, latency …                                                        │
```

A typical accepted episode is 4 steps: retriever → grader → generator → verifier.

---

## 8. Training story & results (what actually happened)

Full detail is in `docs/RESULTS.md`. Summary here.

**Setup.** Corpus expanded 6 → 12 papers; benchmark expanded 60 → 145 questions
(88 train / 28 val / 29 test). LLM: OpenAI `gpt-4o-mini` (after Groq free-tier
rate limits). Primary metric: Token F1 vs gold. Baseline: `simple_hybrid_rag`
(hybrid retrieval → top-8 → one generation call).

**The iterative arc** — gap = Token F1 vs `simple_hybrid_rag` on the *same*
test set (absolute Token F1 is not comparable across the two benchmarks
because the expanded gold answers are longer):

| Step | Change | Gap to baseline |
|---|---|---:|
| v1–v3 | First trained policy + evidence-floor / reward-rebalance fixes | −11 to −14% |
| v3-exp | Un-throttled `top_k`; retrained on expanded benchmark | −8.6% |
| **v4** | **Reward redesign (200 ep)** | **−2.8%** |
| v4-500 | Same reward, 500 ep | −14.0% (overfit) |

**The recurring failure mode.** Through v1–v3-exp the policy reward-hacked by
using only ~1–2 evidence chunks — enough to satisfy the verifier, too little
for a good answer. The fix that worked (v4) was a **reward redesign**: a
semantic embedding-similarity blend for answer quality, a new
evidence-utilization reward term, and a higher `evidence_keep_ratio` floor.
That broke the minimal-evidence optimum (evidence count 2.1 → 4.5) and closed
the gap to −2.8%.

**Final result (v4, 200 ep — the recommended checkpoint):** Token F1
0.474 ± 0.028 vs the baseline's 0.487 ± 0.025 (n=29) — **a statistical tie**
(gap < 1 SE). MADDPG **wins 2 of 8 categories**, including the flagship
`multi_chunk_synthesis` (n=7). It matches the baseline on pass rate (100%) and
failure rate (0%) but is **slower and more LLM-expensive** (14 s / 2.1 calls vs
2 s / 1 call) — its value is adaptivity and parity, not efficiency.

**The training-budget finding.** v4 at 500 episodes scored *worse* than at 200
(Token F1 0.419 vs 0.474) — **proxy-reward over-optimization**: with 1,062
updates on 88 questions the policy kept improving the reward while the held-out
metric fell. 200 episodes is near the optimal budget.

**Defensible thesis claim:** *"A stage-conditioned MADDPG-style continuous-control
controller, with early stopping, reaches answer quality statistically
indistinguishable from a strong fixed-pipeline baseline (Token F1 0.474 vs
0.487, n=29) and wins the multi-chunk-synthesis category outright. The project
also identifies and fixes a reward-hacking failure mode and demonstrates
proxy-reward over-optimization via a training-budget curve."* It is **parity,
not superiority** — claiming a clean win would be dishonest.

---

## 9. Glossary — terms you must be able to define on the spot

| Term | One-line definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation — answer questions by retrieving documents then generating from them |
| **Chunk** | A ~1300-char passage of a document; the unit of retrieval |
| **Dense embedding** | A vector capturing *meaning*; similar meanings → nearby vectors (BGE-M3, 1024-d) |
| **Sparse embedding** | A bag-of-words vector capturing *exact term match* (BM25) |
| **Hybrid retrieval** | Combining dense + sparse results |
| **RRF** | Reciprocal Rank Fusion — scale-invariant way to merge two ranked lists |
| **Bi-encoder** | Embeds query and doc separately; fast, approximate |
| **Cross-encoder / reranker** | Scores a (query, doc) pair jointly; slow, accurate |
| **Qdrant** | Vector database storing dense + sparse vectors and doing RRF |
| **MDP** | Markov Decision Process — state, action, reward, next state |
| **MARL** | Multi-Agent Reinforcement Learning — the *field* |
| **MADDPG** | Multi-Agent Deep Deterministic Policy Gradient — a specific MARL *algorithm* |
| **Actor** | Network mapping state → action |
| **Critic** | Network estimating Q-value (expected future reward) of a state-action |
| **Q-value** | Expected total discounted future reward from taking an action in a state |
| **Off-policy** | Learning from a replay buffer of past experience, not just fresh rollouts |
| **Replay buffer** | Store of past transitions, sampled in mini-batches |
| **Target network** | A slow-moving copy of a network, used to stabilize the TD target |
| **Soft update (τ)** | Nudging a target network slightly toward its live network each step |
| **TD target** | `reward + γ·Q(next state)` — what the critic is trained to predict |
| **γ (gamma)** | Discount factor — how much future reward is worth vs immediate |
| **OU noise** | Ornstein-Uhlenbeck process — temporally correlated exploration noise |
| **Stage gating** | Only one agent is legal to act at each step, set by the action mask |
| **Action mask** | Binary vector marking which discrete actions are legal right now |
| **CTDE** | Centralized Training, Decentralized Execution |
| **Stage-conditioned critic** | Critic told which agent/action is acting, instead of seeing the joint action |
| **Token F1** | Harmonic mean of token precision & recall between answer and gold |

---

## 10. Anticipated defense questions & answers

**Q: Why is this called MADDPG if it isn't textbook MADDPG?**
A: It is MADDPG-*inspired*: per-agent deterministic actors, a centralized
critic, off-policy replay, target networks. It deviates in three ways
(sequential agents, stage-conditioned critic, shared observation), each
motivated by the stage-gated RAG environment. We call it "MADDPG-style" for
exactly that reason.

**Q: Why not just hard-code good RAG parameters?**
A: That is exactly the baseline (`simple_hybrid_rag`). The question is whether
per-query learned control matches or beats a fixed configuration. Our result:
after reward redesign the learned controller reaches a statistical tie on
answer quality (Token F1 0.474 vs 0.487, n=29, gap < 1 SE) and wins the
multi-chunk-synthesis category — parity with a strong fixed baseline.

**Q: Does the trained policy beat the baseline?**
A: Honestly, no — not on aggregate. The best result (v4, 200 ep) is a
*statistical tie* on Token F1 and a win on 2 of 8 categories. We claim parity,
not superiority. The baseline is also faster and cheaper. The contribution is
(a) showing learned continuous control reaches parity with a strong fixed
pipeline, (b) winning synthesis-heavy queries, and (c) the methodological
findings — diagnosing a reward-hacking failure mode and demonstrating
proxy-reward over-optimization. See `docs/RESULTS.md`.

**Q: Why did 500 training episodes do worse than 200?**
A: Proxy-reward over-optimization. The reward is a proxy for answer quality.
With 1,062 gradient updates on 88 training questions the policy kept improving
the *reward* (evidence count climbed) while Token F1 on the held-out set fell —
the proxy and the true metric diverged. 200 episodes is near the optimal
budget; this U-shaped budget curve is itself a reported result.

**Q: Why dense AND sparse retrieval?**
A: Complementary failure modes. Dense matches paraphrase but blurs rare terms;
sparse matches exact terms but fails on paraphrase. Hybrid covers both.

**Q: Why a reranker on top of hybrid retrieval?**
A: Two-stage retrieval. Hybrid is a fast wide net (bi-encoder/​BM25); the
cross-encoder reranker is slow but accurate, so it re-orders only the 20
candidates the first stage produced.

**Q: Why is the critic "stage-conditioned"?**
A: Only one agent acts per step. A joint-action critic would see 4 of 5 action
slots as zero — a wasteful, diluted gradient. Stage-conditioning gives every
training sample the same informative shape.

**Q: How do you know training actually happened?**
A: `total_gradient_updates` is logged (378 for the 200-ep run); critic loss
converged from 0.16 → 0.005; actor losses are stable and negative; the trained
actor's parameter distributions are coherent and clearly non-random. The
`trained` flag in the aggregate JSON is true.

**Q: Is 38 training questions enough?**
A: It is enough to demonstrate the architecture trains and to show a trade-off
result, but not enough for strong generalization claims. The 18-question test
set has high variance. Benchmark expansion is the identified next step.

**Q: What is the discrete action vs the continuous action?**
A: The actor outputs a continuous vector; a mapper turns it into RAG parameters
and *also* picks a discrete action name within the legal masked set. The env
executes the discrete name; the continuous params tune how. The discrete choice
is non-differentiable — the actor learns the continuous part.

---

## 11. Where everything lives (code map)

| Component | Path |
|---|---|
| PDF parse + figure description | `ingestion/watcher.py` |
| Semantic chunking | `ingestion/chunker.py` |
| Embedding + Qdrant indexing | `ingestion/indexer.py` |
| Hybrid retrieval (RRF) | `brain/retriever_shared.py` |
| Retrieval strategies for the agent | `brain/context_marl_ac/adapters/retriever_adapter.py` |
| CrossEncoder reranker | `brain/final_arch/reranker_shared.py` |
| LLM adapter (provider-aware) | `brain/context_marl_ac/adapters/llm_adapter.py`, `brain/llm_config.py` |
| The 5 RAG agents | `brain/context_marl_ac/agents/` |
| Environment + stage gating | `brain/context_marl_ac/marl/marl_env.py`, `action_masking.py` |
| Reward function | `brain/context_marl_ac/marl/reward.py`, `config.py` |
| State features (14-d) + CEB (20-d) | `feature_encoder.py`, `brain/maddpg/context_engineering_block.py` |
| Actors / critic | `brain/maddpg/maddpg_actor.py`, `maddpg_critic.py` |
| Continuous→params mapper | `brain/maddpg/continuous_action_mapper.py` |
| Replay buffer | `brain/maddpg/replay_buffer.py` |
| Trainer (update logic) | `brain/maddpg/trainer.py` |
| Train / evaluate entry points | `brain/maddpg/train_maddpg.py`, `evaluate_maddpg.py` |
| Architecture doc | `docs/MADDPG_EXTENSION.md` |

---

*Read alongside `docs/MADDPG_EXTENSION.md` (architecture detail) and the
deviations discussion (MADDPG vs textbook MADDPG).*
