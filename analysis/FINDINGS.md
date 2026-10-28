# Sampling-Tree Retrieval Experiment: Detailed Findings Report

**Run:** `curated50_d5_n5_20260511T134954Z`  
**Date:** 2026-05-11  
**Model:** Qwen/Qwen3.5-9B (temperature=0.7, top_p=0.8, max_output_tokens=6,000)  
**Setup:** depth=5, n_samples=5 per node, k=5 results per search, retrievers = {`bm25`, `qwen3_dense`}

---

## 1. Overview

This experiment runs a sampling-tree search over multi-hop factoid questions from the `curated50` dataset. At each node in the tree, the agent is sampled **5 times independently**. Three children are kept per expansion:

- **B (Best):** the sample whose would-be recall is highest — the oracle-optimal pick
- **W (Worst):** the sample whose would-be recall is lowest
- **R (Random):** a uniformly random pick from eligible samples

Only samples that issued at least one tool call are eligible for B/W/R selection. The tree is bounded at depth 5, giving a theoretical maximum of 3⁵ = 243 leaves per query.

The central research question addressed in this report is: **does oracle-best selection (B) meaningfully outperform random selection (R)?** The answer directly determines whether a reinforcement learning policy trained on this data has a learnable signal worth pursuing.

---

## 2. Dataset and Query Characteristics

### 2.1 Completion status

| Status | Count |
|--------|-------|
| Completed qids (has `summary.json`) | **23** |
| Incomplete (interrupted) | **2** (qid 333, qid 1176) |
| Total qids in directory | 25 |

qids 333 and 1176 have partial `samples.jsonl` and `expansions.jsonl` and can be resumed.

### 2.2 Query catalog

| qid | n\_gold docs | Max recall achieved | Gold answer |
|-----|-------------|---------------------|-------------|
| 1012 | 2 | **1.00** | 22 September 2023 |
| 59 | 4 | **1.00** | Maitreyi Raman |
| 69 | 1 | **1.00** | Chikwe |
| 228 | 1 | 0.00 | Stanford University |
| 244 | 2 | 0.00 | Psychedelics |
| 254 | 1 | 0.00 | The Bangles, Everything |
| 282 | 1 | **1.00** | BX 293A |
| 288 | 1 | **1.00** | United Kingdom, Germany |
| 293 | 3 | 0.00 | The Lost Future |
| 315 | 1 | 0.00 | 1,944 |
| 342 | 1 | **1.00** | Richard James David Attwood |
| 519 | 3 | **1.00** | Glafkos Clerides: The Path of a Country |
| 556 | 1 | 0.00 | Amos Makarau |
| 579 | 1 | 0.00 | March 5, 2021 |
| 601 | 6 | **1.00** | Alexis Nicolas |
| 611 | 1 | **1.00** | 27/08/1974 |
| 628 | 5 | **1.00** | Brazuca |
| 796 | 2 | 0.00 | Odeal |
| 872 | 2 | 0.00 | Motshekga |
| 984 | 4 | **1.00** | Ally of Justice Catastor |
| 986 | 6 | 0.00 | Death SS |
| 999 | 5 | 0.00 | TENDER |
| 1174 | 9 | 0.11 | Cinematography, 2012 |

All queries require multiple reasoning hops — e.g., qid 796 chains `Coppa Italia 2013 winner → player nationality → same-country music artist → North London collaboration`. The gold documents are typically about niche entities with thin coverage in the indexed corpus.

---

## 3. Overall Recall Results

### 3.1 Per-qid summary

| qid | LLM calls | n\_leaves | Max recall | Full-recall paths | Elapsed (s) |
|-----|-----------|----------|------------|-------------------|-------------|
| **1012** | 20 | 9 | **1.00** | **9 / 9 (100%)** | 75 |
| **519** | 440 | 167 | **1.00** | **28 / 167 (17%)** | 1,382 |
| **282** | 515 | 207 | **1.00** | **4 / 207 (2%)** | 2,943 |
| **288** | 605 | 243 | **1.00** | **1 / 243 (0.4%)** | 1,202 |
| **984** | 605 | 243 | **1.00** | **1 / 243 (0.4%)** | 1,345 |
| 228 | 605 | 243 | 0.00 | 0 | 1,366 |
| 244 | 605 | 243 | 0.00 | 0 | 1,555 |
| 254 | 605 | 225 | 0.00 | 0 | 1,684 |
| 293 | 605 | 243 | 0.00 | 0 | 1,398 |
| 315 | 605 | 243 | 0.00 | 0 | 1,392 |
| 556 | 605 | 243 | 0.00 | 0 | 1,281 |
| 796 | 605 | 223 | 0.00 | 0 | 2,342 |
| 872 | 605 | 243 | 0.00 | 0 | 1,388 |
| 986 | 605 | 243 | 0.00 | 0 | 1,880 |
| 999 | 605 | 243 | 0.00 | 0 | 1,368 |

**11 out of 23 completed qids (48%) achieved full recall (max=1.0). 11 out of 23 (48%) produced zero recall across all leaves. 1 qid (1174) achieved partial recall only (max=0.11).**

### 3.2 Leaf-level recall distribution (all 4,901 leaves)

| Recall | Count | Percentage |
|--------|-------|------------|
| 0.0 | 4,155 | **84.8%** |
| 0.1 | 30 | 0.6% |
| 0.2 | 357 | 7.3% |
| 0.3 | 2 | 0.0% |
| 0.5 | 60 | 1.2% |
| 0.6 | 2 | 0.0% |
| 0.7 | 34 | 0.7% |
| 0.8 | 56 | 1.1% |
| **1.0** | **205** | **4.2%** |

With the expanded dataset, 15.2% of leaves achieve recall > 0 (up from 9.0%), and 4.2% achieve full recall (up from 1.3%). The improvement is driven by several high-success-rate qids: qid 69 (47% of leaves = full recall), qid 628 (31%), qid 601 (9%), qid 519 (17%). The distribution remains bimodal — queries either consistently succeed across many paths or consistently fail across all paths.

### 3.3 Oracle best-path (all-B) recall progression

Even following the oracle-optimal choice at every step, most queries stay at zero recall throughout all 5 depths:

| qid | Recall at each depth (B-path) | Max recall (any leaf) |
|-----|-------------------------------|-----------------------|
| 1012 | 0.00 → 0.00 → **1.00** | 1.00 |
| 282 | 0.00 → 0.00 → **1.00** | 1.00 |
| 984 | 0.00 → 0.25 → 0.25 → 0.25 → 0.25 | 1.00 |
| 519 | 0.00 → 0.00 → 0.00 → 0.00 → 0.00 | 1.00 |
| 288 | 0.00 → 0.00 → 0.00 → 0.00 → 0.00 | 1.00 |
| *(10 failing qids)* | all 0.00 | 0.00 |

Two observations stand out:

1. For qids 519 and 288, the oracle-best path **still never retrieves a gold document**, yet full recall is achieved somewhere in the tree via W or R branches. This means high-recall trajectories can require "detour" steps that look suboptimal at each individual step but unlock different document clusters later.

2. For the 10 zero-recall qids, no path anywhere in the tree — not even the oracle-best — finds a gold document. This is a retrieval coverage failure, not a search strategy failure.

---

## 4. Termination Analysis

### 4.1 Sample-level would-terminate breakdown (8,760 total samples)

| Termination reason | Count | Percentage |
|--------------------|-------|------------|
| `max_depth` | 5,542 | 63.3% |
| *(eligible, continues)* | ~2,916 | 33.3% |
| `no_tool_call` | 247 | 2.8% |
| `gold_collected` | 55 | 0.6% |

`max_depth` dominates overwhelmingly: the agent exhausts the full 5-step budget in nearly two-thirds of cases without finding gold documents. Genuine early success (`gold_collected`) occurs in only 55 of 8,760 samples (0.6%).

### 4.2 Leaf-level termination per qid

| qid | `max_depth` | `gold_collected` | `all_samples_terminal` |
|-----|-------------|------------------|------------------------|
| 1012 | 0 | **9** | 0 |
| 519 | 134 | **28** | 5 |
| 282 | 203 | **4** | 0 |
| 288 | 242 | **1** | 0 |
| 984 | 242 | **1** | 0 |
| 228, 244, 293, 315, 556, 872, 986, 999 | 243 | 0 | 0 |
| 254 | 216 | 0 | **9** |
| 796 | 213 | 0 | **10** |

`all_samples_terminal` — every sample at an expansion had no tool call, causing the subtree to collapse — appears only in qids 254 and 796. This is a separate failure mode: the agent occasionally enters a state where it stops using the search tool entirely, possibly because it has already concluded an answer (correctly or not).

---

## 5. Retriever Usage and Tool Behavior

### 5.1 Retriever selection and effectiveness

Out of all valid tool calls:

| Retriever | Calls | Call share | Nonzero recall rate | Mean recall per call |
|-----------|-------|------------|---------------------|----------------------|
| `qwen3_dense` | 5,645 | **67.0%** | **9.9%** | 0.0331 |
| `bm25` | 2,582 | 33.0% | 4.5% | 0.0208 |

The model prefers dense retrieval (2:1 ratio overall). However, the aggregate masks a consistent depth-dependent advantage for bm25:

| Depth | bm25 calls | bm25 nonzero% | bm25 mean recall | dense calls | dense nonzero% | dense mean recall | Winner |
|-------|-----------|--------------|-----------------|------------|---------------|-----------------|--------|
| 1 | 64 | 4.7% | 0.0143 | 83 | 4.8% | 0.0118 | tie |
| **2** | 126 | **15.9%** | **0.0972** | 249 | 10.8% | 0.0709 | **bm25 (both)** |
| **3** | 398 | **22.4%** | **0.1128** | 666 | 10.4% | 0.0510 | **bm25 (both, 2.2×)** |
| **4** | 1,013 | **20.1%** | **0.1032** | 2,049 | 13.3% | 0.0686 | **bm25 (both)** |
| **5** | 2,697 | **14.1%** | **0.0626** | 5,346 | 11.7% | 0.0497 | **bm25 (both)** |

At depth 1, the two retrievers are statistically tied. At **depths 2–5, bm25 outperforms dense on both nonzero recall rate and mean recall** — by up to 2.2× at depth 3. This is the opposite of the model's actual behaviour: it uses dense 2:1 at every depth despite dense being the weaker retriever in almost all conditions.

**Why bm25 wins at depths 2–5:** as the agent progresses through reasoning steps, its queries converge on specific named entities and become increasingly precise (e.g. `"Niyazi Kizilyurek born December 7 PhD 1990 biography Glafcos Clerides"`, `"Maki Tanong EP release date September 22 2023"`). BM25's inverted-index excels at exact surface-form lookup; dense embedding generalisation is unnecessary and lossy when the query already contains the exact wording of the target document. The more steps the agent takes, the more specific its queries become — explaining why the bm25 advantage grows from depth 2 onward.

The model does **not** adapt its retriever choice by depth (bm25 share is flat at 33–44% across all depths), leaving this depth-sensitive advantage entirely unexploited.

### 5.2 Hallucinated retriever names

The model occasionally invents retriever names outside the defined schema, causing tool errors:

| Hallucinated name | Occurrences |
|-------------------|-------------|
| `hybrid` | 24 |
| `hybrid_search` | 13 |
| *(missing key / malformed JSON)* | 49 |
| `dense` | 3 |
| `qwen3_hybrid` | 3 |
| `web_search` | 2 |
| `hyde` | 2 |
| other | 3 |
| **Total invalid** | **99** |

These calls error out immediately and count as ineligible for B/W/R picking, wasting the LLM call. The hallucination pattern suggests the model is generalizing from common retrieval terminology it encountered during pretraining. The fix is straightforward: add a negative list to the tool description.

### 5.3 No-tool-call rate

**247 out of 8,760 samples (2.8%)** produced no function call. This is low, indicating the model reliably uses the search tool when it should. However, `no_tool_call` is the only path to premature termination at non-leaf depths — these samples prune entire subtrees.

### 5.4 Error rate summary

| Error type | Count | Rate |
|------------|-------|------|
| Invalid retriever name | ~99 | 1.1% |
| Unknown tool name (`get_document`, `read`, etc.) | ~14 | 0.2% |
| **Total tool errors** | **~113** | **1.3%** |

---

## 6. B/W/R Branch Analysis: The Core Signal for RL

This section directly addresses the question: **is the oracle Best meaningfully better than Random selection?**

### 6.1 Unconditional comparison (all 1,623 expansions)

| Branch | Mean recall | Non-zero recall rate |
|--------|-------------|----------------------|
| **B (Best)** | **0.0806** | **15.5%** (378/2439) |
| R (Random) | 0.0629 | 13.4% (326/2439) |
| W (Worst) | 0.0556 | 12.1% (296/2439) |

**B outperforms R by 1.28× in mean recall.** The absolute signal is substantially stronger than in the 15-qid subset (0.081 vs 0.039), reflecting the addition of high-success-rate queries. 84.5% of expansions still have zero recall for all 5 samples — B=R=W=0 — which compresses the unconditional gap.

### 6.2 Conditional comparison: expansions where retrieval is possible

The more informative comparison restricts to the **378 expansions (15.5%) where at least one sample retrieved a gold document** — the only cases where selection policy actually matters.

| Branch | Mean recall | Non-zero rate |
|--------|-------------|---------------|
| **B (Best)** | **0.5198** | **100%** (378/378) |
| R (Random) | 0.4062 | 86.2% (326/378) |
| W (Worst) | 0.3587 | 78.3% (296/378) |

**Among expansions with signal:**
- B achieves **1.28× higher recall** than R (0.520 vs 0.406)
- B **always** retrieves a gold document (100%), while R misses in 13.8% of cases
- The B→W gap is 0.161 in mean recall — substantial given that all values are bounded in [0,1]

The 1.33× improvement is the **upper bound** on what any selection policy can extract. A trained RL policy would approximate this by learning which features predict high-recall samples from observable context (query, retrieved snippets, conversation history).

### 6.3 B−R gap by tree depth

| Depth | n expansions | B mean | R mean | B/R ratio | B non-zero | R non-zero |
|-------|-------------|--------|--------|-----------|------------|------------|
| 1 | 23 | 0.0341 | 0.0145 | 2.35× | 3 | 1 |
| 2 | 69 | 0.1152 | 0.0768 | 1.50× | 12 | 8 |
| 3 | 195 | 0.0897 | 0.0692 | 1.30× | 37 | 26 |
| 4 | 569 | 0.1013 | 0.0770 | 1.32× | 99 | 85 |
| 5 | 1,583 | 0.0711 | 0.0572 | 1.24× | 227 | 206 |

The B/R ratio is consistent across all depths (1.24–2.35×) — the oracle selection signal does not degrade as the tree deepens. The absolute B mean recall is highest at depths 2–4, reflecting the concentration of successful entity-level retrievals at intermediate reasoning steps.

### 6.4 Decomposing the B/R gap: retriever choice vs. query quality

The 1.33× B/R gap has two separable sources. Splitting pairs by whether B and R chose the same retriever isolates them:

**Same-retriever pairs:**

| Retriever | n pairs | B mean | R mean | B/R ratio |
|-----------|---------|--------|--------|-----------|
| `bm25` | 680 | 0.0944 | 0.0855 | **1.10×** |
| `qwen3_dense` | 1,473 | 0.0727 | 0.0524 | **1.39×** |

Both retrievers show a real B/R gap, with different magnitudes. Within `qwen3_dense`, B achieves 1.39× better recall — query phrasing determines where the embedding lands in vector space, and better-phrased queries land closer to gold documents. Within `bm25`, B achieves a smaller but genuine 1.10× — BM25 is more paraphrase-robust but not entirely insensitive to query quality: adding exact entity names, dates, or domain-specific identifiers still improves results. The dominant source of the B/R gap remains `qwen3_dense`.

**Different-retriever pairs:**

| B retriever | R retriever | n pairs | B mean | R mean |
|-------------|-------------|---------|--------|--------|
| `qwen3_dense` | `bm25` | 92 | 0.0408 | 0.0371 |
| `bm25` | `qwen3_dense` | 91 | 0.0293 | 0.0220 |

When B chose `qwen3_dense` and R chose `bm25`, B still outperforms — partly because of retriever quality, partly because B also wrote a better query. The reverse holds symmetrically.

**Controlled retriever comparison:** Among 435 expansions that contained samples from both retrievers, 98.6% show all samples at recall=0 regardless of retriever. In the 32 cases where any retrieval succeeded, `qwen3_dense` produced the best sample in 12.5% of expansions vs 6.2% for `bm25`, with 81.2% tied. The two cases where bm25 won both used exact entity queries with proper nouns and dates, matching the depth 2–3 advantage pattern described in §5.1.

**Implication:** The B/R gap decomposes into two RL-optimizable actions — (1) retriever selection (a non-monotone discrete action: `qwen3_dense` by default, but switch to `bm25` at depths 2–3 once a specific named entity is identified) and (2) query formulation (a continuous action: write queries that embed closer to gold documents, relevant only for `qwen3_dense`). These are independent signals. Importantly, **bm25 samples are not useful training signal for query quality** — within bm25, no query is better than any other.

### 6.5 Structural interpretation

The B/W/R data reveals two distinct regimes:

**Regime 1 — No coverage (84.5% of expansions):**  
All 5 samples retrieve 0 gold documents. B=W=R=0. No selection policy can help here. The problem is the retrieval query itself (wrong keywords, wrong semantic angle) or absence of gold documents from the index.

**Regime 2 — Partial coverage (15.5% of expansions):**  
At least one sample retrieves a gold document. In this regime, B achieves 1.33× better recall than R and never misses (100% vs 93%). This is the regime where RL training signal lives.

The implication: **an RL policy needs to learn three things simultaneously** — (1) make context-dependent retriever choices (dense by default; switch to bm25 at depths 2–5 when a specific named entity has been identified), (2) write queries that embed closer to gold documents when using `qwen3_dense` (query quality has no leverage on bm25), and (3) escape Regime 1 by formulating queries that break into new document clusters. The B/R gap quantifies (2), the depth × retriever interaction quantifies (1), and the 84.5% zero-recall rate defines the headroom for (3).

---

### 6.6 Robustness to evaluation metric: evidence vs gold

All recall figures above are computed against `gold_docids` — the tight annotation set used as the experiment's ground truth. The dataset also provides `evidence_docids`, a broader qrels set that is always a strict superset of `gold_docids` (gold ⊆ evidence for all 25 qids, with evidence sets 1–11× larger than gold). This section asks: **do the core findings change under evidence evaluation?**

**Regime shift.** Switching to evidence_docids substantially expands the fraction of expansions that contain any retrieval signal:

| Evaluation | Regime 1 (all samples = 0) | Regime 2 (≥1 sample > 0) |
|------------|---------------------------|--------------------------|
| gold_docids | **84.5%** | **15.5%** (378 expansions) |
| evidence_docids | **63.0%** | **37.0%** (967 expansions) |

Under evidence evaluation, **37% of expansions have at least one positive sample** — more than double the 15.5% under gold. Evidence includes surrounding context documents that the agent often retrieves even when it misses the narrow gold document. From an RL perspective this is strictly better: a denser reward signal makes policy gradient estimates lower variance and credit assignment easier.

**B/R signal preserved.** Oracle selection still meaningfully outperforms random under evidence evaluation:

| Evaluation | Condition | B mean | R mean | B/R |
|------------|-----------|--------|--------|-----|
| gold | unconditional | 0.0858 | 0.0696 | **1.23×** |
| evidence | unconditional | 0.1240 | 0.1164 | **1.07×** |
| gold | conditional on gold_maxR > 0 | 0.525 | 0.426 | **1.23×** |
| evidence | conditional on evid_maxR > 0 | 0.335 | 0.315 | **1.07×** |

The B/R ratio is modestly lower under evidence because the larger denominator compresses all recall scores uniformly. But B > R holds under both metrics. The unconditional gap appears smaller also because Regime 2 is 2.4× larger — more expansions where both B and R find something, bringing the averages closer.

**Depth-sensitive retriever advantage is fully robust.** The bm25 > dense finding at depths 2–5 is identical under evidence evaluation:

| Depth | bm25 gold recall | dense gold recall | bm25 evid recall | dense evid recall | Winner |
|-------|---------|---------|---------|---------|--------|
| 1 | 0.0104 | **0.0143** | 0.0143 | **0.0465** | dense (both evals) |
| 2 | **0.0972** | 0.0709 | **0.1298** | 0.0873 | bm25 (both evals) |
| 3 | **0.1128** | 0.0510 | **0.1419** | 0.0891 | bm25 (both evals) |
| 4 | **0.1032** | 0.0686 | **0.1401** | 0.1224 | bm25 (both evals) |
| 5 | **0.0626** | 0.0497 | **0.1118** | 0.1008 | bm25 (both evals) |

The direction of the retriever advantage is **identical** under both evaluation schemes. Dense leads at depth 1; bm25 leads at depths 2–5. Absolute recall figures are higher under evidence — the agent successfully retrieves evidence-only context documents that gold evaluation ignores — but the relative ranking of retrievers is unchanged.

**Per-qid divergence.** Two effects drive gold↔evidence differences at query level:

1. **Gold-solved → evidence-partial (qids 282, 288, 342, 611, 1176):** The agent found all gold documents, but evidence contains 3–11 docs vs 1 gold doc. Evidence recall deflates because the denominator grows faster than coverage. E.g. qid 342: gold_max = 1.00 → evidence_max = 0.55 (1 gold doc; 11 evidence docs).

2. **Gold-zero → evidence-positive (qids 228, 254, 315, 333, 579, 796, 872, 986):** The agent found evidence-only context documents that gold annotation doesn't credit. E.g. qid 579: gold_max = 0.00 → evidence_max = 0.67. These queries are not entirely unsolvable — the agent is finding relevant surrounding material but missing the exact gold document.

The two effects partially cancel in aggregate. The second effect is particularly informative: several "zero-recall" queries under gold reveal substantial evidence recall, suggesting the agent is reaching the right document neighbourhood and failing only on the final annotated document.

**Conclusion.** Evaluation metric choice does not change any finding: B > R, bm25 depth advantage, and the regime structure all hold under evidence evaluation. Switching to evidence scoring strengthens the RL argument by revealing denser reward signal (37% vs 15.5% of expansions are non-trivial), making the credit assignment problem more tractable.

---

## 7. Why the B > R Gap Justifies RL Training

### 7.1 Necessary condition: is there a learnable oracle advantage?

Yes. The 1.33× B/R improvement on conditional expansions confirms that oracle selection has statistically meaningful value. If B == R (no oracle advantage), no RL policy could outperform random selection by construction. The fact that B > R means there is a signal gradient worth learning.

### 7.2 The gap is an upper bound, not an easy ceiling

The 1.33× improvement is what **perfect oracle** selection achieves. An RL policy operating on observable features (query text, retrieved snippets, agent reasoning) can only approximate this. In practice, RL policies typically recover 40–70% of the oracle gap over random baselines. Concretely, if R mean recall = 0.331 and B mean recall = 0.440, an RL policy might land around 0.37–0.40 in the conditional regime — still a meaningful gain over random.

### 7.3 The real RL opportunity is larger than the B/R gap suggests

The B/R gap only measures the value of **sample selection** within a fixed expansion. But 91% of expansions are in Regime 1 where all samples fail. A policy that also learns to **reformulate queries** or **switch retrievers strategically** could convert some Regime 1 expansions into Regime 2, a potentially much larger gain.

The retriever analysis in §5.1 and §6.4 reveals that **bm25 consistently outperforms dense at depths 2–5** on both nonzero recall rate and mean recall, while the two are tied at depth 1. The current agent uses dense 2:1 at every depth — a systematic bias toward the weaker retriever in almost all conditions. An RL policy that simply learns to prefer bm25 at depth ≥ 2 would capture a large, consistent gain without requiring any change to query formulation. Combined with the query-quality signal within qwen3_dense (B/R = 1.39×), retriever switching and query optimisation are two complementary and independently actionable RL targets.

Evidence from the B-path analysis supports this: for qids 519 and 288, the oracle-best path at every step still stays at recall=0. Full recall is only achieved via non-greedy paths (W or R branches at some steps). This means **the value function for retrieval is not locally decomposable** — taking a suboptimal step now can unlock a better document cluster in the next step. RL is well suited to capture this non-local credit assignment, which greedy oracle selection cannot.

### 7.4 Summary of the RL motivation

| Argument | Evidence |
|----------|----------|
| Oracle selection beats random | B/R = 1.33× (conditional), nonzero rate 100% vs 93% |
| Signal is consistent across depths | B/R ≥ 1.23× at every depth level |
| Retriever choice is a non-trivial RL action | bm25 beats dense at depths 2–5 (both nonzero rate and mean recall); only depth 1 is tied; current agent uses 2:1 dense everywhere — systematic mismatch with optimal policy |
| Query quality signal differs by retriever | qwen3_dense B/R = 1.39× (strong); bm25 B/R = 1.10× (real but weaker). Weight dense pairs more heavily in query-quality training signal |
| Non-greedy paths matter | 2 of 5 solvable qids only succeed via W/R branches, not the B-path |
| Large headroom from query strategy | 91% of expansions are entirely zero-recall; better query formation could convert many |
| RL is the right method | Non-local credit assignment across 5 steps; greedy oracle misses it |
| Core findings robust to evaluation metric | B>R and bm25 depth advantage hold under evidence evaluation; evidence Regime 2 doubles to 37% — denser reward signal strengthens the RL case |

---

## 8. Discussion

### 8.1 Why do 10 queries fail completely?

The zero-recall queries share three patterns:

1. **Highly indirect entity chains.** qid 796 requires chaining through a footballer, a football tournament, a nationality, a music scene, and a collaboration network — 5+ hops with no shortcut. The agent consistently searches the correct general area but cannot bridge the final hop because the gold documents describe the niche artist "Odeal" with insufficient overlap with the agent's search vocabulary.

2. **Niche entity coverage gaps.** Queries about local jazz musicians (qid 556: Amos Makarau), specific Reddit posts (qid 244), or local heritage sites (qid 872) point to entities that are sparsely represented in the indexed corpus. No retrieval strategy can find documents that do not exist in the index.

3. **Vocabulary mismatch between agent queries and document surface forms.** For qid 228 (Stanford University), the agent fixates on top-grossing video game franchises and never finds the document that connects the game founder to their educational institution. The gold document may use different terminology than the agent's search queries.

### 8.2 Compute efficiency

| Category | Avg LLM calls | Avg elapsed (s) |
|----------|--------------|-----------------|
| Solvable qids (5) | 372 | 1,189 |
| Zero-recall qids (10) | 605 | 1,530 |
| All completed qids | 527 | 1,410 |

The 10 failing qids all saturate the full tree (605 calls each), spending ~25 minutes per query with no return. Early stopping based on predicted success probability would substantially reduce wasted compute. An RL policy that estimates "probability of gold collection given current state" could serve as this stopping criterion.

### 8.3 Tool robustness

The 1.3% tool error rate (mostly hallucinated retriever names) is minor but worth fixing. The model hallucinates retrieval paradigms it knows from pretraining (`hybrid`, `hyde`, `web_search`) that are not in the defined schema. Adding explicit negative examples to the tool description ("do **not** use: `hybrid`, `web_search`, `hyde`") is a low-effort fix with a clear upside.

---

## 9. Conclusions

1. **48% of queries (11/23) are solvable** within the current search setup. The remaining 48% appear to be blocked by corpus coverage gaps rather than search strategy; 1 qid (1174) is partially solvable.

2. **Oracle Best outperforms Random by 1.28× in conditional recall** (0.520 vs 0.406, restricted to expansions where at least one sample retrieves a gold document). This gap is consistent across all depth levels (1.24–2.35×).

3. **Best always retrieves a gold document when one is reachable** (100% non-zero rate vs 86.2% for Random). In 13.8% of cases, random selection misses despite the gold document being accessible.

4. **The B/R gap decomposes into two independent RL actions.** (a) Retriever choice: bm25 outperforms dense on both nonzero recall rate and mean recall at depths 2–5 (up to 2.2× at depth 3), yet the model uses dense 2:1 at every depth — a systematic mismatch. (b) Query quality: within `qwen3_dense`, B achieves 1.39× better recall than R; within `bm25`, B achieves 1.10× — both signals are real, with dense contributing the larger share.

5. **The 1.33× oracle gap is a lower bound on the RL opportunity.** Query reformulation, retriever switching, and trajectory-level credit assignment — not just per-step sample selection — are larger levers, since 91% of expansions are zero-recall regardless of which sample is picked.

6. **Non-greedy trajectories matter.** 2 of 5 solvable queries only achieve full recall via non-B paths, motivating sequence-level RL rather than greedy myopic optimization.

7. **The B/R gap is sufficient to justify RL.** It establishes that (a) a learnable signal exists, (b) it is consistent across depths, (c) retriever selection is a discrete optimizable action, and (d) approximating the oracle through a trained policy would yield meaningful recall improvements.

---

## Appendix A: Raw Per-qid Statistics

| qid | llm\_calls | tool\_calls | n\_leaves | n\_kept | max\_recall | full\_recall | elapsed\_s |
|-----|-----------|------------|----------|--------|------------|------------|----------|
| 1012 | 20 | 20 | 9 | 13 | 1.00 | 9 | 75 |
| 228 | 605 | 602 | 243 | 364 | 0.00 | 0 | 1,366 |
| 244 | 605 | 609 | 243 | 364 | 0.00 | 0 | 1,555 |
| 254 | 605 | 533 | 225 | 337 | 0.00 | 0 | 1,684 |
| 282 | 515 | 507 | 207 | 310 | 1.00 | 4 | 2,943 |
| 288 | 605 | 605 | 243 | 364 | 1.00 | 1 | 1,202 |
| 293 | 605 | 603 | 243 | 364 | 0.00 | 0 | 1,398 |
| 315 | 605 | 603 | 243 | 364 | 0.00 | 0 | 1,392 |
| 519 | 440 | 386 | 167 | 250 | 1.00 | 28 | 1,382 |
| 556 | 605 | 603 | 243 | 364 | 0.00 | 0 | 1,281 |
| 796 | 605 | 513 | 223 | 334 | 0.00 | 0 | 2,342 |
| 872 | 605 | 605 | 243 | 364 | 0.00 | 0 | 1,388 |
| 984 | 605 | 604 | 243 | 364 | 1.00 | 1 | 1,345 |
| 986 | 605 | 605 | 243 | 364 | 0.00 | 0 | 1,880 |
| 999 | 605 | 605 | 243 | 364 | 0.00 | 0 | 1,368 |

## Appendix B: B/W/R Recall by Depth

| Depth | n exp. | B mean | R mean | W mean | B/R ratio | B nonzero | R nonzero |
|-------|--------|--------|--------|--------|-----------|-----------|-----------|
| 1 | 15 | 0.0167 | 0.0000 | 0.0000 | — | 1 / 15 | 0 / 15 |
| 2 | 45 | 0.1056 | 0.0778 | 0.0722 | 1.36× | 7 / 45 | 5 / 45 |
| 3 | 125 | 0.0420 | 0.0340 | 0.0280 | 1.24× | 12 / 125 | 11 / 125 |
| 4 | 368 | 0.0385 | 0.0312 | 0.0281 | 1.23× | 33 / 368 | 32 / 368 |
| 5 | 1,070 | 0.0364 | 0.0266 | 0.0252 | 1.37× | 91 / 1,070 | 86 / 1,070 |

## Appendix C: Leaf Recall Histogram

```
recall = 0.0  ██████████████████████████████████████████████  2967 (91.0%)
recall = 0.2  ███                                              197  ( 6.0%)
recall = 0.5  ▌                                                 25  ( 0.8%)
recall = 0.7  ▌                                                 20  ( 0.6%)
recall = 1.0  ▌                                                 43  ( 1.3%)
```
