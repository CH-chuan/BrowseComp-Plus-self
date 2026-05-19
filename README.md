# BrowseComp-Plus — Sampling-Tree Sweep

Follow-up research fork of [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus). The upstream benchmark is preserved; this fork adds a **branching sampling-tree experiment** over a 50-query curated subset, with a multi-retriever search tool (BM25 + Qwen3 dense) the agent can choose between on every `search` call.

The entry point is now the 50-query sweep, not the full 830-query eval.

| [📄 Upstream paper](https://arxiv.org/pdf/2508.06600) | [🤗 Upstream dataset](https://huggingface.co/datasets/Tevatron/browsecomp-plus) | [🔍 Upstream project](https://texttron.github.io/BrowseComp-Plus/) | [📘 Collaborator handoff](HANDOFF.md) |

---

## What's in this fork

- **`topics-qrels/curated50.jsonl`** — 50-query subset of `Tevatron/browsecomp-plus`. Selection rationale and reproduction snippet in `topics-qrels/README.md`.
- **`search_agent/sampling_tree_client.py`** — branching driver. For each query it expands a Best/Worst/Random tree (factor 3, configurable depth, `n_samples` per expansion). Per-call raw responses, parsed tool calls, and cumulative recall/termination metadata are streamed to disk; any path is reproducible from the logs.
- **`search_agent/multi_retriever_tool.py`** + **`configs/retrievers.json`** — single `search` tool that takes a `retriever` enum argument. Ships with `bm25` (Lucene) and `qwen3_dense` (Qwen3-Embedding-0.6B FAISS).
- **`serve_vllm.sh`** — Qwen3 vLLM launcher with the correct tool-call parser per family. Reasoning parser is **off by default** (attaching it in non-thinking mode causes vLLM to swallow `<tool_call>` XML — see comment in the script).
- **`run_full_curated50.sh`** — self-daemonizing wrapper for the full 50-query sweep. Timestamped, resumable run dirs under `output-runs/`.
- **`run_sampling_tree.sh`** — single invocation of the sampling-tree client. Used directly for smoke tests; `run_full_curated50.sh` calls it under the hood.
- **`HANDOFF.md`** — full from-scratch reproduction doc for a collaborator on a different machine.

---

## Quick start

Assumes you've completed the prerequisite setup below (env, indexes, model cache).

```bash
# 1. Start vLLM with Qwen3.5-9B (non-thinking, qwen3_coder tool parser).
nohup ./serve_vllm.sh > logs/vllm.log 2>&1 &
disown
until curl -s localhost:8000/v1/models | grep -q "Qwen/Qwen3.5-9B"; do sleep 5; done

# 2. Launch the 50-query sweep (defaults: depth=10, n_samples=10).
./run_full_curated50.sh
```

The launcher self-daemonizes and prints the run name, log path, and a resume command. Outputs land in `output-runs/curated50_d10_n10_<utc>/`.

### Common variants

```bash
# Smaller / faster sweep (matches the split run described in HANDOFF.md).
DEPTH=5 N_SAMPLES=5 ./run_full_curated50.sh

# Reverse dataset order — second runner picks up from the tail so two boxes
# meet in the middle. (See HANDOFF.md for the split-run pattern.)
DEPTH=5 N_SAMPLES=5 RUN_PREFIX=curated50_d5_n5_rev ./run_full_curated50.sh --reverse-dataset

# Resume a crashed sweep. RUN_NAME, DEPTH, N_SAMPLES must all match the
# original — defaults are 10/10, so re-pass them if the original wasn't 10/10.
RUN_NAME=curated50_d5_n5_20260511T031500Z DEPTH=5 N_SAMPLES=5 ./run_full_curated50.sh

# Single-query smoke test (skip the full sweep wrapper).
./run_sampling_tree.sh --depth 2 --limit 1
```

### Monitor / stop

```bash
tail -f logs/full_<RUN_NAME>.log
find output-runs/<RUN_NAME> -name summary.json | wc -l   # qids done / 50
pkill -f sampling_tree_client.py                         # stop; resume via RUN_NAME=...
```

---

## Output schema (per `output-runs/<run>/<qid>/`)

- **`meta.json`** — static run config: query, formatted query, gold docids, model + sampling params, tool definitions as offered to the LLM, configured retrievers.
- **`samples.jsonl`** — one line per LLM call. **All** `n_samples` per expansion are saved; the 3 picked nodes (Best/Worst/Random) get `node_label` stamped, unpicked siblings get `node_label: null`. Each line has the full `raw_response`, parsed `tool_calls` (with docids + short snippet truncations), `has_tool_call`, `cumulative_if_picked` (recall etc.), and `would_termination_reason`. Line 0 is a root pseudo-line with the user prompt.
- **`expansions.jsonl`** — one line per expanded node. Records `parent_label`, `eligible_indices` (samples with a tool call), and `picks: {B,W,R: {sample_idx, among_ties}}`. Indexes into `samples.jsonl` by `(parent_label, sample_idx)`.
- **`summary.json`** — written at end of qid. Aggregate stats + a leaves index (recomputable from the JSONL files).

Resume granularity is per expansion: if a qid crashed mid-expansion, orphan samples without an expansion line are dropped and that node is re-sampled.

---

## Prerequisite setup

### Python env + Java

`uv` with Python 3.10:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
uv pip install --no-build-isolation flash-attn   # required for FAISS path
```

Java 21 for pyserini/BM25:

```bash
conda install -c conda-forge openjdk=21
# or: sudo apt install -y openjdk-21-jdk
```

> `run_sampling_tree.sh` and `serve_vllm.sh` have absolute paths to a conda env hardcoded (`/home/cche/miniforge3/envs/browsecomp/...`). Replace these with your equivalents before first launch — `HANDOFF.md` has a one-line `sed` for this.

### Indexes (BM25 + Qwen3 FAISS)

```bash
huggingface-cli download Tevatron/browsecomp-plus-indexes --repo-type=dataset \
  --include="bm25/*" --local-dir ./indexes
huggingface-cli download Tevatron/browsecomp-plus-indexes --repo-type=dataset \
  --include="qwen3-embedding-0.6b/*" --local-dir ./indexes
```

These land at `./indexes/bm25/` and `./indexes/qwen3-embedding-0.6b/`, matching the paths in `configs/retrievers.json`.

### Dataset

`topics-qrels/curated50.jsonl` is already committed — no decryption needed for the 50-query sweep. To decrypt the full 830-query upstream dataset (not required for this fork's entry point):

```bash
pip install datasets
python scripts_build_index/decrypt_dataset.py \
  --output data/browsecomp_plus_decrypted.jsonl \
  --generate-tsv topics-qrels/queries.tsv
```

The corpus itself is unobfuscated:

```python
from datasets import load_dataset
ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
```

---

## Splitting a sweep across two machines

`HANDOFF.md` is the canonical write-up for running the same 50-query sweep on a second machine in reverse order so the two runners meet in the middle. Short version:

- Machine A: `./run_full_curated50.sh`                                       (forward, qid 0 → 49)
- Machine B: `./run_full_curated50.sh --reverse-dataset`                     (reverse, qid 49 → 0)

The two output directories don't collide; merge after both finish.

---

## Upstream BrowseComp-Plus

This fork inherits the upstream benchmark untouched — the original full-corpus evaluation, leaderboard submission flow, retrieval-only eval, and per-vendor agent docs are all still present. The upstream entry points are documented in `docs/` (see [`docs/openai.md`](docs/openai.md), [`docs/anthropic.md`](docs/anthropic.md), [`docs/gemini.md`](docs/gemini.md), [`docs/qwen.md`](docs/qwen.md), [`docs/oss.md`](docs/oss.md), [`docs/search-r1.md`](docs/search-r1.md), [`docs/tongyi.md`](docs/tongyi.md), [`docs/custom_retriever.md`](docs/custom_retriever.md)).

If you only want to reproduce upstream results — not run the sampling-tree experiment — `git log a5446eb~10..a5446eb -- README.md` shows the previous README, or use the upstream repo directly.

---

## Citation (upstream)

```
@article{chen2025BrowseCompPlus,
  title={BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent},
  author={Chen, Zijian and Ma, Xueguang and Zhuang, Shengyao and Nie, Ping and Zou, Kai and Liu, Andrew and Green, Joshua and Patel, Kshama and Meng, Ruoxi and Su, Mingyi and Sharifymoghaddam, Sahel and Li, Yanxi and Hong, Haoran and Shi, Xinyu and Liu, Xuye and Thakur, Nandan and Zhang, Crystina and Gao, Luyu and Chen, Wenhu and Lin, Jimmy},
  year={2025},
  journal={arXiv preprint arXiv:2508.06600}
}
```

## Upstream contact

- Zijian Chen — s42chen@uwaterloo.ca
- Xueguang Ma — x93ma@uwaterloo.ca
- Shengyao Zhuang — s.zhuang@uq.edu.au
