# Handoff: Sampling-Tree Sweep on `curated50`

Hey — running out of time on my end. **Please run the same 50-query sweep on your machine in reverse order, so we meet in the middle.** This doc covers full setup from scratch.

## TL;DR

- I'm running 50 queries (`topics-qrels/curated50.jsonl`) forward (qid index 0 → 49) on my box at `depth=5, n_samples=5`.
- You run the same 50 in **reverse** (qid index 49 → 0) on your box. The new `--reverse-dataset` flag handles that.
- We each get our own `output-runs/<run-name>/` directory; nothing collides.

## What this experiment does

For each query we expand a B/W/R sampling tree (factor 3, depth 5, 5 samples per expansion) using `Qwen/Qwen3.5-9B` as the agent. The agent picks between `bm25` and `qwen3_dense` retrievers on each `search` call. Per-call raw responses and tool results are streamed to disk so any path can be reproduced; details in `search_agent/sampling_tree_client.py` and the per-qid output schema described below.

---

## Hardware

- 1 NVIDIA GPU with ≥ 24 GB VRAM (Qwen3.5-9B at bf16 in eager mode @ `--gpu-memory-utilization 0.92` fits).
- ~80 GB disk free (models + FAISS indexes + output).
- CUDA 12.x toolkit + driver.

---

## Setup (one-time)

### 1. Clone & checkout the branch I've been working on

```bash
git clone <this-repo-url> BrowseComp-Plus-self
cd BrowseComp-Plus-self
git checkout main          # branch with the sampling-tree work; latest commit is what you want
```

### 2. Python env + Java (per the upstream README)

`uv` with python 3.10 is what the project uses:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
uv pip install --no-build-isolation flash-attn    # required for FAISS path
```

Java 21 for pyserini / BM25:

```bash
conda install -c conda-forge openjdk=21
```

Make a note of where conda put it — you'll plug it into `JAVA_HOME` / `JVM_PATH` below.

### 3. HF_HOME for model + embedder caches

Pick a disk with ~40 GB free; this is where Qwen3.5-9B (the agent) and Qwen3-Embedding-0.6B (the dense retriever) will be downloaded.

```bash
export HF_HOME=/path/to/your/models   # also set this in your shell rc so vLLM picks it up
```

### 4. BrowseComp+ dataset

`topics-qrels/curated50.jsonl` is already in the repo (50-query curated subset). You don't need to decrypt the full 830-query set for this sweep. If you want it later:

```bash
python scripts_build_index/decrypt_dataset.py \
  --output data/browsecomp_plus_decrypted.jsonl \
  --generate-tsv topics-qrels/queries.tsv
# may prompt for huggingface-cli login
```

### 5. Indexes

You need BM25 (Lucene) and Qwen3-Embedding-0.6B FAISS. Download both:

```bash
huggingface-cli download Tevatron/browsecomp-plus-indexes --repo-type=dataset --include="bm25/*"                    --local-dir ./indexes
huggingface-cli download Tevatron/browsecomp-plus-indexes --repo-type=dataset --include="qwen3-embedding-0.6b/*" --local-dir ./indexes
```

These land at `./indexes/bm25/` and `./indexes/qwen3-embedding-0.6b/`, which is what `configs/retrievers.json` already points to.

### 6. Adjust hardcoded paths

Both shell scripts have my home directory baked in. Search for `/home/cche/` and replace:

- `serve_vllm.sh`: replace `/home/cche/miniforge3/envs/browsecomp/lib` and `/home/cche/miniforge3/envs/browsecomp/bin` with **your** conda env paths (or just adjust to point at the venv's `vllm` binary if you used uv instead of conda).
- `run_sampling_tree.sh`: replace `JAVA_HOME`, `JVM_PATH`, `LD_LIBRARY_PATH`, `PATH`, and the python binary path with your equivalents.

Quick `sed` if your conda env is at `/home/<you>/miniforge3/envs/browsecomp`:

```bash
sed -i 's|/home/cche|/home/<you>|g' serve_vllm.sh run_sampling_tree.sh
```

---

## Verify setup

```bash
# 1) Python imports
python -c "import openai, vllm, pyserini, faiss, transformers; print('ok')"

# 2) Indexes exist
ls indexes/bm25/ indexes/qwen3-embedding-0.6b/

# 3) Dataset readable
head -1 topics-qrels/curated50.jsonl | python -m json.tool | head -5
```

---

## Launch the reverse-order sweep

### Step 1: start vLLM (Qwen3.5-9B, qwen3_coder tool parser, **no reasoning parser**)

```bash
nohup ./serve_vllm.sh > logs/vllm.log 2>&1 &
disown
```

Wait until it's ready (~2-3 min for model load):

```bash
until curl -s localhost:8000/v1/models | grep -q "Qwen/Qwen3.5-9B"; do sleep 5; done
echo "vLLM ready"
```

**Important — do NOT attach `--reasoning-parser qwen3`** while running in non-thinking mode. It causes vLLM to bundle the model's whole second-turn output (including the embedded `<tool_call>` XML) into a single "reasoning" item, so the tool-call parser never sees the XML and tool calls vanish. This is the default behavior of `serve_vllm.sh` after our last fix.

### Step 2: launch the reverse-order sweep

```bash
DEPTH=5 N_SAMPLES=5 RUN_PREFIX=curated50_d5_n5_rev ./run_full_curated50.sh --reverse-dataset
```

That script self-daemonizes: it prints the PID, log path, and resume command, then exits. Safe to close your shell.

You'll see something like:

```
=== full curated50 sweep (background) ===
  run_name:       curated50_d5_n5_rev_<utc_timestamp>
  output dir:     output-runs/curated50_d5_n5_rev_<utc_timestamp>/
  log:            logs/full_<run_name>.log
  ...
  resume later:   RUN_NAME=<run_name> ./run_full_curated50.sh --reverse-dataset
```

Save that `resume later:` line — passing the same `RUN_NAME` env var picks up where it left off if the daemon dies.

### Step 3: monitor

```bash
tail -f logs/full_curated50_d5_n5_rev_*.log
find output-runs/curated50_d5_n5_rev_* -name summary.json | wc -l   # qids done / 50
ls   output-runs/curated50_d5_n5_rev_*/ | wc -l                     # qids started
du -sh output-runs/curated50_d5_n5_rev_*                            # disk used
```

To stop: `pkill -f sampling_tree_client.py`. Resume with the `RUN_NAME=...` env var.

---

## Output schema (per qid)

Each `output-runs/<run>/<qid>/` contains:

- **`meta.json`** — static run config: query, formatted_query, gold_docids, model, request params (temperature/top_p/max_tokens), tool_definitions exactly as offered to the LLM, configured retrievers.
- **`samples.jsonl`** — one line per LLM call. **ALL** 5 samples per expansion saved (3 picked B/W/R get `node_label` stamped; the other 2 have `node_label: null`). Each line has the full `raw_response`, the parsed `tool_calls` (with docids + 30-char snippet truncations), `has_tool_call`, `cumulative_if_picked` (recall etc.), and `would_termination_reason`. Plus a line-0 root pseudo-line with the user prompt.
- **`expansions.jsonl`** — one line per expanded node. Records `parent_label`, `eligible_indices` (samples with a tool call), and `picks: {B,W,R: {sample_idx, among_ties}}`. Picks index into `samples.jsonl` by `(parent_label, sample_idx)`.
- **`summary.json`** — written at end. Stats + leaves index (recomputable from the JSONL files).

Resume granularity: per expansion. If a qid crashed mid-expansion (samples without an expansion line), orphan samples are dropped and that node is re-sampled.

---

## Troubleshooting

- **`vllm serve` exits immediately, "CUDA out of memory"**: lower GPU util via `GPU_UTIL=0.85 ./serve_vllm.sh`, or `MAX_LEN=12288 ./serve_vllm.sh` if your card is tight.
- **`pyserini` import fails with JVM error**: `JAVA_HOME` not set or wrong Java version. Check `java --version` shows 21.
- **`step-2 has_tool_call=0/N`**: `--reasoning-parser qwen3` is attached in your vLLM launch — drop it. (Set `REASONING_PARSER=` empty in `serve_vllm.sh`.)
- **vLLM 400 with "model not found"**: the model name in `run_full_curated50.sh` (`Qwen/Qwen3.5-9B`) doesn't match what vLLM is serving. Check `curl -s localhost:8000/v1/models`.
- **Sweep dies with "max_output_tokens > max_model_len"**: vLLM was launched with `--max-model-len 8192` (default for non-Qwen3.5 models) but the sweep is asking for `--max-tokens 6000`. Either lower max-tokens or raise `MAX_LEN=16384 ./serve_vllm.sh`.

---

## Pointer to my run state (just so you know what NOT to duplicate)

I'm running on my machine with:
- `RUN_NAME=curated50_d5_n5_<my_timestamp>`  (forward order, no `--reverse-dataset`)
- Output dir: `output-runs/curated50_d5_n5_<my_timestamp>/`
- Started at ~2026-05-11 13:50 UTC

Your reverse-order run will be in a completely separate directory, so we won't collide. When we're both done, I'll merge the result dirs.

---

Questions: ping me. The fastest debug is to (1) confirm vLLM is up + model name matches, (2) check the latest line of `logs/full_*.log`, (3) inspect `samples.jsonl` of the qid currently being worked on.
