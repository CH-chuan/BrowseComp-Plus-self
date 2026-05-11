"""
Sampling-tree client: explore agent trajectories as a tree.

At each step (one assistant turn = one tool call cycle), we sample N candidate
continuations from the LLM (using temperature for diversity), score each by the
cumulative gold-doc recall achieved by the resulting path, and branch into
three children:

  - B (best):   the sample whose path now has the highest cumulative gold recall.
  - W (worst):  the sample with the lowest cumulative gold recall.
  - R (random): one sample drawn uniformly at random.

Eligibility rule: samples whose LLM response contained no function_call (the
agent's final answer) are excluded from B/W/R selection. They are still logged
to samples.jsonl with node_label=null.

Each of the three eligible children then expands the same way at the next step,
giving a fully branching tree of factor 3. The tree grows until either:

  - the configured max depth is reached, or
  - the path has retrieved every gold doc for the query, or
  - none of the 10 samples at a node produced a tool call.

Output layout (run-namespaced, per-qid folder, streaming JSONL):

  {output-dir}/{run-name}/{qid}/
    meta.json          static run config + tool definitions + gold info
    samples.jsonl      one line per LLM call; ALL 10 per expansion (3 kept + 7 unkept)
    expansions.jsonl   one line per expanded node; the picks index
    summary.json       written at qid end; stats + leaves index

Resume: passing the same --run-name re-uses the folder. Already-completed
expansions (parent line in expansions.jsonl) are loaded from disk; the BFS
short-circuits past them without LLM calls.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai
from prompts import format_query
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_retriever_tool import MultiRetrieverToolHandler, load_retrievers_config
from oss_client import SearchToolHandler
from searcher.searchers import SearcherType


# How many chars of each doc snippet to keep when writing sample lines to disk.
# Set in main() from --save-snippet-chars. The LLM is still served the full
# snippet (controlled by --snippet-max-tokens); only the on-disk record is
# truncated. NOTE: resume from disk will see truncated snippets in the
# reconstructed function_call_output → not byte-equivalent to a fresh run.
_SAVE_SNIPPET_MAX_CHARS: Optional[int] = 30


def _normalize_docs(value) -> List[str]:
    """gold_docs/evidence_docs come either as list[str] (curated50.jsonl) or
    list[{"docid": str, "text": ...}] (browsecomp_plus_decrypted.jsonl)."""
    if not value:
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, dict) and "docid" in item:
            out.append(str(item["docid"]))
        else:
            out.append(str(item))
    return out


def load_dataset_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load BrowseComp+ JSONL (curated50.jsonl or browsecomp_plus_decrypted.jsonl).
    Returns a list of {query_id, query, gold_docids: set, evidence_docids: set, answer}."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset JSONL not found: {path}")
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append({
                "query_id": str(d.get("query_id")),
                "query": d.get("query", ""),
                "answer": d.get("answer"),
                "gold_docids": set(_normalize_docs(d.get("gold_docs"))),
                "evidence_docids": set(_normalize_docs(d.get("evidence_docs"))),
            })
    return records


@dataclass
class Node:
    """In-memory only; not the serialization format. Used to carry per-step
    state forward across BFS layers for picked children."""
    path_label: str
    step: int
    messages: List[Dict[str, Any]]
    cumulative_retrieved: set = field(default_factory=set)
    cumulative_golds: set = field(default_factory=set)
    recall: float = 0.0
    terminated: bool = False
    termination_reason: Optional[str] = None
    children: List["Node"] = field(default_factory=list)


def _sample_one(
    client: openai.OpenAI,
    base_request: dict,
    messages: List[dict],
    tool_handler,
    sample_idx: int,
    gold_set: set,
    parent_cum_retrieved: set,
    parent_step: int,
    max_depth: int,
) -> dict:
    """Run one continuation: ask the LLM for one assistant turn, then execute
    any tool calls it makes. Returns all the fields needed to serialize a
    sample line and to construct a child Node."""
    request = copy.deepcopy(base_request)
    request["input"] = messages

    try:
        response = client.responses.create(**request)
    except Exception as e:
        cum_retrieved = set(parent_cum_retrieved)
        cum_golds = cum_retrieved & gold_set
        recall = (len(cum_golds) / len(gold_set)) if gold_set else 0.0
        return {
            "sample_idx": sample_idx,
            "raw_response": None,
            "tool_calls": [],
            "tool_error": f"{type(e).__name__}: {e}",
            "has_tool_call": False,
            "retrieved_this_step": [],
            "new_messages": [],
            "cum_retrieved": cum_retrieved,
            "cum_golds": cum_golds,
            "recall": recall,
            "would_terminate": True,
            "would_termination_reason": "llm_error",
        }

    response_dict = response.model_dump(mode="python")
    output = response_dict.get("output") or []

    new_messages: List[dict] = []
    tool_calls: List[dict] = []
    retrieved_this_step: List[str] = []
    tool_error: Optional[str] = None
    has_tool_call = False

    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "reasoning":
            # Qwen3 multi-turn guidance: do NOT include thinking content in subsequent
            # turns' input. raw_response still preserves it for inspection.
            continue
        if item.get("type") == "function_call":
            has_tool_call = True
            new_messages.append(item)
            name = item.get("name")
            args_str = item.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
            except Exception:
                args = {}
            try:
                result_str = tool_handler.execute_tool(name, args)
            except Exception as e:
                err = f"Error executing {name}: {e}"
                new_messages.append(
                    {"type": "function_call_output", "call_id": item.get("call_id"), "output": err}
                )
                tool_calls.append({
                    "call_id": item.get("call_id"),
                    "name": name,
                    "arguments": args if isinstance(args, dict) else {},
                    "result": [],
                    "error": str(e),
                })
                tool_error = (tool_error or "") + err + "; "
                continue
            output_str = result_str if isinstance(result_str, str) else json.dumps(result_str)
            try:
                result = json.loads(output_str) if isinstance(output_str, str) else output_str
                if not isinstance(result, list):
                    result = []
            except Exception:
                result = []
            new_messages.append(
                {"type": "function_call_output", "call_id": item.get("call_id"), "output": output_str}
            )
            tool_calls.append({
                "call_id": item.get("call_id"),
                "name": name,
                "arguments": args if isinstance(args, dict) else {},
                "result": result,
            })
            for entry in result:
                if isinstance(entry, dict) and "docid" in entry:
                    retrieved_this_step.append(str(entry["docid"]))
        else:
            new_messages.append(item)

    cum_retrieved = set(parent_cum_retrieved) | set(retrieved_this_step)
    cum_golds = cum_retrieved & gold_set
    recall = (len(cum_golds) / len(gold_set)) if gold_set else 0.0

    step = parent_step + 1
    would_terminate = False
    would_reason: Optional[str] = None
    if not has_tool_call:
        would_terminate = True
        would_reason = "no_tool_call"
    elif gold_set and cum_golds == gold_set:
        would_terminate = True
        would_reason = "gold_collected"
    elif step >= max_depth:
        would_terminate = True
        would_reason = "max_depth"

    return {
        "sample_idx": sample_idx,
        "raw_response": response_dict,
        "tool_calls": tool_calls,
        "tool_error": tool_error,
        "has_tool_call": has_tool_call,
        "retrieved_this_step": retrieved_this_step,
        "new_messages": new_messages,
        "cum_retrieved": cum_retrieved,
        "cum_golds": cum_golds,
        "recall": recall,
        "would_terminate": would_terminate,
        "would_termination_reason": would_reason,
    }


# ---------------------------------------------------------------------------
# Streaming writers
# ---------------------------------------------------------------------------

def _write_meta(qid_dir: Path, qid: str, qtext: str, formatted_query: str,
                gold_set: set, evidence_set: set, answer: Any,
                args, tool_definitions, retriever_metadata, searcher_type: Optional[str],
                run_name: str, started_at: str) -> None:
    meta = {
        "query_id": qid,
        "run_name": run_name,
        "started_at": started_at,
        "query": qtext,
        "formatted_query": formatted_query,
        "answer": answer,
        "model": args.model,
        "request": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_tokens,
            "truncation": "auto",
        },
        "tool_definitions": tool_definitions,
        "searcher_type": searcher_type,
        "configured_retrievers": retriever_metadata,
        "experiment": {
            "n_samples": args.n_samples,
            "depth": args.depth,
            "seed": args.seed,
            "k": args.k,
            "snippet_max_tokens": args.snippet_max_tokens,
            "query_template": args.query_template,
            "dataset": args.dataset,
            "gold_source": args.gold_source,
        },
        "gold_docids": sorted(gold_set),
        "evidence_docids": sorted(evidence_set),
        "n_gold": len(gold_set),
    }
    (qid_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def _write_root_line(fp, formatted_query: str) -> None:
    rec = {
        "kind": "root",
        "parent_label": None,
        "step": 0,
        "messages_root": [{"role": "user", "content": formatted_query}],
        "cumulative": {"retrieved": [], "retrieved_count": 0, "golds": [], "recall": 0.0},
    }
    fp.write(json.dumps(rec))
    fp.write("\n")
    fp.flush()


def _truncate_snippets_for_disk(tool_calls, max_chars: Optional[int]):
    """Return a copy of tool_calls with each result[].snippet truncated to
    `max_chars` chars. None means no truncation."""
    if max_chars is None or max_chars < 0 or not tool_calls:
        return tool_calls
    out = []
    for tc in tool_calls:
        tc2 = dict(tc)
        result2 = []
        for r in tc.get("result") or []:
            r2 = dict(r)
            s = r2.get("snippet")
            if isinstance(s, str) and len(s) > max_chars:
                r2["snippet"] = s[:max_chars]
            result2.append(r2)
        tc2["result"] = result2
        out.append(tc2)
    return out


def _write_sample_line(fp, parent_label: str, step: int, sample_idx: int,
                       sample: dict,
                       picked_label: Optional[str],
                       picked_among_ties: Optional[bool]) -> None:
    node_label = (parent_label + picked_label) if picked_label else None
    rec = {
        "kind": "sample",
        "parent_label": parent_label,
        "sample_idx": sample_idx,
        "step": step,
        "node_label": node_label,
        "branch": picked_label,
        "among_ties": picked_among_ties if picked_label else None,
        "raw_response": sample.get("raw_response"),
        "tool_error": sample.get("tool_error"),
        "tool_calls": _truncate_snippets_for_disk(
            sample.get("tool_calls"), _SAVE_SNIPPET_MAX_CHARS
        ),
        "has_tool_call": sample.get("has_tool_call", False),
        "retrieved_this_step": list(sample.get("retrieved_this_step") or []),
        "cumulative_if_picked": {
            "retrieved": sorted(sample.get("cum_retrieved") or []),
            "retrieved_count": len(sample.get("cum_retrieved") or []),
            "golds": sorted(sample.get("cum_golds") or []),
            "recall": sample.get("recall", 0.0),
        },
        "would_terminate": sample.get("would_terminate", False),
        "would_termination_reason": sample.get("would_termination_reason"),
    }
    fp.write(json.dumps(rec, default=str))
    fp.write("\n")
    fp.flush()


def _write_expansion_line(fp, parent_label: str, step: int,
                          eligible_indices: List[int],
                          picks: Optional[Dict[str, Dict[str, Any]]]) -> None:
    rec = {
        "parent_label": parent_label,
        "step": step,
        "eligible_indices": list(eligible_indices),
        "n_eligible": len(eligible_indices),
        "picks": picks,
    }
    fp.write(json.dumps(rec))
    fp.write("\n")
    fp.flush()


def _walk_tree(root: Node) -> Tuple[List[Node], List[Node]]:
    nodes: List[Node] = []
    leaves: List[Node] = []
    stack = [root]
    while stack:
        n = stack.pop()
        nodes.append(n)
        if n.children:
            stack.extend(n.children)
        else:
            leaves.append(n)
    return nodes, leaves


def _write_summary(qid_dir: Path, qid: str, started_at: str, elapsed_sec: float,
                   samples_path: Path, root: Node) -> None:
    nodes, leaves = _walk_tree(root)
    leaves.sort(key=lambda n: (len(n.path_label), n.path_label))

    n_llm_calls = 0
    n_tool_calls = 0
    with samples_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("kind") == "sample":
                n_llm_calls += 1
                n_tool_calls += len(r.get("tool_calls") or [])

    max_recall = max((l.recall for l in leaves), default=0.0)
    n_full = sum(1 for l in leaves if l.recall >= 1.0 - 1e-9)

    summary = {
        "query_id": qid,
        "started_at": started_at,
        "completed_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stats": {
            "llm_calls": n_llm_calls,
            "tool_calls": n_tool_calls,
            "elapsed_sec": round(elapsed_sec, 2),
            "n_leaves": len(leaves),
            "n_kept_nodes": len(nodes),
            "max_recall_leaf": max_recall,
            "n_paths_full_recall": n_full,
        },
        "leaves": [
            {
                "node_label": l.path_label,
                "depth": l.step,
                "terminated": l.terminated,
                "termination_reason": l.termination_reason,
                "recall": l.recall,
                "cumulative_golds": sorted(l.cumulative_golds),
                "cumulative_retrieved_count": len(l.cumulative_retrieved),
            }
            for l in leaves
        ],
    }
    (qid_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def _load_resume_state(qid_dir: Path) -> Optional[Dict[str, Any]]:
    """Read existing samples.jsonl / expansions.jsonl. Drop orphan sample groups
    (parents with samples but no matching expansion line) by rewriting
    samples.jsonl. Return None if there's no resume state."""
    samples_path = qid_dir / "samples.jsonl"
    exp_path = qid_dir / "expansions.jsonl"
    if not samples_path.exists() and not exp_path.exists():
        return None

    completed_expansions: Dict[str, dict] = {}
    if exp_path.exists():
        with exp_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    exp = json.loads(line)
                except Exception:
                    continue
                completed_expansions[exp["parent_label"]] = exp

    samples_by_parent: Dict[str, Dict[int, dict]] = {}
    root_record: Optional[dict] = None
    if samples_path.exists():
        with samples_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("kind") == "root":
                    root_record = rec
                    continue
                if rec.get("kind") == "sample":
                    samples_by_parent.setdefault(rec["parent_label"], {})[rec["sample_idx"]] = rec

    orphans = set(samples_by_parent) - set(completed_expansions)
    if orphans:
        good: List[dict] = []
        if root_record is not None:
            good.append(root_record)
        for parent in completed_expansions:
            for idx in sorted(samples_by_parent.get(parent, {})):
                good.append(samples_by_parent[parent][idx])
        with samples_path.open("w") as f:
            for rec in good:
                f.write(json.dumps(rec, default=str))
                f.write("\n")
        for p in orphans:
            samples_by_parent.pop(p, None)

    return {
        "completed_expansions": completed_expansions,
        "samples_by_parent": samples_by_parent,
        "root_record": root_record,
    }


def _reconstruct_new_messages(sample_rec: dict) -> List[dict]:
    """From a saved sample line, rebuild the message-delta that would have been
    appended to parent.messages after this sample's LLM call.

    Mirrors _sample_one's policy: drop reasoning items (Qwen3 multi-turn guidance)."""
    new_messages: List[dict] = []
    output = (sample_rec.get("raw_response") or {}).get("output") or []
    tc_by_id = {tc.get("call_id"): tc for tc in (sample_rec.get("tool_calls") or [])}
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "reasoning":
            continue
        new_messages.append(item)
        if item.get("type") == "function_call":
            cid = item.get("call_id")
            tc = tc_by_id.get(cid)
            if tc is None:
                continue
            result = tc.get("result") or []
            output_str = json.dumps(result, indent=2)
            new_messages.append({"type": "function_call_output", "call_id": cid, "output": output_str})
    return new_messages


def _reconstruct_child_from_saved(parent: Node, sample_rec: dict, branch: str,
                                  among_ties: bool, gold_set: set, max_depth: int) -> Node:
    new_messages = _reconstruct_new_messages(sample_rec)
    cum = sample_rec.get("cumulative_if_picked") or {}
    cum_retrieved = set(cum.get("retrieved") or [])
    cum_golds = cum_retrieved & gold_set
    recall = (len(cum_golds) / len(gold_set)) if gold_set else 0.0

    step = parent.step + 1
    terminated = bool(sample_rec.get("would_terminate"))
    reason = sample_rec.get("would_termination_reason")
    if gold_set and cum_golds == gold_set:
        terminated = True
        reason = reason or "gold_collected"
    elif step >= max_depth and not terminated:
        terminated = True
        reason = "max_depth"

    return Node(
        path_label=parent.path_label + branch,
        step=step,
        messages=list(parent.messages) + new_messages,
        cumulative_retrieved=cum_retrieved,
        cumulative_golds=cum_golds,
        recall=recall,
        terminated=terminated,
        termination_reason=reason,
    )


def _make_child_from_sample(parent: Node, sample: dict, branch: str,
                            among_ties: bool, gold_set: set, max_depth: int) -> Node:
    return Node(
        path_label=parent.path_label + branch,
        step=parent.step + 1,
        messages=list(parent.messages) + sample["new_messages"],
        cumulative_retrieved=set(sample["cum_retrieved"]),
        cumulative_golds=set(sample["cum_golds"]),
        recall=sample["recall"],
        terminated=bool(sample.get("would_terminate")),
        termination_reason=sample.get("would_termination_reason"),
    )


# ---------------------------------------------------------------------------
# BFS expansion
# ---------------------------------------------------------------------------

def _expand(
    node: Node,
    client: openai.OpenAI,
    base_request: dict,
    tool_handler,
    gold_set: set,
    n_samples: int,
    max_depth: int,
    rng: random.Random,
    sample_pool: ThreadPoolExecutor,
    stats: dict,
    samples_fp,
    exp_fp,
    resume_state: Optional[Dict[str, Any]],
) -> None:
    if node.terminated or node.step >= max_depth:
        return

    # Resume short-circuit
    if resume_state and node.path_label in resume_state["completed_expansions"]:
        exp = resume_state["completed_expansions"][node.path_label]
        samples_map = resume_state["samples_by_parent"].get(node.path_label, {})
        stats["resumed_expansions"] = stats.get("resumed_expansions", 0) + 1
        if exp.get("picks") is None:
            node.terminated = True
            node.termination_reason = "all_samples_terminal"
            return
        for label in ("B", "W", "R"):
            sidx = exp["picks"][label]["sample_idx"]
            sample_rec = samples_map.get(sidx)
            if sample_rec is None:
                continue
            child = _reconstruct_child_from_saved(
                parent=node, sample_rec=sample_rec, branch=label,
                among_ties=exp["picks"][label].get("among_ties", False),
                gold_set=gold_set, max_depth=max_depth,
            )
            node.children.append(child)
        return

    futures = [
        sample_pool.submit(
            _sample_one, client, base_request, node.messages, tool_handler, i, gold_set,
            node.cumulative_retrieved, node.step, max_depth,
        )
        for i in range(n_samples)
    ]
    samples = [f.result() for f in futures]
    stats["llm_calls"] += n_samples
    stats["tool_calls"] += sum(len(s["tool_calls"]) for s in samples)

    eligible = [
        i for i, s in enumerate(samples)
        if s.get("has_tool_call") and s.get("tool_error") is None
    ]
    step = node.step + 1

    if not eligible:
        for i, s in enumerate(samples):
            _write_sample_line(samples_fp, node.path_label, step, i, s, None, None)
        _write_expansion_line(exp_fp, node.path_label, step, [], None)
        node.terminated = True
        node.termination_reason = "all_samples_terminal"
        return

    eligible_scores = {i: samples[i]["recall"] for i in eligible}
    best_val = max(eligible_scores.values())
    worst_val = min(eligible_scores.values())
    best_idxs = [i for i, v in eligible_scores.items() if v == best_val]
    worst_idxs = [i for i, v in eligible_scores.items() if v == worst_val]
    best_i = rng.choice(best_idxs)
    worst_i = rng.choice(worst_idxs)
    random_i = rng.choice(eligible)
    picks = {
        "B": {"sample_idx": best_i, "among_ties": len(best_idxs) > 1},
        "W": {"sample_idx": worst_i, "among_ties": len(worst_idxs) > 1},
        "R": {"sample_idx": random_i, "among_ties": len(eligible) > 1},
    }

    # Pick a primary label per idx in case the same sample_idx is picked for
    # multiple labels (e.g., n_eligible==1). Priority order: B, W, R.
    branch_by_idx: Dict[int, Tuple[str, bool]] = {}
    for label in ("B", "W", "R"):
        sidx = picks[label]["sample_idx"]
        if sidx not in branch_by_idx:
            branch_by_idx[sidx] = (label, picks[label]["among_ties"])

    for i, s in enumerate(samples):
        if i in branch_by_idx:
            label, among_ties = branch_by_idx[i]
            _write_sample_line(samples_fp, node.path_label, step, i, s, label, among_ties)
        else:
            _write_sample_line(samples_fp, node.path_label, step, i, s, None, None)

    _write_expansion_line(exp_fp, node.path_label, step, eligible, picks)

    for label in ("B", "W", "R"):
        sidx = picks[label]["sample_idx"]
        sample = samples[sidx]
        child = _make_child_from_sample(
            node, sample, label, picks[label]["among_ties"], gold_set, max_depth,
        )
        node.children.append(child)


def _expand_tree(
    root: Node,
    client: openai.OpenAI,
    base_request: dict,
    tool_handler,
    gold_set: set,
    n_samples: int,
    max_depth: int,
    rng: random.Random,
    sample_workers: int,
    stats: dict,
    samples_fp,
    exp_fp,
    resume_state: Optional[Dict[str, Any]],
) -> None:
    with ThreadPoolExecutor(max_workers=sample_workers) as sample_pool:
        frontier: List[Node] = [root]
        for _depth in range(max_depth):
            next_frontier: List[Node] = []
            for node in frontier:
                _expand(
                    node, client, base_request, tool_handler, gold_set,
                    n_samples, max_depth, rng, sample_pool, stats,
                    samples_fp, exp_fp, resume_state,
                )
                for c in node.children:
                    if not c.terminated:
                        next_frontier.append(c)
            if not next_frontier:
                break
            frontier = next_frontier


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

def _build_tool_handler(args, parser):
    if args.retrievers_config:
        if args.get_document:
            parser.error("--get-document is not supported with --retrievers-config")
        configs = load_retrievers_config(args.retrievers_config)
        handler = MultiRetrieverToolHandler(
            configs=configs,
            snippet_max_tokens=args.snippet_max_tokens,
            k=args.k,
            include_get_document=False,
        )
        meta = handler.metadata()
        return handler, meta, None
    else:
        searcher_class = SearcherType.get_searcher_class(args.searcher_type)
        searcher = searcher_class(args)
        handler = SearchToolHandler(
            searcher=searcher,
            snippet_max_tokens=args.snippet_max_tokens,
            k=args.k,
            include_get_document=args.get_document,
        )
        return handler, None, args.searcher_type


# ---------------------------------------------------------------------------
# Per-qid driver
# ---------------------------------------------------------------------------

def _process_query(
    qid: str,
    qtext: str,
    *,
    client: openai.OpenAI,
    tool_handler,
    args,
    gold_set: set,
    evidence_set: set,
    retriever_metadata: Optional[List[dict]],
    searcher_type: Optional[str],
    qid_dir: Path,
    run_name: str,
    tool_definitions: List[dict],
    answer: Optional[str] = None,
) -> None:
    qid_dir.mkdir(parents=True, exist_ok=True)

    resume_state = _load_resume_state(qid_dir)

    formatted = format_query(qtext, args.query_template)
    started_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    if resume_state is None:
        _write_meta(
            qid_dir, qid, qtext, formatted, gold_set, evidence_set, answer,
            args, tool_definitions, retriever_metadata, searcher_type,
            run_name, started_at,
        )

    samples_fp = (qid_dir / "samples.jsonl").open("a")
    exp_fp = (qid_dir / "expansions.jsonl").open("a")

    try:
        if resume_state is None or resume_state.get("root_record") is None:
            _write_root_line(samples_fp, formatted)

        base_request = {
            "model": args.model,
            "max_output_tokens": args.max_tokens,
            "tools": tool_definitions,
            "truncation": "auto",
            "temperature": args.temperature,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": args.thinking}},
        }
        if args.top_p is not None:
            base_request["top_p"] = args.top_p
        if args.reasoning_effort:
            base_request["reasoning"] = {"effort": args.reasoning_effort}

        root = Node(
            path_label="",
            step=0,
            messages=[{"role": "user", "content": formatted}],
            cumulative_retrieved=set(),
            cumulative_golds=set(),
            recall=0.0,
        )

        stats = {"llm_calls": 0, "tool_calls": 0, "resumed_expansions": 0}
        rng = random.Random(args.seed + (int(hash(qid)) % (2**31)))
        started = time.time()

        _expand_tree(
            root=root,
            client=client,
            base_request=base_request,
            tool_handler=tool_handler,
            gold_set=gold_set,
            n_samples=args.n_samples,
            max_depth=args.depth,
            rng=rng,
            sample_workers=args.sample_workers,
            stats=stats,
            samples_fp=samples_fp,
            exp_fp=exp_fp,
            resume_state=resume_state,
        )

        elapsed = time.time() - started
        _write_summary(qid_dir, qid, started_at, elapsed, qid_dir / "samples.jsonl", root)

        _nodes, leaves = _walk_tree(root)
        max_recall = max((l.recall for l in leaves), default=0.0)
        resumed = stats.get("resumed_expansions", 0)
        print(
            f"[{qid}] depth={args.depth} leaves={len(leaves)} new_llm_calls={stats['llm_calls']} "
            f"resumed_exp={resumed} max_recall={max_recall:.2f} elapsed={elapsed:.1f}s -> {qid_dir}"
        )
    finally:
        samples_fp.close()
        exp_fp.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run agentic search as a branching tree (best/worst/random).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="topics-qrels/curated50.jsonl",
                        help="BrowseComp+ JSONL with query_id, query, gold_docs "
                             "(curated50.jsonl for smoke; browsecomp_plus_decrypted.jsonl "
                             "for the full 830 queries).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N queries from --dataset (smoke runs).")
    parser.add_argument("--reverse-dataset", action="store_true",
                        help="Process queries in reverse order (last entry first). Useful for "
                             "splitting a sweep between two machines that meet in the middle.")
    parser.add_argument("--gold-source", choices=["gold", "evidence", "union"], default="gold",
                        help="Score recall against gold_docs (strict), evidence_docs (loose), or the union.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--model-url", default="http://localhost:8000/v1")
    parser.add_argument("--output-dir", default="output-test/sampling_tree")
    parser.add_argument("--run-name", default=None,
                        help="Sub-directory under --output-dir. Same name resumes a prior run; "
                             "defaults to UTC timestamp YYYYMMDDTHHMMSSZ.")
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--query-template",
                        choices=["QUERY_TEMPLATE", "QUERY_TEMPLATE_NO_GET_DOCUMENT",
                                 "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION"],
                        default="QUERY_TEMPLATE_NO_GET_DOCUMENT")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default=None,
                        help="OpenAI-style reasoning effort. Ignored by vLLM for Qwen3 — use --thinking to toggle thinking mode instead.")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable Qwen3 thinking mode (default: disabled). Maps to "
                             "extra_body.chat_template_kwargs.enable_thinking. Disabling cuts "
                             "reasoning-token overhead ~10x.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature. Qwen3 non-thinking spec: 0.7. Thinking spec: 0.6.")
    parser.add_argument("--top-p", type=float, default=0.8,
                        help="Top-p sampling. Qwen3 non-thinking spec: 0.8. Thinking spec: 0.95.")

    parser.add_argument("--n-samples", type=int, default=10,
                        help="Samples per node")
    parser.add_argument("--depth", type=int, default=8, help="Max tree depth")
    parser.add_argument("--sample-workers", type=int, default=10,
                        help="Parallel API calls per node (<= n-samples)")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--snippet-max-tokens", type=int, default=512,
                        help="Truncate doc text to this many tokens BEFORE sending to the LLM. "
                             "Controls what the model sees.")
    parser.add_argument("--save-snippet-chars", type=int, default=30,
                        help="Truncate doc snippets to this many chars when writing samples.jsonl. "
                             "Set <0 to keep full snippets. Independent from --snippet-max-tokens. "
                             "NOTE: resume from disk will see truncated snippets in reconstructed "
                             "function_call_output - not byte-equivalent to a fresh run.")
    parser.add_argument("--k", type=int, default=5, help="Top-k results per search")
    parser.add_argument("--get-document", action="store_true")

    parser.add_argument("--searcher-type", choices=SearcherType.get_choices(), default=None)
    parser.add_argument("--retrievers-config", default=None,
                        help="Multi-retriever JSON config (mutually exclusive with --searcher-type)")

    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--hf-home", type=str, default=None)

    temp_args, _ = parser.parse_known_args()
    if temp_args.retrievers_config and temp_args.searcher_type:
        parser.error("--retrievers-config and --searcher-type are mutually exclusive")
    if not temp_args.retrievers_config and not temp_args.searcher_type:
        parser.error("Either --searcher-type or --retrievers-config must be supplied")
    if temp_args.searcher_type:
        SearcherType.get_searcher_class(temp_args.searcher_type).parse_args(parser)

    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    global _SAVE_SNIPPET_MAX_CHARS
    _SAVE_SNIPPET_MAX_CHARS = args.save_snippet_chars if args.save_snippet_chars >= 0 else None

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    client = openai.OpenAI(base_url=args.model_url, api_key="EMPTY")

    tool_handler, retriever_metadata, searcher_type = _build_tool_handler(args, parser)
    if retriever_metadata is not None:
        names = ", ".join(f"{m['name']}({m['type']})" for m in retriever_metadata)
        print(f"sampling-tree multi-retriever: {names}")
    else:
        print(f"sampling-tree single-searcher: {searcher_type}")

    tool_definitions = tool_handler.get_tool_definitions()

    records = load_dataset_jsonl(args.dataset)
    print(f"Loaded {len(records)} queries from {args.dataset}")
    if args.reverse_dataset:
        records = list(reversed(records))
        print(f"--reverse-dataset -> processing in reverse order (last query first)")
    if args.limit:
        records = records[: args.limit]
        print(f"--limit {args.limit} -> processing first {len(records)} queries")

    out_dir = Path(args.output_dir).expanduser().resolve()
    run_dir = out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    processed = {p.parent.name for p in run_dir.glob("*/summary.json")}
    remaining = [r for r in records if r["query_id"] not in processed]
    print(f"Processing {len(remaining)} queries (skipping {len(processed)} already done)")

    for rec in tqdm(remaining, desc="queries", unit="q"):
        qid = rec["query_id"]
        qtext = rec["query"]
        if args.gold_source == "gold":
            gold_set = rec["gold_docids"]
        elif args.gold_source == "evidence":
            gold_set = rec["evidence_docids"]
        else:
            gold_set = rec["gold_docids"] | rec["evidence_docids"]
        if not gold_set:
            print(f"[{qid}] WARNING: no {args.gold_source} docs for this query; recall will be 0")
        try:
            _process_query(
                qid=qid, qtext=qtext, client=client, tool_handler=tool_handler,
                args=args, gold_set=gold_set, evidence_set=rec["evidence_docids"],
                retriever_metadata=retriever_metadata, searcher_type=searcher_type,
                qid_dir=run_dir / qid, run_name=args.run_name,
                tool_definitions=tool_definitions, answer=rec.get("answer"),
            )
        except Exception as e:
            print(f"[{qid}] ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
