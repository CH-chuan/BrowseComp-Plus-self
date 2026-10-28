"""Experiment-oriented agent runner.

Two modes:
    - single: one full agent trajectory per query, with per-search recall logged
              to a JSONL sidecar. Equivalent to a stripped oss_client.py + recall.
    - tree:   tree-search over the agent's search-step space. At each depth,
              sample N candidate next-search continuations from each surviving
              parent, score by (gold_recall, evidence_recall, retriever_pref,
              query_len), keep best/worst/random per parent. Stop a subtree
              when cumulative docids ⊇ gold_set, or at --max-depth.

vLLM-backed (OpenAI-compatible Responses API). Trajectories are stored per
node so any node JSON can be loaded with --from-trajectory and tree expansion
continues from there.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import openai
from rich import print as rprint
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts import format_query
from trajectory_utils import (
    build_request_config,
    build_trajectory_entry,
    truncate_trajectory_tool_result,
)
from utils import extract_retrieved_docids_from_result

from multi_retriever_tool import MultiRetrieverToolHandler, load_retrievers_config
from oss_client import SearchToolHandler
from searcher.searchers import SearcherType

from evaluation.qrel_utils import (
    classify_retriever,
    extract_docids_from_tool_output,
    family_from_searcher_type,
    load_qrel,
    reconstruct_chat,
)


# =====================================================================
# Recall logging
# =====================================================================


def _utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


class RecallLogger:
    """Per-query JSONL logger.

    Appends one line per search call (whether or not the call's node ends up
    being selected as a survivor) plus selection_event lines and a final
    summary line. Flushes after every write.
    """

    def __init__(
        self,
        qid: str,
        path: Path,
        gold_set: set[str],
        evidence_set: set[str],
    ):
        self.qid = qid
        self.path = path
        self.gold_set = set(gold_set)
        self.evidence_set = set(evidence_set)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def _write(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, default=str) + "\n")
        self._fh.flush()

    def log_node(
        self,
        *,
        node_id: str,
        depth: int,
        parent_id: str | None,
        select_label: str | None,
        tool_name: str | None,
        tool_arguments: Any,
        new_docids: list[str],
        cumulative_docids: list[str],
        is_terminal: bool,
    ) -> tuple[float, float]:
        cumulative_set = set(cumulative_docids)
        gold_recall = (
            len(cumulative_set & self.gold_set) / len(self.gold_set)
            if self.gold_set else 0.0
        )
        evidence_recall = (
            len(cumulative_set & self.evidence_set) / len(self.evidence_set)
            if self.evidence_set else 0.0
        )
        self._write({
            "qid": self.qid,
            "event": "node",
            "node_id": node_id,
            "depth": depth,
            "parent_id": parent_id,
            "select_label": select_label,
            "tool_name": tool_name,
            "tool_arguments": tool_arguments,
            "new_docids": new_docids,
            "cumulative_docids": sorted(cumulative_set),
            "cumulative_count": len(cumulative_set),
            "gold_recall": gold_recall,
            "evidence_recall": evidence_recall,
            "gold_total": len(self.gold_set),
            "evidence_total": len(self.evidence_set),
            "is_terminal": is_terminal,
            "timestamp": _utc_now(),
        })
        return gold_recall, evidence_recall

    def log_selection(self, depth: int, survivors: list[dict]) -> None:
        self._write({
            "qid": self.qid,
            "event": "selection",
            "depth": depth,
            "survivors": survivors,
            "timestamp": _utc_now(),
        })

    def log_summary(self, summary: dict) -> None:
        body = {
            "qid": self.qid,
            "event": "summary",
            "gold_total": len(self.gold_set),
            "evidence_total": len(self.evidence_set),
            "timestamp": _utc_now(),
        }
        body.update(summary)
        self._write(body)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class RecallTrackingToolHandler:
    """Wraps a SearchToolHandler / MultiRetrieverToolHandler and forwards
    every search-tool result to a RecallLogger. Used in single mode."""

    def __init__(self, inner, recall_logger: RecallLogger | None):
        self.inner = inner
        self.recall_logger = recall_logger
        self._cumulative: set[str] = set()
        self._step = 0
        self._searches_until_full_gold: int | None = None
        self._searches_until_full_evidence: int | None = None

    def get_tool_definitions(self):
        return self.inner.get_tool_definitions()

    def execute_tool(self, tool_name: str, arguments: dict):
        result = self.inner.execute_tool(tool_name, arguments)
        if self.recall_logger and _is_search_tool(tool_name):
            self._step += 1
            new_docids = extract_docids_from_tool_output(result)
            self._cumulative.update(new_docids)
            gold_recall, evidence_recall = self.recall_logger.log_node(
                node_id=f"step-{self._step}",
                depth=self._step,
                parent_id=f"step-{self._step - 1}" if self._step > 1 else None,
                select_label=None,
                tool_name=tool_name,
                tool_arguments=arguments,
                new_docids=new_docids,
                cumulative_docids=sorted(self._cumulative),
                is_terminal=False,
            )
            if (
                self._searches_until_full_gold is None
                and self.recall_logger.gold_set
                and self.recall_logger.gold_set.issubset(self._cumulative)
            ):
                self._searches_until_full_gold = self._step
            if (
                self._searches_until_full_evidence is None
                and self.recall_logger.evidence_set
                and self.recall_logger.evidence_set.issubset(self._cumulative)
            ):
                self._searches_until_full_evidence = self._step
        return result

    def finalize_summary(self) -> dict:
        return {
            "total_searches": self._step,
            "cumulative_count": len(self._cumulative),
            "cumulative_docids": sorted(self._cumulative),
            "searches_until_full_gold": self._searches_until_full_gold,
            "searches_until_full_evidence": self._searches_until_full_evidence,
        }


def _is_search_tool(tool_name: str) -> bool:
    n = (tool_name or "").lower()
    return ("search" in n or "retrieval" in n) and "document" not in n


# =====================================================================
# Tree-mode data structures
# =====================================================================


@dataclass
class NodeState:
    qid: str
    node_id: str
    depth: int
    parent_id: str | None
    input_items: list[dict]                 # full OpenAI input array up through THIS node
    cumulative_docids: list[str]            # sorted; union along ancestor chain incl. self
    new_docids: list[str] = field(default_factory=list)
    tool_name: str | None = None
    tool_arguments: Any = None
    tool_output: str | None = None
    tool_call_id: str | None = None
    raw_response: dict | None = None
    gold_recall: float = 0.0
    evidence_recall: float = 0.0
    retriever_family: str = "unknown"       # bm25/embedding/unknown
    query_len: int = 0
    is_terminal: bool = False
    select_labels: list[str] = field(default_factory=list)


@dataclass
class QueryTree:
    qid: str
    gold_set: set[str]
    evidence_set: set[str]
    nodes: dict[str, NodeState] = field(default_factory=dict)
    survivors_by_depth: dict[int, list[str]] = field(default_factory=dict)


# =====================================================================
# Per-query I/O
# =====================================================================


def _qid_run_dir(out_dir: Path, qid: str, mode: str) -> Path:
    if mode == "tree":
        return out_dir / "runs" / qid
    return out_dir / "runs"


def _qid_recall_path(out_dir: Path, qid: str) -> Path:
    return out_dir / "recall" / f"{qid}.jsonl"


def _write_node_json(out_dir: Path, node: NodeState, request_config: dict | None) -> None:
    qid_dir = out_dir / "runs" / node.qid
    qid_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "qid": node.qid,
        "node_id": node.node_id,
        "depth": node.depth,
        "parent_id": node.parent_id,
        "select_labels": node.select_labels,
        "tool_call": (
            {
                "name": node.tool_name,
                "arguments": node.tool_arguments,
                "call_id": node.tool_call_id,
            }
            if node.tool_name else None
        ),
        "tool_output": node.tool_output,
        "new_docids": node.new_docids,
        "cumulative_docids": node.cumulative_docids,
        "gold_recall": node.gold_recall,
        "evidence_recall": node.evidence_recall,
        "retriever_family": node.retriever_family,
        "query_len": node.query_len,
        "is_terminal": node.is_terminal,
        "input_items": node.input_items,
        "raw_response": node.raw_response,
        "request_config": request_config,
        "timestamp": _utc_now(),
    }
    with (qid_dir / f"{node.node_id}.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _write_index(out_dir: Path, tree: QueryTree, args_snapshot: dict, status: str) -> None:
    qid_dir = out_dir / "runs" / tree.qid
    qid_dir.mkdir(parents=True, exist_ok=True)
    survivors_serializable = {
        str(d): [
            {"node_id": nid, "labels": tree.nodes[nid].select_labels}
            for nid in nids
        ]
        for d, nids in tree.survivors_by_depth.items()
    }
    scores = {
        nid: {
            "gold_recall": n.gold_recall,
            "evidence_recall": n.evidence_recall,
            "retriever_family": n.retriever_family,
            "query_len": n.query_len,
            "cumulative_count": len(n.cumulative_docids),
            "is_terminal": n.is_terminal,
        }
        for nid, n in tree.nodes.items()
    }
    with (qid_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump({
            "qid": tree.qid,
            "status": status,
            "max_depth_reached": max(tree.survivors_by_depth.keys()) if tree.survivors_by_depth else 0,
            "gold_total": len(tree.gold_set),
            "evidence_total": len(tree.evidence_set),
            "args": args_snapshot,
            "survivors_by_depth": survivors_serializable,
            "scores": scores,
            "timestamp": _utc_now(),
        }, f, indent=2, default=str)


# =====================================================================
# Single-mode trajectory persistence (deterministic per-qid filename)
# =====================================================================


def _persist_single_response(
    out_dir: Path,
    initial_request: dict,
    messages: list,
    tool_usage: dict,
    status: str,
    *,
    query_id: str,
    trajectory: list,
    request_config: dict,
) -> Path:
    """Like oss_client._persist_response but uses runs/<qid>.json."""
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    call_output_by_id: dict[str, Any] = {}
    for item in messages or []:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            call_output_by_id[item.get("call_id")] = item.get("output")

    normalized_results: list[dict] = []
    for item in messages or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "function_call":
            normalized_results.append({
                "type": "tool_call",
                "tool_name": item.get("name"),
                "arguments": item.get("arguments"),
                "output": call_output_by_id.get(item.get("call_id")),
            })
        elif itype == "reasoning":
            summary = item.get("summary")
            if isinstance(summary, list) and summary:
                reasoning_output = summary
            else:
                reasoning_output = []
                for part in item.get("content", []) or []:
                    if isinstance(part, dict) and part.get("type") in {
                        "reasoning_text", "output_text", "text",
                    }:
                        text_val = str(part.get("text", "")).strip()
                        if text_val:
                            reasoning_output.append(text_val)
            normalized_results.append({
                "type": "reasoning",
                "tool_name": None,
                "arguments": None,
                "output": reasoning_output,
            })
        elif itype == "message":
            text_chunks: list[str] = []
            for part in item.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text_chunks.append(str(part.get("text", "")))
            text = "\n".join(c for c in text_chunks if c).strip()
            if text:
                normalized_results.append({
                    "type": "output_text",
                    "tool_name": None,
                    "arguments": None,
                    "output": text,
                })

    normalized_tool_counts: dict[str, int] = {}
    for tool_name, count in (tool_usage or {}).items():
        norm = (
            "search"
            if isinstance(tool_name, str) and "retrieval" in tool_name.lower()
            else tool_name
        )
        normalized_tool_counts[norm] = normalized_tool_counts.get(norm, 0) + int(count or 0)

    path = runs_dir / f"{query_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "model": initial_request.get("model"),
                "reasoning": initial_request.get("reasoning"),
                "output_dir": str(out_dir),
            },
            "query_id": query_id,
            "tool_call_counts": normalized_tool_counts,
            "status": status,
            "retrieved_docids": extract_retrieved_docids_from_result(normalized_results),
            "result": normalized_results,
            "request_config": request_config,
            "trajectory": trajectory,
        }, f, indent=2, default=str)
    return path


# =====================================================================
# Single-mode agent loop (mirrors oss_client.run_conversation_with_tools)
# =====================================================================


def _run_full_conversation(
    client: openai.OpenAI,
    initial_request: dict,
    tool_handler,
    max_iterations: int,
    verbose: bool,
):
    tool_usage: dict[str, int] = {}
    trajectory: list[dict] = []
    messages = initial_request["input"]
    iteration = 1

    while iteration <= max_iterations:
        try:
            request = initial_request.copy()
            request["input"] = messages
            response = client.responses.create(**request)
        except Exception as e:
            if verbose:
                print(f"[run_full_conversation] error: {e}")
            iteration += 1
            continue

        response_dict = response.model_dump(mode="python")
        turn_usage = None
        if response_dict.get("usage"):
            ru = response_dict["usage"]
            turn_usage = {
                "input_tokens": ru.get("input_tokens"),
                "output_tokens": ru.get("output_tokens"),
                "total_tokens": ru.get("total_tokens"),
            }

        messages.extend(response_dict["output"])

        # Reasoning-only retry (matches oss_client behavior)
        if (
            len(response_dict["output"]) >= 1
            and response_dict["output"][-1]["type"] == "reasoning"
        ):
            messages.pop()
            continue

        function_calls = [
            it for it in response_dict["output"] if it["type"] == "function_call"
        ]

        if not function_calls:
            trajectory.append(build_trajectory_entry(
                turn=iteration - 1,
                raw_response=response_dict,
                usage=turn_usage,
                stop_reason=response_dict.get("status"),
                tool_results=[],
            ))
            return messages, tool_usage, "completed", trajectory

        new_messages = messages.copy()
        turn_tool_results = []
        for tool_call in function_calls:
            arguments: dict | None = None
            try:
                arguments = json.loads(tool_call["arguments"])
                result = tool_handler.execute_tool(tool_call["name"], arguments)
                tool_usage[tool_call["name"]] = tool_usage.get(tool_call["name"], 0) + 1
                new_messages.append({
                    "type": "function_call_output",
                    "call_id": tool_call["call_id"],
                    "output": result,
                })
                turn_tool_results.append(truncate_trajectory_tool_result(
                    call_id=tool_call.get("call_id"),
                    tool_name=tool_call["name"],
                    arguments=arguments,
                    raw_output=result,
                ))
            except Exception as e:
                err = f"Error executing {tool_call['name']}: {e}"
                new_messages.append({
                    "type": "function_call_output",
                    "call_id": tool_call.get("call_id"),
                    "output": err,
                })
                turn_tool_results.append(truncate_trajectory_tool_result(
                    call_id=tool_call.get("call_id"),
                    tool_name=tool_call["name"],
                    arguments=arguments,
                    raw_output=err,
                ))

        trajectory.append(build_trajectory_entry(
            turn=iteration - 1,
            raw_response=response_dict,
            usage=turn_usage,
            stop_reason="tool_calls",
            tool_results=turn_tool_results,
        ))
        messages = new_messages
        iteration += 1

    return messages, tool_usage, "incomplete", trajectory


# =====================================================================
# Tree-mode single-step sampler
# =====================================================================


def _sample_one_search_step(
    client: openai.OpenAI,
    base_request: dict,
    input_items: list[dict],
    tool_handler,
    *,
    temperature: float,
    seed: int,
    max_iterations: int,
    verbose: bool,
):
    """Run the agent from `input_items` until it issues exactly one tool call
    (or stops without one). Execute that tool call and return the new state.

    Returns dict with keys:
        new_input_items: list[dict]   # input_items + assistant + tool_output
        raw_response:    dict | None  # the LLM response that produced the tool call
        tool_call:       dict | None  # {"name", "arguments", "call_id"} or None
        tool_output:     str | None
    """
    messages = list(input_items)
    iteration = 1
    last_response_dict: dict | None = None
    last_output_items: list[dict] = []

    while iteration <= max_iterations:
        try:
            request = base_request.copy()
            request["input"] = messages
            request["temperature"] = temperature
            # vLLM accepts `seed` for the underlying sampler via extra_body.
            extra_body = dict(request.get("extra_body") or {})
            extra_body["seed"] = seed
            request["extra_body"] = extra_body
            response = client.responses.create(**request)
        except Exception as e:
            if verbose:
                print(f"[sample_one_search_step] error: {e}")
            iteration += 1
            continue

        response_dict = response.model_dump(mode="python")
        output_items = response_dict.get("output") or []
        last_response_dict = response_dict
        last_output_items = output_items
        function_calls = [it for it in output_items if it.get("type") == "function_call"]

        if not function_calls:
            # Pure-reasoning response: retry once or twice in case the next call
            # produces an action. Don't pollute the chat history with reasoning
            # that didn't lead to a tool call.
            if output_items and all(it.get("type") == "reasoning" for it in output_items):
                iteration += 1
                continue
            # Agent stopped (answered or otherwise) without issuing a search.
            return {
                "new_input_items": messages + output_items,
                "raw_response": response_dict,
                "tool_call": None,
                "tool_output": None,
            }

        # Use the first function call only (one-search-per-sample).
        tc = function_calls[0]
        try:
            arguments = json.loads(tc["arguments"])
        except Exception:
            arguments = {}

        try:
            tool_output = tool_handler.execute_tool(tc["name"], arguments)
        except Exception as e:
            tool_output = f"Error executing {tc['name']}: {e}"

        # Append output_items (which may contain reasoning + this tool call) and the tool output.
        new_items = messages + output_items + [{
            "type": "function_call_output",
            "call_id": tc.get("call_id"),
            "output": tool_output,
        }]

        return {
            "new_input_items": new_items,
            "raw_response": response_dict,
            "tool_call": {
                "name": tc.get("name"),
                "arguments": arguments,
                "call_id": tc.get("call_id"),
            },
            "tool_output": tool_output,
        }

    # Exhausted retries — fall back to whatever we last saw (typically pure
    # reasoning). Treated as a terminal candidate by the caller.
    return {
        "new_input_items": messages + last_output_items,
        "raw_response": last_response_dict,
        "tool_call": None,
        "tool_output": None,
    }


# =====================================================================
# Tree expansion
# =====================================================================


def _query_text_for_score(arguments: Any) -> str:
    if not isinstance(arguments, dict):
        return ""
    for key in ("user_query", "query", "search_query"):
        v = arguments.get(key)
        if isinstance(v, str):
            return v
    return ""


def _score(node: NodeState, retriever_pref: list[str]) -> tuple:
    """Lexicographic score: gold_recall ↓, evidence_recall ↓, retriever-pref ↓, query_len ↑."""
    try:
        rank = retriever_pref.index(node.retriever_family)
    except ValueError:
        rank = len(retriever_pref)  # unknown families sort last
    return (node.gold_recall, node.evidence_recall, -rank, -node.query_len)


def _build_node(
    *,
    parent: NodeState,
    sample_idx: int,
    sample_result: dict,
    gold_set: set[str],
    evidence_set: set[str],
    args,
) -> NodeState:
    new_input_items = sample_result["new_input_items"]
    tc = sample_result.get("tool_call") or {}
    tool_name = tc.get("name")
    tool_arguments = tc.get("arguments")
    tool_output = sample_result.get("tool_output")
    tool_call_id = tc.get("call_id")

    new_docids = extract_docids_from_tool_output(tool_output) if tool_output else []
    cumulative_set = set(parent.cumulative_docids) | set(new_docids)
    cumulative = sorted(cumulative_set)

    gold_recall = (
        len(cumulative_set & gold_set) / len(gold_set) if gold_set else 0.0
    )
    evidence_recall = (
        len(cumulative_set & evidence_set) / len(evidence_set) if evidence_set else 0.0
    )

    default_family = getattr(args, "_default_retriever_family", "unknown")
    family = (
        classify_retriever(tool_name or "", tool_output, default_family=default_family)
        if tool_name else "unknown"
    )
    query_len = len(_query_text_for_score(tool_arguments))

    # Mark terminal if (a) agent didn't search OR (b) cumulative covers gold.
    is_terminal = (
        tool_name is None
        or (args.stop_on_full_gold and bool(gold_set) and gold_set.issubset(cumulative_set))
    )

    node_id = f"d{parent.depth + 1}-{parent.node_id}-s{sample_idx}"
    return NodeState(
        qid=parent.qid,
        node_id=node_id,
        depth=parent.depth + 1,
        parent_id=parent.node_id,
        input_items=new_input_items,
        cumulative_docids=cumulative,
        new_docids=new_docids,
        tool_name=tool_name,
        tool_arguments=tool_arguments,
        tool_output=tool_output,
        tool_call_id=tool_call_id,
        raw_response=sample_result.get("raw_response"),
        gold_recall=gold_recall,
        evidence_recall=evidence_recall,
        retriever_family=family,
        query_len=query_len,
        is_terminal=is_terminal,
    )


def _pick_survivors(
    candidates: list[NodeState],
    select_modes: list[str],
    rng: random.Random,
    retriever_pref: list[str],
) -> list[tuple[NodeState, str]]:
    """Returns list of (node, label) pairs. A node may appear under multiple labels."""
    if not candidates:
        return []
    chosen: list[tuple[NodeState, str]] = []
    for label in select_modes:
        if label == "best":
            chosen.append((max(candidates, key=lambda n: _score(n, retriever_pref)), label))
        elif label == "worst":
            chosen.append((min(candidates, key=lambda n: _score(n, retriever_pref)), label))
        elif label == "random":
            chosen.append((rng.choice(candidates), label))
        else:
            raise ValueError(f"Unknown select mode: {label}")
    return chosen


def expand_query_tree(
    *,
    qid: str,
    initial_input_items: list[dict],
    initial_cumulative: list[str],
    starting_depth: int,
    args,
    base_request: dict,
    tool_handler,
    client: openai.OpenAI,
    request_config: dict,
    out_dir: Path,
    gold_set: set[str],
    evidence_set: set[str],
    recall_logger: RecallLogger | None,
    args_snapshot: dict,
    verbose: bool,
) -> QueryTree:
    tree = QueryTree(qid=qid, gold_set=gold_set, evidence_set=evidence_set)

    root_node_id = f"d{starting_depth}-root"
    root = NodeState(
        qid=qid,
        node_id=root_node_id,
        depth=starting_depth,
        parent_id=None,
        input_items=list(initial_input_items),
        cumulative_docids=sorted(set(initial_cumulative)),
    )
    # Root recall snapshot
    root_cum_set = set(root.cumulative_docids)
    root.gold_recall = (
        len(root_cum_set & gold_set) / len(gold_set) if gold_set else 0.0
    )
    root.evidence_recall = (
        len(root_cum_set & evidence_set) / len(evidence_set) if evidence_set else 0.0
    )
    if args.stop_on_full_gold and gold_set and gold_set.issubset(root_cum_set):
        root.is_terminal = True

    tree.nodes[root_node_id] = root
    tree.survivors_by_depth[starting_depth] = [root_node_id]
    _write_node_json(out_dir, root, request_config)
    if recall_logger:
        recall_logger.log_node(
            node_id=root_node_id,
            depth=starting_depth,
            parent_id=None,
            select_label="root",
            tool_name=None,
            tool_arguments=None,
            new_docids=[],
            cumulative_docids=root.cumulative_docids,
            is_terminal=root.is_terminal,
        )

    rng = random.Random(f"{qid}-tree-seed")
    select_modes = args.select_modes_list

    summary_meta = {
        "depth_first_full_gold": None,
        "depth_first_full_evidence": None,
        "best_path_gold_recall": root.gold_recall,
        "best_path_evidence_recall": root.evidence_recall,
    }

    for depth in range(starting_depth + 1, args.max_depth + 1):
        parents = [
            tree.nodes[nid]
            for nid in tree.survivors_by_depth.get(depth - 1, [])
            if not tree.nodes[nid].is_terminal
        ]
        if not parents:
            break

        new_survivors: list[str] = []
        for parent in parents:
            n_samples = (
                args.initial_samples if depth == starting_depth + 1
                else args.branching_factor
            )
            candidates: list[NodeState] = []
            for sample_idx in range(n_samples):
                seed = abs(hash((qid, parent.node_id, sample_idx))) % (2**31 - 1)
                sample_result = _sample_one_search_step(
                    client=client,
                    base_request=base_request,
                    input_items=parent.input_items,
                    tool_handler=tool_handler,
                    temperature=args.temperature,
                    seed=seed,
                    max_iterations=args.max_iterations_per_sample,
                    verbose=verbose,
                )
                node = _build_node(
                    parent=parent,
                    sample_idx=sample_idx,
                    sample_result=sample_result,
                    gold_set=gold_set,
                    evidence_set=evidence_set,
                    args=args,
                )
                tree.nodes[node.node_id] = node
                _write_node_json(out_dir, node, request_config)
                if recall_logger:
                    recall_logger.log_node(
                        node_id=node.node_id,
                        depth=node.depth,
                        parent_id=node.parent_id,
                        select_label=None,
                        tool_name=node.tool_name,
                        tool_arguments=node.tool_arguments,
                        new_docids=node.new_docids,
                        cumulative_docids=node.cumulative_docids,
                        is_terminal=node.is_terminal,
                    )
                candidates.append(node)

            picks = _pick_survivors(candidates, select_modes, rng, args.retriever_pref_list)
            seen_in_parent: set[str] = set()
            survivor_summary: list[dict] = []
            for node, label in picks:
                if label not in node.select_labels:
                    node.select_labels.append(label)
                survivor_summary.append({"node_id": node.node_id, "label": label})
                if node.node_id in seen_in_parent:
                    continue
                seen_in_parent.add(node.node_id)
                new_survivors.append(node.node_id)

                # Re-write node JSON now that select_labels is populated.
                _write_node_json(out_dir, node, request_config)

                if (
                    summary_meta["depth_first_full_gold"] is None
                    and gold_set and gold_set.issubset(set(node.cumulative_docids))
                ):
                    summary_meta["depth_first_full_gold"] = node.depth
                if (
                    summary_meta["depth_first_full_evidence"] is None
                    and evidence_set and evidence_set.issubset(set(node.cumulative_docids))
                ):
                    summary_meta["depth_first_full_evidence"] = node.depth
                if node.gold_recall > summary_meta["best_path_gold_recall"]:
                    summary_meta["best_path_gold_recall"] = node.gold_recall
                if node.evidence_recall > summary_meta["best_path_evidence_recall"]:
                    summary_meta["best_path_evidence_recall"] = node.evidence_recall

            if recall_logger:
                recall_logger.log_selection(depth, survivor_summary)

        tree.survivors_by_depth[depth] = new_survivors
        _write_index(out_dir, tree, args_snapshot, status="in_progress")
        if not new_survivors:
            break

    _write_index(out_dir, tree, args_snapshot, status="complete")
    if recall_logger:
        recall_logger.log_summary({
            "total_nodes": len(tree.nodes),
            "max_depth_reached": max(tree.survivors_by_depth.keys()),
            **summary_meta,
        })
    return tree


# =====================================================================
# Top-level per-query orchestration
# =====================================================================


def _build_base_request(args, tool_definitions: list[dict]) -> dict:
    req = {
        "model": args.model,
        "max_output_tokens": args.max_tokens,
        "tools": tool_definitions,
        "truncation": "auto",
        "reasoning": {"effort": args.reasoning_effort, "summary": "detailed"},
    }
    return req


def _build_initial_input(
    args, system_prompt: str | None, formatted_query: str
) -> list[dict]:
    items: list[dict] = []
    if system_prompt:
        items.append({"role": "system", "content": system_prompt})
    items.append({"role": "user", "content": formatted_query})
    return items


def _process_one_query(
    qid: str,
    qtext: str,
    *,
    args,
    client: openai.OpenAI,
    tool_handler,
    qrel_golds: dict[str, set[str]],
    qrel_evidence: dict[str, set[str]],
    system_prompt: str | None,
    request_config_base: dict,
    out_dir: Path,
    args_snapshot: dict,
):
    qid = str(qid)
    gold_set = qrel_golds.get(qid, set())
    evidence_set = qrel_evidence.get(qid, set())

    if args.no_recall_logging:
        recall_logger = None
    else:
        recall_logger = RecallLogger(qid, _qid_recall_path(out_dir, qid), gold_set, evidence_set)

    try:
        formatted_query = format_query(qtext, args.query_template)
        request_config = {
            **request_config_base,
            "formatted_query": formatted_query,
            "system_prompt": system_prompt,
            "qid": qid,
            "mode": args.mode,
        }

        if args.mode == "single":
            base_input = _build_initial_input(args, system_prompt, formatted_query)
            initial_request = _build_base_request(args, tool_handler.get_tool_definitions())
            initial_request["input"] = base_input
            if args.temperature is not None:
                initial_request["temperature"] = args.temperature

            wrapped = RecallTrackingToolHandler(tool_handler, recall_logger)
            messages, tool_usage, status, trajectory = _run_full_conversation(
                client, initial_request, wrapped, args.max_iterations, args.verbose
            )
            _persist_single_response(
                out_dir, initial_request, messages, tool_usage, status,
                query_id=qid,
                trajectory=trajectory,
                request_config=request_config,
            )
            if recall_logger:
                recall_logger.log_summary(wrapped.finalize_summary())

        else:  # tree
            base_input, initial_cumulative, starting_depth = _resolve_tree_starting_state(
                args, system_prompt, formatted_query
            )
            base_request = _build_base_request(args, tool_handler.get_tool_definitions())
            expand_query_tree(
                qid=qid,
                initial_input_items=base_input,
                initial_cumulative=initial_cumulative,
                starting_depth=starting_depth,
                args=args,
                base_request=base_request,
                tool_handler=tool_handler,
                client=client,
                request_config=request_config,
                out_dir=out_dir,
                gold_set=gold_set,
                evidence_set=evidence_set,
                recall_logger=recall_logger,
                args_snapshot=args_snapshot,
                verbose=args.verbose,
            )
    finally:
        if recall_logger:
            recall_logger.close()


def _resolve_tree_starting_state(args, system_prompt: str | None, formatted_query: str):
    """Returns (input_items, cumulative_docids, starting_depth) for tree expansion."""
    if args.from_trajectory:
        path = Path(args.from_trajectory).expanduser().resolve()
        with path.open() as f:
            node_json = json.load(f)
        input_items = reconstruct_chat(node_json)
        cumulative = node_json.get("cumulative_docids", []) or []
        starting_depth = int(node_json.get("depth", 0))
        return list(input_items), list(cumulative), starting_depth

    return _build_initial_input(args, system_prompt, formatted_query), [], 0


# =====================================================================
# Resume / query loading
# =====================================================================


def _is_done(out_dir: Path, qid: str, mode: str) -> bool:
    if mode == "tree":
        idx = out_dir / "runs" / qid / "index.json"
        if not idx.exists():
            return False
        try:
            with idx.open() as f:
                return json.load(f).get("status") == "complete"
        except Exception:
            return False
    else:
        return (out_dir / "runs" / f"{qid}.json").exists()


def _load_queries(args) -> list[tuple[str, str]]:
    path = Path(args.query_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Query file not found: {path}")
    queries: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            queries.append((row[0].strip(), row[1].strip()))

    id_filter: set[str] | None = None
    if args.query_ids:
        id_filter = {x.strip() for x in args.query_ids.split(",") if x.strip()}
    elif args.query_ids_file:
        with Path(args.query_ids_file).expanduser().resolve().open() as f:
            id_filter = {line.strip() for line in f if line.strip()}

    if id_filter is not None:
        queries = [(qid, qtxt) for qid, qtxt in queries if qid in id_filter]

    return queries


def _load_system_prompt(path: str | None) -> str | None:
    if not path:
        return None
    return Path(path).expanduser().resolve().read_text(encoding="utf-8").strip()


# =====================================================================
# CLI
# =====================================================================


def _add_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Input")
    g.add_argument("--query-file", required=True, help="TSV path with `qid\\tquery` rows")
    g.add_argument("--query-ids", default=None, help="Comma-separated qids subset")
    g.add_argument("--query-ids-file", default=None, help="File with one qid per line")

    g = parser.add_argument_group("Model / backend (vLLM-served, OpenAI-compatible)")
    g.add_argument("--model", default="Qwen/Qwen3.5-9B")
    g.add_argument("--model-url", default="http://localhost:8000/v1")
    g.add_argument("--max-tokens", type=int, default=10000)
    g.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"])
    g.add_argument("--temperature", type=float, default=None,
                   help="Sampling temperature. Default: 0.0 in single mode, 0.7 in tree mode.")

    g = parser.add_argument_group("Searcher / tools")
    g.add_argument("--searcher-type", choices=SearcherType.get_choices(), default=None)
    g.add_argument("--retrievers-config", default=None,
                   help="JSON config for multi-retriever mode (mutually exclusive with --searcher-type)")
    g.add_argument("--get-document", action="store_true",
                   help="Register the get_document tool (only valid with --searcher-type)")
    g.add_argument("--snippet-max-tokens", type=int, default=512)
    g.add_argument("--k", type=int, default=5)
    g.add_argument("--hf-token", default=None)
    g.add_argument("--hf-home", default=None)

    g = parser.add_argument_group("Prompt")
    g.add_argument("--query-template",
                   choices=["QUERY_TEMPLATE", "QUERY_TEMPLATE_NO_GET_DOCUMENT",
                            "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION"],
                   default="QUERY_TEMPLATE_NO_GET_DOCUMENT")
    g.add_argument("--system-prompt-file", default=None,
                   help="Path to a file whose contents become the system message")

    g = parser.add_argument_group("Output")
    g.add_argument("--output-dir", required=True)

    g = parser.add_argument_group("Recall / qrels")
    g.add_argument("--qrel-golds", default="topics-qrels/qrel_golds.txt")
    g.add_argument("--qrel-evidence", default="topics-qrels/qrel_evidence.txt")
    g.add_argument("--no-recall-logging", action="store_true",
                   help="Skip writing recall sidecars (e.g. when qrels are unavailable)")

    g = parser.add_argument_group("Mode")
    g.add_argument("--mode", choices=["single", "tree"], default="tree")
    g.add_argument("--initial-samples", type=int, default=10,
                   help="Tree mode: candidates at the first expansion depth")
    g.add_argument("--branching-factor", type=int, default=10,
                   help="Tree mode: candidates per surviving parent at depth >= 2")
    g.add_argument("--select-modes", default="best,worst,random",
                   help="Comma-separated subset of {best,worst,random}")
    g.add_argument("--max-depth", type=int, default=8)
    g.add_argument("--stop-on-full-gold", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--retriever-pref", default="bm25,embedding",
                   help="Tiebreak preference order for retriever families")

    g = parser.add_argument_group("Replay / resume")
    g.add_argument("--from-trajectory", default=None,
                   help="Path to a node JSON to resume tree expansion from")
    g.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    g = parser.add_argument_group("Runtime")
    g.add_argument("--num-threads", type=int, default=1)
    g.add_argument("--max-iterations", type=int, default=100,
                   help="Single mode: cap on full-conversation iterations")
    g.add_argument("--max-iterations-per-sample", type=int, default=3,
                   help="Tree mode: cap on agent turns per single-step sample")
    g.add_argument("--verbose", action="store_true")


def _validate_and_finalize_args(parser: argparse.ArgumentParser, args) -> None:
    if args.searcher_type and args.retrievers_config:
        parser.error("--searcher-type and --retrievers-config are mutually exclusive")
    if not args.searcher_type and not args.retrievers_config:
        parser.error("One of --searcher-type or --retrievers-config is required")
    if args.retrievers_config and args.get_document:
        parser.error("--get-document is not supported in multi-retriever mode")

    args.select_modes_list = [s.strip() for s in args.select_modes.split(",") if s.strip()]
    for s in args.select_modes_list:
        if s not in {"best", "worst", "random"}:
            parser.error(f"Invalid --select-modes entry: {s!r}")

    args.retriever_pref_list = [s.strip() for s in args.retriever_pref.split(",") if s.strip()]

    if args.temperature is None:
        args.temperature = 0.0 if args.mode == "single" else 0.7


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_args(parser)

    temp_args, _ = parser.parse_known_args()

    if temp_args.searcher_type and not temp_args.retrievers_config:
        searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
        searcher_class.parse_args(parser)
    else:
        searcher_class = None

    args = parser.parse_args()
    _validate_and_finalize_args(parser, args)

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(exist_ok=True)
    (out_dir / "recall").mkdir(exist_ok=True)

    client = openai.OpenAI(base_url=args.model_url, api_key="EMPTY")

    retriever_metadata = None
    if args.retrievers_config:
        retriever_configs = load_retrievers_config(args.retrievers_config)
        tool_handler = MultiRetrieverToolHandler(
            configs=retriever_configs,
            snippet_max_tokens=args.snippet_max_tokens,
            k=args.k,
            include_get_document=False,
        )
        retriever_metadata = tool_handler.metadata()
        names = ", ".join(f"{m['name']}({m['type']})" for m in retriever_metadata)
        print(f"Multi-retriever mode: {len(retriever_metadata)} retrievers — {names}")
    else:
        searcher = searcher_class(args)
        tool_handler = SearchToolHandler(
            searcher=searcher,
            snippet_max_tokens=args.snippet_max_tokens,
            k=args.k,
            include_get_document=args.get_document,
        )
        print(f"Single-searcher mode: {args.searcher_type}")

    args._default_retriever_family = (
        "unknown" if args.retrievers_config
        else family_from_searcher_type(args.searcher_type)
    )

    qrel_golds = load_qrel(Path(args.qrel_golds).expanduser().resolve())
    qrel_evidence = load_qrel(Path(args.qrel_evidence).expanduser().resolve())
    print(
        f"Loaded qrels: {len(qrel_golds)} qids with golds, "
        f"{len(qrel_evidence)} qids with evidence"
    )

    queries = _load_queries(args)
    if args.resume:
        before = len(queries)
        queries = [(qid, qt) for qid, qt in queries if not _is_done(out_dir, qid, args.mode)]
        skipped = before - len(queries)
        if skipped:
            print(f"Resume: skipping {skipped} already-done qids")

    if not queries:
        print("Nothing to do.")
        return

    system_prompt = _load_system_prompt(args.system_prompt_file)

    request_config_base = build_request_config(
        system_prompt=system_prompt,
        query_template=args.query_template,
        searcher_type=args.searcher_type,
        snippet_max_tokens=args.snippet_max_tokens,
        k=args.k,
        extra={"retrievers": retriever_metadata} if retriever_metadata else None,
    )
    args_snapshot = {
        k: v for k, v in vars(args).items()
        if k not in {"hf_token"} and not k.startswith("_")
    }
    # Cast non-JSON-serializable fields to strings.
    for k, v in list(args_snapshot.items()):
        if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
            args_snapshot[k] = str(v)

    if args.mode == "tree":
        total_calls_estimate = _estimate_tree_calls(args)
        print(
            f"Tree mode: ~{total_calls_estimate:,} LLM calls per qid "
            f"(initial={args.initial_samples}, branching={args.branching_factor}, "
            f"select_modes={args.select_modes_list}, max_depth={args.max_depth})"
        )

    print(f"Processing {len(queries)} queries → {out_dir}")

    def _run_one(qid: str, qtext: str):
        try:
            _process_one_query(
                qid, qtext,
                args=args,
                client=client,
                tool_handler=tool_handler,
                qrel_golds=qrel_golds,
                qrel_evidence=qrel_evidence,
                system_prompt=system_prompt,
                request_config_base=request_config_base,
                out_dir=out_dir,
                args_snapshot=args_snapshot,
            )
        except Exception as exc:
            print(f"[Error] qid={qid} failed: {exc}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    if args.num_threads <= 1:
        with tqdm(queries, desc="Queries", unit="q") as pbar:
            for qid, qtext in pbar:
                _run_one(qid, qtext)
    else:
        with (
            ThreadPoolExecutor(max_workers=args.num_threads) as ex,
            tqdm(total=len(queries), desc="Queries", unit="q") as pbar,
        ):
            futures = [ex.submit(_run_one, qid, qtext) for qid, qtext in queries]
            for _ in as_completed(futures):
                pbar.update(1)


def _estimate_tree_calls(args) -> int:
    n_select = max(1, len(set(args.select_modes_list)))
    total = args.initial_samples
    survivors = n_select  # depth-1 survivors
    for _ in range(args.max_depth - 1):
        total += survivors * args.branching_factor
        survivors *= n_select
    return total


if __name__ == "__main__":
    main()
