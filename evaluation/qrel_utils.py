"""Shared qrel + trajectory helpers for experiment runners.

Lifted from scripts_evaluation/analyze_retrieval.py so the in-loop recall
logger and the post-hoc analyzer share a single source of truth.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_qrel(qrel_path: Path) -> dict[str, set[str]]:
    """Load TREC qrel file into {query_id: set(docids)}."""
    qrel: dict[str, set[str]] = defaultdict(set)
    if not qrel_path.exists():
        return dict(qrel)
    with qrel_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qrel[parts[0]].add(parts[2])
    return dict(qrel)


def extract_docids_from_tool_output(output: Any) -> list[str]:
    """Extract docids from a single search tool call's output, in returned order."""
    if output is None:
        return []

    parsed = None
    raw_str = ""

    if isinstance(output, str):
        raw_str = output
        try:
            parsed = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            parsed = None
    elif isinstance(output, list):
        parsed = output

    docids: list[str] = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and "docid" in item:
                docids.append(str(item["docid"]))
    else:
        for m in re.findall(r'"docid"\s*:\s*"([^"]+)"', raw_str):
            docids.append(str(m))
        for m in re.findall(r'"docid"\s*:\s*(\d+)', raw_str):
            docids.append(str(m))

    return docids


def classify_retriever(
    tool_name: str,
    tool_output: Any = None,
    default_family: str = "unknown",
) -> str:
    """Best-effort classification of which retriever family a search call used.

    Returns one of "bm25", "embedding", "unknown". Used for the lexicographic
    tiebreak in tree-mode survivor selection.

    Resolution order:
      1. Substring of tool_name (when callers register tool names like
         "bm25_search" / "embedding_search").
      2. Multi-retriever output: per-result "retriever" field (majority wins).
      3. Caller-provided default_family (e.g. when the agent uses a generic
         "local_knowledge_base_retrieval" name but the handler knows it wraps
         a single bm25 / faiss searcher).
    """
    name = (tool_name or "").lower()
    if "bm25" in name:
        return "bm25"
    if "faiss" in name or "embedding" in name or "dense" in name or "reasonir" in name:
        return "embedding"

    parsed = None
    if isinstance(tool_output, str):
        try:
            parsed = json.loads(tool_output)
        except (json.JSONDecodeError, TypeError):
            parsed = None
    elif isinstance(tool_output, list):
        parsed = tool_output

    if isinstance(parsed, list):
        votes: dict[str, int] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            r = str(item.get("retriever", "")).lower()
            if not r:
                continue
            family = (
                "bm25" if "bm25" in r
                else "embedding" if any(k in r for k in ("faiss", "dense", "embed", "reasonir"))
                else "unknown"
            )
            votes[family] = votes.get(family, 0) + 1
        if votes:
            return max(votes, key=votes.get)

    return default_family


def family_from_searcher_type(searcher_type: str | None) -> str:
    """Map a CLI --searcher-type value (bm25/faiss/reasonir/custom) to a family."""
    if not searcher_type:
        return "unknown"
    t = searcher_type.lower()
    if t == "bm25":
        return "bm25"
    if t in {"faiss", "reasonir"}:
        return "embedding"
    return "unknown"



def reconstruct_chat(node_json: dict) -> list[dict]:
    """Return the OpenAI-style input_items list saved in a per-node JSON.

    The list already contains every system/user/assistant/tool message needed
    to resume the agent from that node — no transformation needed.
    """
    items = node_json.get("input_items")
    if not isinstance(items, list):
        raise ValueError("Node JSON has no usable 'input_items' field")
    return items
