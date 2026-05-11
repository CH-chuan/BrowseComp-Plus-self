"""
Multi-retriever search tool handler.

Lets the agent pick which retriever (e.g. bm25, faiss, reasonir) to call on each
`search` invocation, instead of being bound to a single retriever at startup.

Configured via a JSON file passed with --retrievers-config; each entry names a
retriever, its searcher type, a description (surfaced into the tool description),
and the searcher-specific args (snake_case, matching argparse dest names).
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List

from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from searcher.searchers import SearcherType
from searcher.searchers.base import BaseSearcher


@dataclass
class RetrieverConfig:
    name: str
    type: str
    description: str = ""
    args: Dict[str, Any] = field(default_factory=dict)


def load_retrievers_config(path: str) -> List[RetrieverConfig]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict) or "retrievers" not in raw:
        raise ValueError(
            f"Retrievers config {path} must be an object with a 'retrievers' list"
        )

    valid_types = set(SearcherType.get_choices())
    seen_names: set[str] = set()
    configs: List[RetrieverConfig] = []

    for i, entry in enumerate(raw["retrievers"]):
        if not isinstance(entry, dict):
            raise ValueError(f"retrievers[{i}] must be an object")
        name = entry.get("name")
        rtype = entry.get("type")
        if not name or not isinstance(name, str):
            raise ValueError(f"retrievers[{i}].name is required and must be a string")
        if name in seen_names:
            raise ValueError(f"Duplicate retriever name: {name!r}")
        if rtype not in valid_types:
            raise ValueError(
                f"retrievers[{i}].type={rtype!r} is not a valid searcher type. "
                f"Valid: {sorted(valid_types)}"
            )
        seen_names.add(name)
        configs.append(
            RetrieverConfig(
                name=name,
                type=rtype,
                description=entry.get("description", ""),
                args=entry.get("args", {}) or {},
            )
        )

    if not configs:
        raise ValueError(f"Retrievers config {path} has no entries")

    return configs


class MultiRetrieverToolHandler:
    """Tool handler that routes `search` calls to one of N pre-built searchers
    based on a `retriever` argument supplied by the agent."""

    def __init__(
        self,
        configs: List[RetrieverConfig],
        snippet_max_tokens: int | None = None,
        k: int = 5,
        include_get_document: bool = False,
        tool_name: str = "search",
        query_arg_name: str = "query",
    ):
        if include_get_document:
            raise ValueError(
                "get_document is not supported in multi-retriever mode (v1). "
                "Re-run without --get-document, or use single --searcher-type mode."
            )

        self.configs = configs
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = False
        self.tool_name = tool_name
        self.query_arg_name = query_arg_name

        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

        self.searchers: Dict[str, BaseSearcher] = {}
        for cfg in configs:
            searcher_cls = SearcherType.get_searcher_class(cfg.type)
            ns = argparse.Namespace(**cfg.args)
            self.searchers[cfg.name] = searcher_cls(ns)

    def _retriever_names(self) -> List[str]:
        return [cfg.name for cfg in self.configs]

    def _tool_description(self) -> str:
        # Use the first retriever's base description as a template, then append
        # the per-retriever guide so the agent knows when to pick which.
        first = next(iter(self.searchers.values()))
        base = first.search_description(self.k)
        lines = [base, "", "Available retrievers:"]
        for cfg in self.configs:
            desc = cfg.description.strip() or "(no description provided)"
            lines.append(f"- {cfg.name} ({cfg.type}): {desc}")
        lines.append(
            "Pick the retriever best suited to the query for each call. The same "
            "search query may give different results across retrievers."
        )
        return "\n".join(lines)

    def get_tool_definitions(self):
        return [
            {
                "type": "function",
                "name": self.tool_name,
                "description": self._tool_description(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        self.query_arg_name: {
                            "type": "string",
                            "description": "Search query string",
                        },
                        "retriever": {
                            "type": "string",
                            "enum": self._retriever_names(),
                            "description": (
                                "Which retriever to use for this call. "
                                "Must be one of the retrievers listed in the description."
                            ),
                        },
                    },
                    "required": [self.query_arg_name, "retriever"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

    def execute_tool(self, tool_name: str, arguments: dict):
        if tool_name != self.tool_name:
            raise ValueError(
                f"Unknown tool: {tool_name} "
                f"(multi-retriever mode only supports {self.tool_name!r})"
            )
        return self._search(arguments[self.query_arg_name], arguments["retriever"])

    def _search(self, query: str, retriever: str):
        if retriever not in self.searchers:
            raise ValueError(
                f"Unknown retriever: {retriever!r}. "
                f"Available: {self._retriever_names()}"
            )

        searcher = self.searchers[retriever]
        candidates = searcher.search(query, self.k)

        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
            for cand in candidates:
                text = cand["text"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > self.snippet_max_tokens:
                    truncated_tokens = tokens[: self.snippet_max_tokens]
                    cand["snippet"] = self.tokenizer.decode(
                        truncated_tokens, skip_special_tokens=True
                    )
                else:
                    cand["snippet"] = text
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]

        results = []
        for cand in candidates:
            entry = {"docid": cand["docid"], "retriever": retriever}
            if cand.get("score") is not None:
                entry["score"] = cand["score"]
            entry["snippet"] = cand["snippet"]
            results.append(entry)

        return json.dumps(results, indent=2)

    def metadata(self) -> List[Dict[str, Any]]:
        """Compact, JSON-serializable summary of configured retrievers; used by
        the trajectory's request_config so analyses know which retrievers were
        available and what the agent could choose from."""
        return [
            {"name": cfg.name, "type": cfg.type, "description": cfg.description}
            for cfg in self.configs
        ]
