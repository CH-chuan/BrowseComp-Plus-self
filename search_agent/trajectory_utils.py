"""
Shared utilities for saving raw trajectory data in run logs.

Provides truncation of tool outputs (search results, documents) to keep
trajectory logs compact while preserving docids, scores, and metadata.
"""

import json
from typing import Any


def truncate_tool_output(
    tool_name: str,
    raw_output: str | dict | list | None,
    snippet_preview_chars: int = 20,
    doc_preview_chars: int = 50,
) -> dict:
    """Truncate tool output for trajectory logging.

    For search results: preserves docids and scores, truncates snippet text.
    For get_document: preserves docid, truncates document text.

    Returns a dict with truncated preview and original length info.
    """
    if raw_output is None:
        return {"output_preview": None, "raw_length": 0}

    parsed = None
    raw_str = ""

    if isinstance(raw_output, str):
        raw_str = raw_output
        try:
            parsed = json.loads(raw_output)
        except (json.JSONDecodeError, TypeError):
            # Not JSON — return raw string truncated
            return {
                "output_preview": raw_output[:doc_preview_chars],
                "raw_length": len(raw_output),
            }
    elif isinstance(raw_output, (list, dict)):
        parsed = raw_output
        raw_str = json.dumps(raw_output)

    tool_lower = (tool_name or "").lower()

    # Search results: list of {docid, score, snippet/text}
    if ("search" in tool_lower or "retrieval" in tool_lower) and isinstance(
        parsed, list
    ):
        preview = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            entry = {"docid": item.get("docid")}
            if "score" in item:
                entry["score"] = item["score"]
            # Find the text field (could be snippet, text, or nested)
            for text_key in ("snippet", "text", "contents"):
                if text_key in item:
                    full_text = str(item[text_key])
                    entry["snippet_preview"] = full_text[:snippet_preview_chars]
                    entry["snippet_length"] = len(full_text)
                    break
            preview.append(entry)
        return {"output_preview": preview, "num_results": len(parsed)}

    # get_document: single {docid, text}
    if "document" in tool_lower or (
        isinstance(parsed, dict) and "docid" in parsed and "text" in parsed
    ):
        if isinstance(parsed, dict):
            full_text = str(parsed.get("text", ""))
            return {
                "output_preview": {
                    "docid": parsed.get("docid"),
                    "text_preview": full_text[:doc_preview_chars],
                    "text_length": len(full_text),
                },
            }

    # Fallback: unknown tool or structure — truncate raw string
    return {
        "output_preview": raw_str[:doc_preview_chars],
        "raw_length": len(raw_str),
    }


def build_request_config(
    *,
    system_prompt: str | None = None,
    query_template: str | None = None,
    formatted_query: str | None = None,
    tool_definitions: list | None = None,
    searcher_type: str | None = None,
    snippet_max_tokens: int | None = None,
    k: int | None = None,
    extra: dict | None = None,
) -> dict:
    """Build the request_config dict to save alongside the run."""
    config = {
        "system_prompt": system_prompt,
        "query_template": query_template,
        "formatted_query": formatted_query,
        "tool_definitions": tool_definitions,
        "searcher_type": searcher_type,
        "snippet_max_tokens": snippet_max_tokens,
        "k": k,
    }
    if extra:
        config.update(extra)
    return config


def build_trajectory_entry(
    *,
    turn: int,
    raw_response: dict | None = None,
    usage: dict | None = None,
    stop_reason: str | None = None,
    tool_results: list[dict] | None = None,
) -> dict:
    """Build one trajectory turn entry."""
    return {
        "turn": turn,
        "raw_response": raw_response,
        "usage": usage,
        "stop_reason": stop_reason,
        "tool_results": tool_results or [],
    }


def truncate_trajectory_tool_result(
    *,
    call_id: str | None = None,
    tool_name: str | None = None,
    arguments: dict | str | None = None,
    raw_output: str | dict | list | None = None,
    snippet_preview_chars: int = 20,
    doc_preview_chars: int = 50,
) -> dict:
    """Build a single tool_result entry for the trajectory with truncated output."""
    truncated = truncate_tool_output(
        tool_name or "",
        raw_output,
        snippet_preview_chars=snippet_preview_chars,
        doc_preview_chars=doc_preview_chars,
    )
    return {
        "call_id": call_id,
        "tool_name": tool_name,
        "arguments": arguments,
        **truncated,
    }
