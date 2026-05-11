"""
Analyze retrieval quality of agent run outputs against qrel ground truth.

Computes per-query retrieval metrics (evidence/gold recall) and flags cases
where the agent retrieved all needed documents but failed to produce an answer.

Usage:
    python scripts_evaluation/analyze_retrieval.py --input-dir <path>
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def load_qrel(qrel_path: Path) -> dict[str, set[str]]:
    """Load TREC qrel file into {query_id: set(docids)}."""
    qrel = defaultdict(set)
    if not qrel_path.exists():
        return dict(qrel)
    with qrel_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qrel[parts[0]].add(parts[2])
    return dict(qrel)


def extract_docids_from_tool_output(output) -> list[str]:
    """Extract docids from a single search tool call's output field, in order."""
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

    docids = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and "docid" in item:
                docids.append(str(item["docid"]))
    else:
        # Fallback regex
        for m in re.findall(r'"docid"\s*:\s*"([^"]+)"', raw_str):
            docids.append(str(m))
        for m in re.findall(r'"docid"\s*:\s*(\d+)', raw_str):
            docids.append(str(m))

    return docids


def walk_result_for_searches(result: list) -> list[list[str]]:
    """Walk result array in order, return list of docid-lists per search call."""
    search_docids_per_call = []
    for item in result or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "tool_call":
            continue
        tool_name = str(item.get("tool_name") or "")
        if "search" not in tool_name.lower() and "retrieval" not in tool_name.lower():
            continue
        docids = extract_docids_from_tool_output(item.get("output"))
        search_docids_per_call.append(docids)
    return search_docids_per_call


def searches_until_covered(
    searches: list[list[str]], target: set[str]
) -> int | None:
    """Return 1-based search call index at which target set is fully covered, or None."""
    if not target:
        return None
    seen: set[str] = set()
    for idx, docids in enumerate(searches, start=1):
        seen.update(docids)
        if target.issubset(seen):
            return idx
    return None


def has_substantive_answer(result: list) -> bool:
    """Check if result contains an output_text with >10 stripped chars."""
    for item in result or []:
        if isinstance(item, dict) and item.get("type") == "output_text":
            text = str(item.get("output", "")).strip()
            if len(text) > 10:
                return True
    return False


def analyze_run(run_data: dict, qrel_evidence: dict, qrel_golds: dict) -> dict:
    qid = str(run_data.get("query_id", ""))
    result = run_data.get("result", [])
    retrieved = set(str(d) for d in run_data.get("retrieved_docids", []))

    evidence_set = qrel_evidence.get(qid, set())
    gold_set = qrel_golds.get(qid, set())

    evidence_found = retrieved & evidence_set
    gold_found = retrieved & gold_set

    searches = walk_result_for_searches(result)
    tool_counts = run_data.get("tool_call_counts", {}) or {}
    total_searches = sum(
        v for k, v in tool_counts.items()
        if "search" in str(k).lower() or "retrieval" in str(k).lower()
    )
    # Fallback if tool_call_counts missing
    if total_searches == 0:
        total_searches = len(searches)

    return {
        "query_id": qid,
        "has_answer": has_substantive_answer(result),
        "status": run_data.get("status", ""),
        "total_searches": total_searches,
        "total_docs_retrieved": len(retrieved),
        "evidence_needed": len(evidence_set),
        "evidence_found": len(evidence_found),
        "evidence_recall": (
            len(evidence_found) / len(evidence_set) if evidence_set else 0.0
        ),
        "full_evidence": bool(evidence_set) and evidence_set.issubset(retrieved),
        "gold_needed": len(gold_set),
        "gold_found": len(gold_found),
        "gold_recall": (
            len(gold_found) / len(gold_set) if gold_set else 0.0
        ),
        "full_gold": bool(gold_set) and gold_set.issubset(retrieved),
        "searches_until_all_gold": searches_until_covered(searches, gold_set),
        "searches_until_all_evidence": searches_until_covered(searches, evidence_set),
    }


def print_summary(rows: list[dict], input_dir: Path, csv_path: Path) -> None:
    total = len(rows)
    if total == 0:
        print(f"No run files found in {input_dir}")
        return

    with_answer = sum(1 for r in rows if r["has_answer"])
    full_evidence = sum(1 for r in rows if r["full_evidence"])
    full_gold = sum(1 for r in rows if r["full_gold"])

    avg_evidence_recall = sum(r["evidence_recall"] for r in rows) / total * 100
    avg_gold_recall = sum(r["gold_recall"] for r in rows) / total * 100

    # Four buckets
    bucket_full_ev_answer = [r for r in rows if r["full_evidence"] and r["has_answer"]]
    bucket_full_ev_no_answer = [r for r in rows if r["full_evidence"] and not r["has_answer"]]
    bucket_partial_ev_answer = [r for r in rows if not r["full_evidence"] and r["has_answer"]]
    bucket_partial_ev_no_answer = [r for r in rows if not r["full_evidence"] and not r["has_answer"]]

    print(f"=== Retrieval Analysis: {input_dir} ===")
    print(f"Total queries:       {total}")
    print(f"With answer:         {with_answer} ({with_answer/total*100:.1f}%)")
    print(f"Without answer:      {total - with_answer} ({(total-with_answer)/total*100:.1f}%)")
    print(f"Full evidence:       {full_evidence} ({full_evidence/total*100:.1f}%)")
    print(f"Full gold:           {full_gold} ({full_gold/total*100:.1f}%)")
    print(f"Avg evidence recall: {avg_evidence_recall:.1f}%")
    print(f"Avg gold recall:     {avg_gold_recall:.1f}%")
    print()
    print("=== Buckets ===")
    print(f"[1] Full evidence + answer (best case):      {len(bucket_full_ev_answer)}")
    for r in bucket_full_ev_answer:
        print(f"      qid={r['query_id']}  evidence={r['evidence_found']}/{r['evidence_needed']}  gold={r['gold_found']}/{r['gold_needed']}  searches={r['total_searches']}")
    print(f"[2] Full evidence + NO answer (reasoning fail): {len(bucket_full_ev_no_answer)}")
    for r in bucket_full_ev_no_answer:
        print(f"      qid={r['query_id']}  evidence={r['evidence_found']}/{r['evidence_needed']}  gold={r['gold_found']}/{r['gold_needed']}  searches={r['total_searches']}")
    print(f"[3] Partial evidence + answer (guessing):    {len(bucket_partial_ev_answer)}")
    for r in bucket_partial_ev_answer:
        print(f"      qid={r['query_id']}  evidence={r['evidence_found']}/{r['evidence_needed']}  gold={r['gold_found']}/{r['gold_needed']}  searches={r['total_searches']}")
    print(f"[4] Partial/no evidence + no answer:         {len(bucket_partial_ev_no_answer)}")

    print()
    print(f"CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory with run_*.json files")
    parser.add_argument(
        "--qrel-evidence",
        default="topics-qrels/qrel_evidence.txt",
        help="Path to qrel evidence file",
    )
    parser.add_argument(
        "--qrel-golds",
        default="topics-qrels/qrel_golds.txt",
        help="Path to qrel golds file",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    qrel_evidence = load_qrel(Path(args.qrel_evidence))
    qrel_golds = load_qrel(Path(args.qrel_golds))

    run_files = sorted(input_dir.glob("run_*.json"))
    if not run_files:
        print(f"No run_*.json files found in {input_dir}", file=sys.stderr)
        sys.exit(0)

    rows = []
    for f in run_files:
        try:
            with f.open() as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"Warning: could not read {f}: {e}", file=sys.stderr)
            continue
        rows.append(analyze_run(data, qrel_evidence, qrel_golds))

    csv_path = input_dir / "retrieval_analysis.csv"
    fieldnames = [
        "query_id",
        "has_answer",
        "status",
        "total_searches",
        "total_docs_retrieved",
        "evidence_needed",
        "evidence_found",
        "evidence_recall",
        "full_evidence",
        "gold_needed",
        "gold_found",
        "gold_recall",
        "full_gold",
        "searches_until_all_gold",
        "searches_until_all_evidence",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_out = dict(row)
            for k in ("searches_until_all_gold", "searches_until_all_evidence"):
                if row_out[k] is None:
                    row_out[k] = "NA"
            row_out["evidence_recall"] = f"{row_out['evidence_recall']:.4f}"
            row_out["gold_recall"] = f"{row_out['gold_recall']:.4f}"
            writer.writerow(row_out)

    print_summary(rows, input_dir, csv_path)


if __name__ == "__main__":
    main()
