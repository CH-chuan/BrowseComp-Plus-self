"""
Compare BM25 vs dense retriever performance broken down by tree step (depth).

For each query directory under --input-dir, walks samples.jsonl and aggregates
search-tool-call hits per (step, retriever) against gold/evidence docs.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def analyze_query(qdir: Path):
    meta_path = qdir / "meta.json"
    samples_path = qdir / "samples.jsonl"
    if not meta_path.exists() or not samples_path.exists():
        return None

    meta = json.loads(meta_path.read_text())
    gold = set(str(d) for d in meta.get("gold_docids", []))
    evidence = set(str(d) for d in meta.get("evidence_docids", []))

    # (step, retriever) -> stats
    stats = defaultdict(lambda: {
        "n_calls": 0,
        "calls_with_evidence": 0,
        "calls_with_gold": 0,
        "evidence_hits_per_query": set(),
        "gold_hits_per_query": set(),
    })

    with samples_path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") != "sample":
                continue
            step = rec.get("step")
            for tc in rec.get("tool_calls") or []:
                if tc.get("name") != "search":
                    continue
                retriever = (tc.get("arguments") or {}).get("retriever") or "unknown"
                docids = {str(r.get("docid")) for r in (tc.get("result") or []) if r.get("docid") is not None}
                key = (step, retriever)
                stats[key]["n_calls"] += 1
                gh = docids & gold
                eh = docids & evidence
                if gh:
                    stats[key]["calls_with_gold"] += 1
                if eh:
                    stats[key]["calls_with_evidence"] += 1
                stats[key]["evidence_hits_per_query"] |= eh
                stats[key]["gold_hits_per_query"] |= gh

    return {
        "qid": str(meta.get("query_id", qdir.name)),
        "stats": stats,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--bm25-name", default="bm25")
    ap.add_argument("--dense-name", default="qwen3_dense")
    args = ap.parse_args()

    root = Path(args.input_dir)
    if not root.is_dir():
        print(f"Not a dir: {root}", file=sys.stderr)
        sys.exit(1)

    # Aggregate across all queries: (step, retriever) -> totals
    agg = defaultdict(lambda: {
        "n_calls": 0,
        "calls_with_evidence": 0,
        "calls_with_gold": 0,
        "sum_unique_evidence": 0,  # sum over queries of |unique evidence found at this step by this retriever|
        "sum_unique_gold": 0,
    })

    qcount = 0
    for qdir in sorted(root.iterdir()):
        if not qdir.is_dir():
            continue
        r = analyze_query(qdir)
        if not r:
            continue
        qcount += 1
        for (step, retr), s in r["stats"].items():
            a = agg[(step, retr)]
            a["n_calls"] += s["n_calls"]
            a["calls_with_evidence"] += s["calls_with_evidence"]
            a["calls_with_gold"] += s["calls_with_gold"]
            a["sum_unique_evidence"] += len(s["evidence_hits_per_query"])
            a["sum_unique_gold"] += len(s["gold_hits_per_query"])

    bm = args.bm25_name
    dn = args.dense_name
    steps = sorted({s for (s, _) in agg.keys() if s is not None})

    print(f"=== Per-step retriever comparison: {root.name} ({qcount} queries) ===")
    print()

    # Per-call totals + hit rates
    header = f"{'step':>4} | {'bm25 calls':>10} {'bm25 ev%':>9} {'bm25 g%':>8} {'bm25 uEv':>9} {'bm25 uG':>8} | {'dense calls':>11} {'dense ev%':>10} {'dense g%':>9} {'dense uEv':>10} {'dense uG':>9}"
    print(header)
    print("-" * len(header))
    for st in steps:
        ab = agg.get((st, bm), {"n_calls": 0, "calls_with_evidence": 0, "calls_with_gold": 0, "sum_unique_evidence": 0, "sum_unique_gold": 0})
        ad = agg.get((st, dn), {"n_calls": 0, "calls_with_evidence": 0, "calls_with_gold": 0, "sum_unique_evidence": 0, "sum_unique_gold": 0})
        be_rate = (100.0 * ab["calls_with_evidence"] / ab["n_calls"]) if ab["n_calls"] else 0.0
        bg_rate = (100.0 * ab["calls_with_gold"]     / ab["n_calls"]) if ab["n_calls"] else 0.0
        de_rate = (100.0 * ad["calls_with_evidence"] / ad["n_calls"]) if ad["n_calls"] else 0.0
        dg_rate = (100.0 * ad["calls_with_gold"]     / ad["n_calls"]) if ad["n_calls"] else 0.0
        print(f"{st:>4} | {ab['n_calls']:>10} {be_rate:>8.1f}% {bg_rate:>7.1f}% {ab['sum_unique_evidence']:>9} {ab['sum_unique_gold']:>8} | {ad['n_calls']:>11} {de_rate:>9.1f}% {dg_rate:>8.1f}% {ad['sum_unique_evidence']:>10} {ad['sum_unique_gold']:>9}")

    print()
    print("Columns: ev% = % of calls retrieving any evidence doc, g% = % retrieving any gold doc,")
    print("         uEv = sum across queries of unique evidence docs found at this step by this retriever (same for uG/gold).")


if __name__ == "__main__":
    main()
