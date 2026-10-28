"""
Compare BM25 vs dense retriever performance across a sampling-tree run.

For each query directory under --input-dir (containing samples.jsonl + meta.json),
counts per-retriever search calls and per-call hits against the query's
gold_docids and evidence_docids. Reports both per-call totals and a per-query
"winner" tally (which retriever found more evidence / gold for that query).
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
    qid = str(meta.get("query_id", qdir.name))
    gold = set(str(d) for d in meta.get("gold_docids", []))
    evidence = set(str(d) for d in meta.get("evidence_docids", []))

    # Per-retriever aggregates for this query
    per = defaultdict(lambda: {
        "n_calls": 0,
        "calls_with_gold": 0,
        "calls_with_evidence": 0,
        "gold_hits": set(),
        "evidence_hits": set(),
    })

    with samples_path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") != "sample":
                continue
            for tc in rec.get("tool_calls") or []:
                if tc.get("name") != "search":
                    continue
                args = tc.get("arguments") or {}
                retriever = args.get("retriever") or "unknown"
                results = tc.get("result") or []
                docids = {str(r.get("docid")) for r in results if r.get("docid") is not None}
                bucket = per[retriever]
                bucket["n_calls"] += 1
                gh = docids & gold
                eh = docids & evidence
                if gh:
                    bucket["calls_with_gold"] += 1
                if eh:
                    bucket["calls_with_evidence"] += 1
                bucket["gold_hits"] |= gh
                bucket["evidence_hits"] |= eh

    return {
        "qid": qid,
        "n_gold": len(gold),
        "n_evidence": len(evidence),
        "per": {k: {
            "n_calls": v["n_calls"],
            "calls_with_gold": v["calls_with_gold"],
            "calls_with_evidence": v["calls_with_evidence"],
            "unique_gold_found": len(v["gold_hits"]),
            "unique_evidence_found": len(v["evidence_hits"]),
        } for k, v in per.items()},
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

    rows = []
    for qdir in sorted(root.iterdir()):
        if not qdir.is_dir():
            continue
        r = analyze_query(qdir)
        if r:
            rows.append(r)

    if not rows:
        print("No query dirs with samples.jsonl found.")
        sys.exit(0)

    bm = args.bm25_name
    dn = args.dense_name

    # Aggregate per-call totals
    tot = {bm: defaultdict(int), dn: defaultdict(int)}
    for r in rows:
        for name in (bm, dn):
            p = r["per"].get(name)
            if not p:
                continue
            for k, v in p.items():
                tot[name][k] += v

    # Per-query winners (by unique evidence found, then unique gold, then calls_with_evidence)
    def score(p):
        if not p:
            return (0, 0, 0)
        return (p["unique_evidence_found"], p["unique_gold_found"], p["calls_with_evidence"])

    bm_wins_ev = dn_wins_ev = ties_ev = 0
    bm_wins_gold = dn_wins_gold = ties_gold = 0
    detailed = []
    for r in rows:
        pb = r["per"].get(bm, {"n_calls": 0, "calls_with_gold": 0, "calls_with_evidence": 0, "unique_gold_found": 0, "unique_evidence_found": 0})
        pd = r["per"].get(dn, {"n_calls": 0, "calls_with_gold": 0, "calls_with_evidence": 0, "unique_gold_found": 0, "unique_evidence_found": 0})

        eb, ed = pb["unique_evidence_found"], pd["unique_evidence_found"]
        if eb > ed: bm_wins_ev += 1
        elif ed > eb: dn_wins_ev += 1
        else: ties_ev += 1

        gb, gd = pb["unique_gold_found"], pd["unique_gold_found"]
        if gb > gd: bm_wins_gold += 1
        elif gd > gb: dn_wins_gold += 1
        else: ties_gold += 1

        detailed.append((r["qid"], pb, pd))

    print(f"=== Retriever comparison: {root.name} ===")
    print(f"Queries analyzed: {len(rows)}")
    print()
    print(f"--- Per-call totals across all queries ---")
    print(f"{'metric':<26} {bm:>12} {dn:>14}")
    for k in ("n_calls", "calls_with_evidence", "calls_with_gold", "unique_evidence_found", "unique_gold_found"):
        print(f"{k:<26} {tot[bm][k]:>12} {tot[dn][k]:>14}")
    print()
    print(f"--- Per-query winners ---")
    print(f"Evidence (unique evidence docs found per query):")
    print(f"  {bm} > {dn}:   {bm_wins_ev}")
    print(f"  {dn} > {bm}:   {dn_wins_ev}")
    print(f"  tie:           {ties_ev}")
    print()
    print(f"Gold (unique gold docs found per query):")
    print(f"  {bm} > {dn}:   {bm_wins_gold}")
    print(f"  {dn} > {bm}:   {dn_wins_gold}")
    print(f"  tie:           {ties_gold}")
    print()
    print(f"--- Per-query detail (qid | {bm} ev/gold (calls) | {dn} ev/gold (calls)) ---")
    for qid, pb, pd in detailed:
        marker = "  "
        if pb["unique_evidence_found"] > pd["unique_evidence_found"]:
            marker = "B*"
        elif pd["unique_evidence_found"] > pb["unique_evidence_found"]:
            marker = "D*"
        print(f"  {marker} {qid:>5} | {bm}: {pb['unique_evidence_found']}/{pb['unique_gold_found']} ({pb['n_calls']} calls) | {dn}: {pd['unique_evidence_found']}/{pd['unique_gold_found']} ({pd['n_calls']} calls)")


if __name__ == "__main__":
    main()
