"""
Per-query breakdown of step-1 retriever performance.

For each query, looks at search tool calls made by step-1 samples only
(the 5 root samples), and compares bm25 vs qwen3_dense hits on
gold and evidence docs.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def step1_for_query(qdir: Path):
    meta = json.loads((qdir / "meta.json").read_text())
    gold = set(str(d) for d in meta.get("gold_docids", []))
    evidence = set(str(d) for d in meta.get("evidence_docids", []))

    per = defaultdict(lambda: {"n_calls": 0, "ev": set(), "g": set()})
    with (qdir / "samples.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") != "sample" or rec.get("step") != 1:
                continue
            for tc in rec.get("tool_calls") or []:
                if tc.get("name") != "search":
                    continue
                retr = (tc.get("arguments") or {}).get("retriever") or "unknown"
                docids = {str(r.get("docid")) for r in (tc.get("result") or []) if r.get("docid") is not None}
                per[retr]["n_calls"] += 1
                per[retr]["ev"] |= (docids & evidence)
                per[retr]["g"]  |= (docids & gold)

    return str(meta.get("query_id", qdir.name)), per


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--bm25-name", default="bm25")
    ap.add_argument("--dense-name", default="qwen3_dense")
    args = ap.parse_args()

    root = Path(args.input_dir)
    bm, dn = args.bm25_name, args.dense_name

    rows = []
    for qdir in sorted(root.iterdir()):
        if not qdir.is_dir() or not (qdir / "meta.json").exists():
            continue
        qid, per = step1_for_query(qdir)
        rows.append((qid, per))

    bm_wins_ev = dn_wins_ev = ties_ev = 0
    bm_wins_g  = dn_wins_g  = ties_g  = 0
    bm_only_called = dn_only_called = both = neither = 0

    print(f"=== Step-1 per-query retriever comparison ({len(rows)} queries) ===")
    print(f"{'qid':>5} | {'bm25 n':>6} {'bm25 ev':>7} {'bm25 g':>7} | {'dense n':>7} {'dense ev':>8} {'dense g':>8} | flag")
    print("-" * 80)
    for qid, per in rows:
        b = per.get(bm, {"n_calls": 0, "ev": set(), "g": set()})
        d = per.get(dn, {"n_calls": 0, "ev": set(), "g": set()})
        bn, dn_calls = b["n_calls"], d["n_calls"]
        be, de = len(b["ev"]), len(d["ev"])
        bg, dg = len(b["g"]), len(d["g"])

        # Coverage status
        if bn and dn_calls: both += 1
        elif bn: bm_only_called += 1
        elif dn_calls: dn_only_called += 1
        else: neither += 1

        if be > de: bm_wins_ev += 1; flag = "B>D ev"
        elif de > be: dn_wins_ev += 1; flag = "D>B ev"
        else: ties_ev += 1; flag = "tie ev"

        if bg > dg: bm_wins_g += 1
        elif dg > bg: dn_wins_g += 1
        else: ties_g += 1

        # Highlight cases where BM25 was at least as good as dense on evidence,
        # and dense was actually called (so it had a chance).
        if dn_calls and be >= de and (be > 0 or de == 0):
            extra = " <-- bm25 ≥ dense on evidence"
        else:
            extra = ""
        print(f"{qid:>5} | {bn:>6} {be:>7} {bg:>7} | {dn_calls:>7} {de:>8} {dg:>8} | {flag}{extra}")

    print()
    print(f"--- Step-1 winners (evidence) ---")
    print(f"  bm25 > dense: {bm_wins_ev}")
    print(f"  dense > bm25: {dn_wins_ev}")
    print(f"  tie:           {ties_ev}")
    print(f"--- Step-1 winners (gold) ---")
    print(f"  bm25 > dense: {bm_wins_g}")
    print(f"  dense > bm25: {dn_wins_g}")
    print(f"  tie:           {ties_g}")
    print()
    print(f"--- Step-1 retriever-call coverage ---")
    print(f"  Both retrievers called: {both}")
    print(f"  Only bm25 called:       {bm_only_called}")
    print(f"  Only dense called:      {dn_only_called}")
    print(f"  Neither called:         {neither}")


if __name__ == "__main__":
    main()
