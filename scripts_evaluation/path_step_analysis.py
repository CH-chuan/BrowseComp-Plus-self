"""
Path-level + per-query analysis to explain the BM25/dense paradox:

1. Reconstructs every root-to-leaf path. For each path computes the *step at
   which the first evidence/gold doc was retrieved*, and the step at which
   the path's cumulative evidence/gold stops growing.

2. Per query: cross-tabs n_calls vs unique evidence found, to show whether
   the dense advantage is volume (more calls) or diversity (more unique
   docs per call).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_query(qdir: Path):
    meta = json.loads((qdir / "meta.json").read_text())
    summary_path = qdir / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    qid = str(meta.get("query_id"))
    gold = set(str(d) for d in meta.get("gold_docids", []))
    evidence = set(str(d) for d in meta.get("evidence_docids", []))

    # sample[(parent_label, sample_idx)] -> { 'retriever': ..., 'docids': set, 'ev_hits': set, 'g_hits': set }
    samples = {}
    with (qdir / "samples.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") != "sample":
                continue
            parent = rec.get("parent_label", "") or ""
            sidx = rec.get("sample_idx")
            docids = set()
            retrievers = []
            for tc in rec.get("tool_calls") or []:
                if tc.get("name") != "search":
                    continue
                args = tc.get("arguments") or {}
                retrievers.append(args.get("retriever") or "unknown")
                for r in tc.get("result") or []:
                    if r.get("docid") is not None:
                        docids.add(str(r["docid"]))
            samples[(parent, sidx)] = {
                "retrievers": retrievers,
                "docids": docids,
                "ev_hits": docids & evidence,
                "g_hits": docids & gold,
            }

    # expansions[(parent_label, step)] -> { 'B': sidx, 'W': sidx, 'R': sidx }
    expansions = {}
    with (qdir / "expansions.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            parent = rec.get("parent_label", "") or ""
            step = rec.get("step")
            picks = {lbl: info["sample_idx"] for lbl, info in (rec.get("picks") or {}).items()}
            expansions[(parent, step)] = picks

    # Reconstruct leaves
    leaves = summary.get("leaves") or []
    paths = []  # list of dicts per leaf
    for leaf in leaves:
        label = leaf.get("node_label") or ""
        # Walk the label letter by letter
        cur_parent = ""
        cum_ev = set()
        cum_g = set()
        first_ev_step = None
        first_g_step = None
        last_growth_ev_step = None
        last_growth_g_step = None
        retrievers_along_path = []
        step_hits = []  # list of (step, retriever_list, ev_hit_count_new, g_hit_count_new)
        path_ok = True
        for i, branch in enumerate(label, start=1):
            picks = expansions.get((cur_parent, i))
            if not picks or branch not in picks:
                path_ok = False
                break
            sidx = picks[branch]
            samp = samples.get((cur_parent, sidx))
            if samp is None:
                path_ok = False
                break
            retrievers_along_path.append(",".join(samp["retrievers"]) or "none")
            new_ev = samp["ev_hits"] - cum_ev
            new_g = samp["g_hits"] - cum_g
            if new_ev and first_ev_step is None:
                first_ev_step = i
            if new_g and first_g_step is None:
                first_g_step = i
            if new_ev:
                last_growth_ev_step = i
            if new_g:
                last_growth_g_step = i
            cum_ev |= samp["ev_hits"]
            cum_g |= samp["g_hits"]
            step_hits.append((i, samp["retrievers"], len(new_ev), len(new_g)))
            cur_parent = cur_parent + branch
        if not path_ok:
            continue
        paths.append({
            "qid": qid,
            "label": label,
            "n_ev": len(cum_ev),
            "n_g": len(cum_g),
            "first_ev_step": first_ev_step,
            "first_g_step": first_g_step,
            "last_growth_ev_step": last_growth_ev_step,
            "last_growth_g_step": last_growth_g_step,
            "retrievers": retrievers_along_path,
            "step_hits": step_hits,
        })

    return qid, paths, samples, expansions, evidence, gold


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True)
    args = ap.parse_args()
    root = Path(args.input_dir)

    all_paths = []
    per_query = []
    for qdir in sorted(root.iterdir()):
        if not qdir.is_dir() or not (qdir / "meta.json").exists():
            continue
        loaded = load_query(qdir)
        if loaded is None:
            continue
        qid, paths, samples, expansions, ev_set, g_set = loaded
        all_paths.extend(paths)
        # per-query call stats grouped by retriever
        pc = defaultdict(lambda: {"n_calls": 0, "ev": set(), "g": set()})
        for (parent, sidx), s in samples.items():
            for retr in s["retrievers"]:
                pc[retr]["n_calls"] += 1
            for retr in set(s["retrievers"]):
                pc[retr]["ev"] |= s["ev_hits"]
                pc[retr]["g"]  |= s["g_hits"]
        per_query.append((qid, pc, len(ev_set), len(g_set)))

    n_paths = len(all_paths)
    print(f"=== Path-level analysis ({n_paths} root-to-leaf paths across queries) ===")

    only_step1_ev = sum(1 for p in all_paths if p["last_growth_ev_step"] == 1 and p["n_ev"] > 0)
    only_step1_g  = sum(1 for p in all_paths if p["last_growth_g_step"]  == 1 and p["n_g"]  > 0)
    any_ev = sum(1 for p in all_paths if p["n_ev"] > 0)
    any_g  = sum(1 for p in all_paths if p["n_g"]  > 0)

    print(f"\nPaths that retrieved any evidence: {any_ev} ({100.0*any_ev/n_paths:.1f}%)")
    print(f"  ...of those, paths where ALL evidence came at step 1 (no further growth): "
          f"{only_step1_ev} ({100.0*only_step1_ev/any_ev:.1f}% of paths-with-evidence, "
          f"{100.0*only_step1_ev/n_paths:.1f}% of all paths)")
    print(f"\nPaths that retrieved any gold:     {any_g} ({100.0*any_g/n_paths:.1f}%)")
    print(f"  ...of those, paths where ALL gold came at step 1: "
          f"{only_step1_g} ({100.0*only_step1_g/any_g:.1f}% of paths-with-gold, "
          f"{100.0*only_step1_g/n_paths:.1f}% of all paths)")

    # Histogram of first_ev_step and last_growth_ev_step
    print(f"\nFirst evidence step (over paths with any evidence):")
    hist_first = defaultdict(int)
    hist_last = defaultdict(int)
    for p in all_paths:
        if p["first_ev_step"] is not None:
            hist_first[p["first_ev_step"]] += 1
        if p["last_growth_ev_step"] is not None:
            hist_last[p["last_growth_ev_step"]] += 1
    for s in sorted(hist_first.keys() | hist_last.keys()):
        print(f"  step {s}: first_ev={hist_first[s]:>5}  last_growth_ev={hist_last[s]:>5}")

    # Per-step new-evidence contribution: how much does each step add on average?
    new_ev_by_step = defaultdict(list)
    new_g_by_step = defaultdict(list)
    for p in all_paths:
        for (step, retrievers, new_ev, new_g) in p["step_hits"]:
            new_ev_by_step[step].append(new_ev)
            new_g_by_step[step].append(new_g)
    print(f"\nAverage NEW evidence added per step (across all paths):")
    for s in sorted(new_ev_by_step.keys()):
        vals = new_ev_by_step[s]
        n_nonzero = sum(1 for v in vals if v > 0)
        avg = sum(vals)/len(vals)
        print(f"  step {s}: avg={avg:.3f}   paths with growth at this step: {n_nonzero}/{len(vals)} ({100*n_nonzero/len(vals):.1f}%)")
    print(f"\nAverage NEW gold added per step (across all paths):")
    for s in sorted(new_g_by_step.keys()):
        vals = new_g_by_step[s]
        n_nonzero = sum(1 for v in vals if v > 0)
        avg = sum(vals)/len(vals)
        print(f"  step {s}: avg={avg:.3f}   paths with growth at this step: {n_nonzero}/{len(vals)} ({100*n_nonzero/len(vals):.1f}%)")

    # Per-query: explain the paradox — call volume vs hit diversity per retriever
    print(f"\n=== Per-query call volume vs unique evidence found ===")
    print(f"(answers: does dense win overall because it's called more, or because each call surfaces more unique docs?)")
    print(f"{'qid':>5} | {'bm calls':>8} {'bm uEv':>6} {'bm uEv/call':>11} | {'dn calls':>8} {'dn uEv':>6} {'dn uEv/call':>11}")
    print("-" * 80)
    sum_bm_calls = sum_dn_calls = 0
    sum_bm_uev = sum_dn_uev = 0
    for qid, pc, _, _ in per_query:
        b = pc.get("bm25", {"n_calls": 0, "ev": set()})
        d = pc.get("qwen3_dense", {"n_calls": 0, "ev": set()})
        bc, dc = b["n_calls"], d["n_calls"]
        be, de = len(b["ev"]), len(d["ev"])
        br = (be/bc) if bc else 0
        dr = (de/dc) if dc else 0
        sum_bm_calls += bc; sum_dn_calls += dc
        sum_bm_uev += be; sum_dn_uev += de
        print(f"{qid:>5} | {bc:>8} {be:>6} {br:>11.4f} | {dc:>8} {de:>6} {dr:>11.4f}")
    print("-" * 80)
    print(f"{'TOT':>5} | {sum_bm_calls:>8} {sum_bm_uev:>6} {sum_bm_uev/sum_bm_calls:>11.4f} | {sum_dn_calls:>8} {sum_dn_uev:>6} {sum_dn_uev/sum_dn_calls:>11.4f}")


if __name__ == "__main__":
    main()
