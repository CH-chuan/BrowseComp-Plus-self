# topics-qrels

Contains `curated50.jsonl` — a 50-query subset of `Tevatron/browsecomp-plus`.

## How `curated50.jsonl` was selected

Source: results of the full BM25 + Qwen3.5-9B run on all 830 queries (`output-test/bm25_qwen3.5-9b/`).

Random seed = 42.

| Group | Count | Criterion |
|---|---|---|
| Substring-correct | 3 | Output had a parseable `Exact Answer:` line AND gold answer (case-normalized) was a substring of it. qids: `69`, `1101`, `1147`. |
| Other with `Exact Answer:` | 21 | Output had a parseable `Exact Answer:` line but was wrong by substring match. |
| Random sample from remaining 806 | 26 | Uniform random; covers cases where the model never emitted an `Exact Answer:`. |

## Record schema

Each line is one JSON object derived from the decrypted upstream dataset, with document fields slimmed to docids only (the upstream dataset bundles full text — fetch text via `datasets.load_dataset("Tevatron/browsecomp-plus-corpus")`):

```
{
  "query_id":      str,
  "query":         str,
  "answer":        str,            # gold short answer
  "gold_docs":     list[str],      # docids containing the answer
  "evidence_docs": list[str],      # docids needed to answer (superset of gold_docs)
  "negative_docs": list[str]       # distractor docids from the upstream dataset
}
```

## Reproducing the selection

```python
import json, pathlib, random, re
random.seed(42)

def extract_exact(t):
    if not t: return None
    m = re.search(r'(?:exact\s*answer|final\s*answer)\s*[:\-]?\s*(.+?)(?:\n\s*confidence|\n\n|$)',
                  t, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip().strip('*').strip() if m else None

exact_files, other_files = [], []
for f in pathlib.Path('output-test/bm25_qwen3.5-9b').glob('*.json'):
    d = json.load(open(f))
    ot = next((r['output'] for r in d.get('result', []) if r['type']=='output_text'), None)
    (exact_files if extract_exact(ot) else other_files).append(f)

selected_qids = {
    str(json.load(open(f))['query_id'])
    for f in exact_files + random.sample(other_files, 50 - len(exact_files))
}

with open('topics-qrels/curated50.jsonl', 'w') as out:
    for line in open('data/browsecomp_plus_decrypted.jsonl'):
        d = json.loads(line)
        if str(d['query_id']) not in selected_qids:
            continue
        for key in ('gold_docs', 'evidence_docs', 'negative_docs'):
            if isinstance(d.get(key), list):
                d[key] = [x['docid'] if isinstance(x, dict) else x for x in d[key]]
        out.write(json.dumps(d) + '\n')
```
