"""Check that the pgvector backend gives the pipeline the same chunks as the numpy one (ROADMAP step 7). No LLM calls.

Run from the repo root after `docker compose up -d`: python scripts/check_pgvector.py
Questions: every golden, test and abstention question with its cached rewrite (results/rewrites_*.jsonl).
Pass: for every question the fused top-30 candidates are the same set and the reranked top-5 (what the
LLM sees) is identical. The dense top-50 order is reported too: pgvector computes distances in its own
float arithmetic, so two chunks whose cosines differ by ~1e-7 can swap places there.
"""
import json
import sys
from pathlib import Path

import numpy as np

from pipeline import N_CANDIDATES, STEM, TOP_K
from reranker import Reranker
from retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


rewrites = {r["id"]: r["rewrite"] for p in sorted((ROOT / "results").glob("rewrites_*.jsonl")) for r in load(p)}
# test_v1 is a subset of golden, so keep each id once
questions = list({q["id"]: q for p in sorted((ROOT / "eval").glob("*.jsonl")) for q in load(p)}.values())

texts = {p.stem: p.read_text(encoding="utf-8") for p in sorted((ROOT / "corpus/text").glob("*.txt"))}
np_ret = HybridRetriever(texts, stem=STEM, dense="numpy")
pg_ret = HybridRetriever(texts, stem=STEM, dense="pgvector")
reranker = Reranker()

stored = np.array([r[0].to_numpy() for r in pg_ret.index.conn.execute("SELECT embedding FROM chunks ORDER BY id").fetchall()])
print(f"chunks: numpy {len(np_ret.vecs)}, pgvector {len(stored)}, "
      f"max |vector diff| {np.abs(stored - np_ret.vecs).max():.2e}")


def key(chunks):
    return [(c["doc"], c["text"]) for c in chunks]


dense_diff, fused_order, fused_set, top5_diff, n_queries = [], [], [], [], 0
for q in questions:
    queries = [q["question"]] + ([rewrites[q["id"]]] if q["id"] in rewrites else [])
    for text in queries:
        n_queries += 1
        a, b = np_ret.ranked_lists(text)[0], pg_ret.ranked_lists(text)[0]
        if a != b:
            dense_diff.append((q["id"], text, a, b))
    ca, cb = np_ret.search_multi(queries, k=N_CANDIDATES), pg_ret.search_multi(queries, k=N_CANDIDATES)
    if key(ca) != key(cb):
        fused_order.append(q["id"])
        if set(key(ca)) != set(key(cb)):
            fused_set.append(q["id"])
        if key(reranker.rerank(q["question"], ca, k=TOP_K)) != key(reranker.rerank(q["question"], cb, k=TOP_K)):
            top5_diff.append(q["id"])

print(f"questions {len(questions)}, with rewrite {sum(q['id'] in rewrites for q in questions)}, queries {n_queries}")
print(f"dense top-{np_ret.depth} order differs: {len(dense_diff)} / {n_queries}")
for qid, text, a, b in dense_diff:
    i = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
    qv = np_ret.model.encode(["query: " + text], normalize_embeddings=True, show_progress_bar=False)[0]
    print(f"  {qid}: {text[:80]}")
    print(f"    first difference at rank {i + 1}: numpy {a[i:i + 3]}, pgvector {b[i:i + 3]}")
    print("    cosine (numpy): " + ", ".join(f"{c}: {np_ret.vecs[c] @ qv:.9f}" for c in sorted(set(a[i:i + 3] + b[i:i + 3]))))
print(f"fused top-{N_CANDIDATES} order differs: {len(fused_order)} / {len(questions)} {fused_order}")
print(f"fused top-{N_CANDIDATES} set differs: {len(fused_set)} / {len(questions)} {fused_set}")
print(f"reranked top-{TOP_K} differs: {len(top5_diff)} / {len(questions)} {top5_diff}")
sys.exit(1 if fused_set or top5_diff else 0)
