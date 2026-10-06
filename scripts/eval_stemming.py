"""BM25 stemmers compared alone, in the hybrid and in the full pipeline (ROADMAP step 5), 50 golden questions.

Run from the repo root: python scripts/eval_stemming.py [--tag v1]

Fixed before the first run:
  stemmers    bm25-none (no stemming), bm25-p5 (first 5 letters, the current one), bm25-snow (Turkish
              Snowball), bm25-zeyrek (zeyrek lemma of the first analysis; unknown words stay as they are)
  levels      BM25 alone; orig hybrid (e5 + BM25); pipeline = fuse (question + cached rewrite from
              results/rewrites_v1.jsonl) top-30 -> bge-reranker-v2-m3 -> ranked list
  adoption    a stemmer replaces p5 only if it raises the pipeline R@5 by >= 0.06 (3 questions)
The p5 rows must reproduce results/rerank_v1_per_question.csv (orig, fuse+bge).
Writes results/stemming_<tag>_summary.csv and results/stemming_<tag>_per_question.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from bm25 import BM25
from golden import facts, load_golden, recall_at_k
from reranker import Reranker
from retriever import HybridRetriever
from textproc import tokenize

ROOT = Path(__file__).resolve().parents[1]
KS = (1, 3, 5, 10)
N_CAND = 30
STEMS = {"none": "none", "p5": "prefix5", "snow": "snow", "zeyrek": "zeyrek"}

sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="v1")
args = parser.parse_args()

golden = load_golden(ROOT / "eval/golden.jsonl")
rewrites = {r["id"]: r["rewrite"] for r in
            map(json.loads, (ROOT / "results/rewrites_v1.jsonl").read_text(encoding="utf-8").splitlines())}
texts = {p.stem: p.read_text(encoding="utf-8") for p in sorted((ROOT / "corpus/text").glob("*.txt"))}
reranker = Reranker()
rows = {g["id"]: {"id": g["id"], "set": "dev" if int(g["id"][1:]) <= 30 else "test"} for g in golden}


def record(qid, system, ranked, ks):
    g = next(q for q in golden if q["id"] == qid)
    for k in ks:
        rows[qid][f"{system} R@{k}"] = recall_at_k(ranked, facts(g), k)


for name, stem in STEMS.items():
    retriever = HybridRetriever(texts, stem=stem)
    bm = BM25([tokenize(c["text"], stem) for c in retriever.chunks])
    for g in golden:
        alone = [retriever.chunks[i]["text"] for i, _ in bm.search(tokenize(g["question"], stem), k=N_CAND)]
        record(g["id"], f"bm25-{name}", alone, KS + (N_CAND,))
        record(g["id"], f"hyb-{name}", [c["text"] for c in retriever.search(g["question"], k=N_CAND)], KS + (N_CAND,))
        cands = retriever.search_multi([g["question"], rewrites[g["id"]]], k=N_CAND)
        record(g["id"], f"fuse-{name}", [c["text"] for c in cands], (N_CAND,))
        record(g["id"], f"pipe-{name}", [c["text"] for c in reranker.rerank(g["question"], cands)], KS)
    print(f"{name} done")

per_q = pd.DataFrame(rows.values())

saved = pd.read_csv(ROOT / "results/rerank_v1_per_question.csv").set_index("id")
for mine, theirs in (("hyb-p5", "orig"), ("pipe-p5", "fuse+bge")):
    for k in KS:
        diff = (per_q.set_index("id")[f"{mine} R@{k}"] - saved.loc[per_q["id"], f"{theirs} R@{k}"]).abs().max()
        assert diff < 1e-9, f"{mine} R@{k} differs from {theirs} in results/rerank_v1_per_question.csv"

summary = []
for level, ceiling in (("bm25", "bm25"), ("hyb", "hyb"), ("pipe", "fuse")):
    for name in STEMS:
        s = f"{level}-{name}"
        summary.append({"level": level, "stemmer": f"bm25-{name}",
                        **{f"R@{k}": per_q[f"{s} R@{k}"].mean() for k in KS},
                        f"R@{N_CAND}": per_q[f"{ceiling}-{name} R@{N_CAND}"].mean()})
summary = pd.DataFrame(summary).round(2)

per_q.to_csv(ROOT / f"results/stemming_{args.tag}_per_question.csv", index=False)
summary.to_csv(ROOT / f"results/stemming_{args.tag}_summary.csv", index=False)
print(summary.to_string(index=False))
for name in ("none", "snow", "zeyrek"):
    for level in ("hyb", "pipe"):
        a, b = per_q[f"{level}-{name} R@5"], per_q[f"{level}-p5 R@5"]
        print(f"{level}-{name} vs {level}-p5 at R@5: gained {per_q.loc[a > b, 'id'].tolist()}, "
              f"lost {per_q.loc[a < b, 'id'].tolist()}")
