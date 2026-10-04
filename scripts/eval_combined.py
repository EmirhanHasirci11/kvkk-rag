"""Recall on all 50 questions (dev + test) at 192/48, saved to results/combined_50q.csv.

BM25 is computed here (no embeddings needed). The e5 and hyb-e5+bm25 values are the mean of the
per-question values already saved in results/ (notebook 02 run), so no model is loaded.

Run from the repo root: python scripts/eval_combined.py
"""
from pathlib import Path

import pandas as pd

from bm25 import BM25
from chunking import chunk_words
from golden import facts, load_golden, recall_at_k
from textproc import tokenize

ROOT = Path(__file__).resolve().parents[1]
SIZE, OVERLAP, DEPTH = 192, 48, 50
KS = (1, 3, 5, 10)

golden = load_golden(ROOT / "eval/golden.jsonl")
docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted((ROOT / "corpus/text").glob("*.txt"))}
chunks = [c for t in docs.values() for c in chunk_words(t, SIZE, OVERLAP)]

rows = []
for stem in ("none", "prefix5"):
    bm = BM25([tokenize(c, stem) for c in chunks])
    ranked = {g["id"]: [chunks[i] for i, _ in bm.search(tokenize(g["question"], stem), k=DEPTH)] for g in golden}
    rows.append({"set": "dev+test", "model": f"bm25-{stem}",
                 **{f"R@{k}": sum(recall_at_k(ranked[g["id"]], facts(g), k) for g in golden) / len(golden) for k in KS}})

# e5 and hybrid: notebook 02 saved all 50 questions (30 dev + 20 test) with R@1/3/5/10 per question
per_q = pd.read_csv(ROOT / "results/test_v1_per_question.csv")
assert per_q["id"].tolist() == [g["id"] for g in golden], "question ids differ from eval/golden.jsonl"

# cross-check: the dev rows must agree with the v4 per-question R@10
v4 = pd.read_csv(ROOT / "results/v4_per_question.csv", index_col=0)
dev = per_q[per_q["set"] == "dev"].set_index("id")
for model in ("e5", "hyb-e5+bm25"):
    assert (dev[f"{model} R@10"] - v4.loc[dev.index, f"{model}-192/48"]).abs().max() < 0.006, f"{model}: dev rows differ from v4"

for model in ("e5", "hyb-e5+bm25"):
    rows.append({"set": "dev+test", "model": model, **{f"R@{k}": per_q[f"{model} R@{k}"].mean() for k in KS}})

out = pd.DataFrame(rows).round(2)
out.to_csv(ROOT / "results/combined_50q.csv", index=False)
print(out.to_string(index=False))
