"""Recall with a cross-encoder reranker on the top-30 hybrid candidates (ROADMAP step 4), 50 golden questions.

Run from the repo root: python scripts/eval_rerank.py [--tag v1]

Fixed before the first run:
  candidates  top-30 of `orig` (question alone) and of `fuse` (question + cached rewrite from
              results/rewrites_v1.jsonl), hyb-e5+bm25 at 192/48, depth 50
  query       the original question for the cross-encoder
  models      BAAI/bge-reranker-v2-m3 (main), cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (small baseline,
              trained on mMARCO, which has no Turkish)
  adoption    rerank goes into the pipeline only if it raises R@5 on all 50 by >= 0.06 (3 questions)
R@30 of the candidates is the ceiling a reranker can reach.
The no-rerank rows must reproduce results/rewrite_v1_per_question.csv.
Writes results/rerank_<tag>_summary.csv and results/rerank_<tag>_per_question.csv.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from golden import facts, load_golden, recall_at_k
from reranker import Reranker
from retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
KS = (1, 3, 5, 10)
N_CAND = 30
MODELS = {"bge": "BAAI/bge-reranker-v2-m3", "mmarco": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"}

sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="v1")
args = parser.parse_args()

golden = load_golden(ROOT / "eval/golden.jsonl")
rewrites = {r["id"]: r["rewrite"] for r in
            map(json.loads, (ROOT / "results/rewrites_v1.jsonl").read_text(encoding="utf-8").splitlines())}
texts = {p.stem: p.read_text(encoding="utf-8") for p in sorted((ROOT / "corpus/text").glob("*.txt"))}
retriever = HybridRetriever(texts)

cands = {}
for g in golden:
    cands[(g["id"], "orig")] = retriever.search_multi([g["question"]], k=N_CAND)
    cands[(g["id"], "fuse")] = retriever.search_multi([g["question"], rewrites[g["id"]]], k=N_CAND)

rows = {g["id"]: {"id": g["id"], "set": "dev" if int(g["id"][1:]) <= 30 else "test"} for g in golden}
for g in golden:
    for base in ("orig", "fuse"):
        ranked = [c["text"] for c in cands[(g["id"], base)]]
        for k in KS + (N_CAND,):
            rows[g["id"]][f"{base} R@{k}"] = recall_at_k(ranked, facts(g), k)

for name, model_name in MODELS.items():
    reranker, t0 = Reranker(model_name), time.time()
    for g in golden:
        # score each distinct chunk once; orig and fuse candidates overlap
        pool = list(dict.fromkeys(c["text"] for base in ("orig", "fuse") for c in cands[(g["id"], base)]))
        score = dict(zip(pool, reranker.scores(g["question"], pool)))
        for base in ("orig", "fuse"):
            order = sorted(cands[(g["id"], base)], key=lambda c: -score[c["text"]])  # stable: ties keep order
            ranked = [c["text"] for c in order]
            for k in KS:
                rows[g["id"]][f"{base}+{name} R@{k}"] = recall_at_k(ranked, facts(g), k)
    print(f"{name}: {time.time() - t0:.0f} s for {len(golden)} questions")

per_q = pd.DataFrame(rows.values())

# no-rerank rows must match the step 3 numbers question by question
saved = pd.read_csv(ROOT / "results/rewrite_v1_per_question.csv").set_index("id")
for base in ("orig", "fuse"):
    for k in KS:
        diff = (per_q.set_index("id")[f"{base} R@{k}"] - saved.loc[per_q["id"], f"{base} R@{k}"]).abs().max()
        assert diff < 1e-9, f"{base} R@{k} differs from results/rewrite_v1_per_question.csv"

systems = ["orig", "orig+mmarco", "orig+bge", "fuse", "fuse+mmarco", "fuse+bge"]
summary = []
for set_name, part in (("dev", per_q[per_q["set"] == "dev"]), ("test", per_q[per_q["set"] == "test"]), ("all", per_q)):
    for s in systems:
        row = {"set": set_name, "n": len(part), "system": s, **{f"R@{k}": part[f"{s} R@{k}"].mean() for k in KS}}
        row[f"R@{N_CAND} (ceiling)"] = part[f"{s.split('+')[0]} R@{N_CAND}"].mean()
        summary.append(row)
summary = pd.DataFrame(summary).round(2)

per_q.to_csv(ROOT / f"results/rerank_{args.tag}_per_question.csv", index=False)
summary.to_csv(ROOT / f"results/rerank_{args.tag}_summary.csv", index=False)
print(summary.to_string(index=False))
for base in ("orig", "fuse"):
    for name in MODELS:
        for k in (3, 5):
            col = f"{base}+{name} R@{k}"
            gained = per_q.loc[per_q[col] > per_q[f"{base} R@{k}"], "id"].tolist()
            lost = per_q.loc[per_q[col] < per_q[f"{base} R@{k}"], "id"].tolist()
            print(f"{base}+{name} vs {base} at R@{k}: gained {gained}, lost {lost}")
