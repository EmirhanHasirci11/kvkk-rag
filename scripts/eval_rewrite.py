"""Recall with and without LLM query rewriting (ROADMAP step 3), all 50 golden questions.

Run from the repo root: python scripts/eval_rewrite.py [--tag v1]

Rewrites are cached in results/rewrites_<tag>.jsonl: one LLM call per question on the first run,
reused afterwards. Systems, all hyb-e5+bm25 at 192/48, depth 50:
  orig   the question alone (same as before; checked against results/test_v1_per_question.csv)
  rw     the rewrite alone
  fuse   RRF over the dense and BM25 lists of both the question and the rewrite
Writes results/rewrite_<tag>_summary.csv and results/rewrite_<tag>_per_question.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from golden import facts, load_golden, recall_at_k
from retriever import HybridRetriever
from rewrite import rewrite

ROOT = Path(__file__).resolve().parents[1]
KS = (1, 3, 5, 10)
SYSTEMS = ("orig", "rw", "fuse")

sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="v1")
args = parser.parse_args()

golden = load_golden(ROOT / "eval/golden.jsonl")
cache = ROOT / f"results/rewrites_{args.tag}.jsonl"
rewrites = {}
if cache.exists():
    rewrites = {r["id"]: r for r in map(json.loads, cache.read_text(encoding="utf-8").splitlines())}
with cache.open("a", encoding="utf-8") as f:
    for g in golden:
        if g["id"] not in rewrites:
            out = rewrite(g["question"])
            rewrites[g["id"]] = {"id": g["id"], "question": g["question"], "rewrite": out["text"],
                                 "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"],
                                 "cost_usd": out["cost_usd"]}
            f.write(json.dumps(rewrites[g["id"]], ensure_ascii=False) + "\n")
            f.flush()

texts = {p.stem: p.read_text(encoding="utf-8") for p in sorted((ROOT / "corpus/text").glob("*.txt"))}
retriever = HybridRetriever(texts)

rows = []
for g in golden:
    rw = rewrites[g["id"]]["rewrite"]
    queries = {"orig": [g["question"]], "rw": [rw], "fuse": [g["question"], rw]}
    row = {"id": g["id"], "set": "dev" if int(g["id"][1:]) <= 30 else "test"}
    for system, qs in queries.items():
        ranked = [c["text"] for c in retriever.search_multi(qs, k=max(KS))]
        for k in KS:
            row[f"{system} R@{k}"] = recall_at_k(ranked, facts(g), k)
    rows.append(row)
per_q = pd.DataFrame(rows)

# orig must reproduce the saved hyb-e5+bm25 numbers question by question
saved = pd.read_csv(ROOT / "results/test_v1_per_question.csv").set_index("id")
for k in KS:
    diff = (per_q.set_index("id")[f"orig R@{k}"] - saved.loc[per_q["id"], f"hyb-e5+bm25 R@{k}"]).abs().max()
    assert diff < 1e-9, f"orig R@{k} differs from results/test_v1_per_question.csv"

summary = []
for name, part in (("dev", per_q[per_q["set"] == "dev"]), ("test", per_q[per_q["set"] == "test"]), ("all", per_q)):
    for system in SYSTEMS:
        summary.append({"set": name, "n": len(part), "system": system,
                        **{f"R@{k}": part[f"{system} R@{k}"].mean() for k in KS}})
summary = pd.DataFrame(summary).round(2)

per_q.to_csv(ROOT / f"results/rewrite_{args.tag}_per_question.csv", index=False)
summary.to_csv(ROOT / f"results/rewrite_{args.tag}_summary.csv", index=False)
print(summary.to_string(index=False))

for system in ("rw", "fuse"):
    for k in (5, 10):
        gained = per_q.loc[per_q[f"{system} R@{k}"] > per_q[f"orig R@{k}"], "id"].tolist()
        lost = per_q.loc[per_q[f"{system} R@{k}"] < per_q[f"orig R@{k}"], "id"].tolist()
        print(f"{system} vs orig at R@{k}: gained {gained}, lost {lost}")
print(f"rewrite cost: ${sum(r['cost_usd'] for r in rewrites.values()):.4f}")
