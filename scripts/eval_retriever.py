"""Recall of HybridRetriever on a question file.

Run from the repo root: python scripts/eval_retriever.py eval/golden.jsonl [--limit 30]
"""
import argparse
from pathlib import Path

from golden import facts, load_golden, recall_at_k
from retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
KS = (1, 3, 5, 10)

parser = argparse.ArgumentParser()
parser.add_argument("path", help="jsonl file with questions")
parser.add_argument("--limit", type=int, help="use only the first N questions")
args = parser.parse_args()

questions = load_golden(args.path)[:args.limit]
texts = {p.stem: p.read_text(encoding="utf-8") for p in sorted((ROOT / "corpus/text").glob("*.txt"))}
retriever = HybridRetriever(texts)

ranked = {g["id"]: [r["text"] for r in retriever.search(g["question"], k=max(KS))] for g in questions}
print(f"{len(questions)} questions from {args.path}")
for k in KS:
    mean = sum(recall_at_k(ranked[g["id"]], facts(g), k) for g in questions) / len(questions)
    print(f"R@{k} = {mean:.2f}")
