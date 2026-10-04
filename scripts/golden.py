"""Shared helpers for eval/golden.jsonl.

The evidence field can take three shapes:
  "sentence"                    one fact
  ["s1", "s2"]                  two facts, both must be retrieved
  [["s1", "s1_alt"], "s2"]      two facts, the first one counts if either s1 or s1_alt is retrieved
"""
import json
from pathlib import Path


def norm(s: str) -> str:
    return " ".join(s.split())


def facts(question: dict) -> list[list[str]]:
    """Return the evidence as a list of facts, each fact a list of acceptable alternatives."""
    ev = question["evidence"]
    if isinstance(ev, str):
        return [[ev]]
    return [[item] if isinstance(item, str) else list(item) for item in ev]


def load_golden(path="eval/golden.jsonl") -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def recall_at_k(ranked_texts: list[str], fact_list: list[list[str]], k: int) -> float:
    """Share of facts with at least one alternative fully contained in the top k chunks."""
    top = [norm(t) for t in ranked_texts[:k]]
    found = sum(any(norm(alt) in t for alt in alts for t in top) for alts in fact_list)
    return found / len(fact_list)
