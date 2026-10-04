"""Validate a question file (default eval/golden.jsonl) against the corpus.

Run from the repo root: python scripts/check_golden.py [eval/test.jsonl]
"""
import json, re, sys
from pathlib import Path

import nltk
from nltk.corpus import stopwords

from golden import facts, norm

TYPES = {"paraphrase", "senaryo", "dogrudan", "coklu"}


def load_stopwords():
    try:
        words = stopwords.words("turkish")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words("turkish")
    # question words appear in almost every question, so they should not count as overlap
    return set(words) | {"hangi", "nasıl", "nedir", "olan"}


STOP = load_stopwords()


def tr_lower(s):
    return s.replace("I", "ı").replace("İ", "i").lower()


def words(s):
    # crude Turkish stemming: compare the first 5 letters, so "silinmesi" and "silinir" count as the same word
    return {w[:5] for w in re.findall(r"[a-zçğıöşüâîû]+", tr_lower(s)) if len(w) > 2 and w not in STOP}


PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/golden.jsonl")
corpus = {p.stem: norm(p.read_text(encoding="utf-8")) for p in Path("corpus/text").glob("*.txt")}
errors, seen = 0, set()
for n, line in enumerate(PATH.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    try:
        q = json.loads(line)
    except json.JSONDecodeError as e:
        print(f"line {n}: invalid JSON ({e})"); errors += 1; continue
    qid = q.get("id", f"line {n}")
    problems = []
    for k in ("id", "question", "evidence", "source", "type"):
        if not q.get(k):
            problems.append(f"missing '{k}'")
    if qid in seen:
        problems.append("duplicate id")
    seen.add(qid)
    if q.get("type") and q["type"] not in TYPES:
        problems.append(f"type must be one of {sorted(TYPES)}")

    fact_list = facts(q) if q.get("evidence") else []
    doc = q.get("source", "").split(",")[0].strip()
    if doc and doc not in corpus:
        problems.append(f"unknown source '{doc}', use one of {sorted(corpus)}")
    in_source = False
    for alts in fact_list:
        for alt in alts:
            found_in = [name for name, t in corpus.items() if norm(alt) in t]
            if not found_in:
                problems.append(f"evidence not found verbatim: '{alt[:60]}...'")
            in_source = in_source or doc in found_in
    if fact_list and doc in corpus and not in_source:
        problems.append(f"no evidence is in source '{doc}'")

    overlap = 0.0
    if q.get("question") and fact_list:
        qw, ew = words(q["question"]), words(" ".join(alts[0] for alts in fact_list))
        overlap = len(qw & ew) / max(len(qw), 1)
    status = "OK " if not problems else "ERR"
    warn = "  <- many words copied from the evidence, may be too easy" if overlap >= 0.6 else ""
    alt_note = "  (+alternatives)" if any(len(a) > 1 for a in fact_list) else ""
    print(f"{status} {qid}  overlap={overlap:.2f}{warn}{alt_note}")
    for p in problems:
        print(f"      {p}")
    errors += bool(problems)
print(f"\n{len(seen)} questions, {errors} with errors")
sys.exit(1 if errors else 0)
