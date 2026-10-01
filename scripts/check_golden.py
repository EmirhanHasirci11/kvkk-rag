"""Validate eval/golden.jsonl against the corpus.

Run from the repo root: python scripts/check_golden.py
"""
import json, re, sys
from pathlib import Path

import nltk
from nltk.corpus import stopwords

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

def norm(s):
    return re.sub(r"\s+", " ", s).strip()

def words(s):
    # crude Turkish stemming: compare the first 5 letters, so "silinmesi" and "silinir" count as the same word
    return {w[:5] for w in re.findall(r"[a-zçğıöşüâîû]+", tr_lower(s)) if len(w) > 2 and w not in STOP}

corpus = {p.stem: norm(p.read_text(encoding="utf-8")) for p in Path("corpus/text").glob("*.txt")}
errors, seen = 0, set()
for n, line in enumerate(Path("eval/golden.jsonl").read_text(encoding="utf-8").splitlines(), 1):
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
    evs = q.get("evidence") or []
    evs = [evs] if isinstance(evs, str) else evs
    doc = q.get("source", "").split(",")[0].strip()
    if doc and doc not in corpus:
        problems.append(f"unknown source '{doc}', use one of {sorted(corpus)}")
    for ev in evs:
        if doc in corpus and norm(ev) in corpus[doc]:
            continue
        found_in = [name for name, t in corpus.items() if norm(ev) in t]
        if found_in:
            problems.append(f"evidence is in {found_in}, not in source '{doc}'")
        else:
            problems.append(f"evidence not found verbatim: '{ev[:60]}...'")
    overlap = 0.0
    if q.get("question") and evs:
        qw, ew = words(q["question"]), words(" ".join(evs))
        overlap = len(qw & ew) / max(len(qw), 1)
    status = "OK " if not problems else "ERR"
    warn = "  <- many words copied from the evidence, may be too easy" if overlap >= 0.6 else ""
    print(f"{status} {qid}  overlap={overlap:.2f}{warn}")
    for p in problems:
        print(f"      {p}")
    errors += bool(problems)
print(f"\n{len(seen)} questions, {errors} with errors")
sys.exit(1 if errors else 0)