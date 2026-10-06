"""Final held-out test (ROADMAP step 6): one run of the frozen pipeline on new questions.

Run from the repo root, after committing the question file:
    python scripts/run_heldout.py eval/test_v2.jsonl [--abstention eval/abstention_v2.jsonl] [--tag heldout_v2]

Frozen here, not taken from the command line:
  FINAL     v4: fuse (question + LLM rewrite) top-30 -> bge-reranker-v2-m3 -> top-5, bm25-snow, prompt v1
  BASELINE  v0: question alone, hybrid top-5, bm25-p5, prompt v0
Steps:
  1. scripts/check_golden.py must pass; ids and questions must not overlap eval/golden.jsonl;
     the question files must be committed and unchanged (written before any result exists)
  2. refuses to run if results/<tag>_* already exists: one run only
  3. rewrites every question once (results/rewrites_<tag>.jsonl)
  4. recall of every retrieval system from steps 2-5 -> results/<tag>_retrieval_{summary,per_question}.csv
  5. answers of BASELINE and FINAL -> results/answers_<tag>_{v0,final}.jsonl + grading CSVs
     (and the same for the abstention file, if given)
  6. config and git commit -> results/<tag>_config.json
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from golden import facts, load_golden, recall_at_k
from reranker import Reranker
from retriever import HybridRetriever
from run_answers import cached_rewrite

ROOT = Path(__file__).resolve().parents[1]
KS = (1, 3, 5, 10)
N_CAND = 30
FINAL = {"rewrite": True, "rerank": True, "stem": "snow", "prompt": "v1"}
BASELINE = {"rewrite": False, "rerank": False, "stem": "prefix5", "prompt": "v0"}
# retrieval systems compared in steps 2-5: (BM25 stemmer, use the rewrite, rerank)
SYSTEMS = {
    "orig": ("prefix5", False, False),
    "rw-only": ("prefix5", None, False),
    "fuse": ("prefix5", True, False),
    "orig+bge": ("prefix5", False, True),
    "fuse+bge": ("prefix5", True, True),
    "fuse+bge+snow (final)": ("snow", True, True),
}


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def run(cmd: list[str]):
    print(">", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)


def answer_args(cfg: dict, rewrites: str) -> list[str]:
    a = ["--prompt", cfg["prompt"], "--stem", cfg["stem"]]
    if cfg["rewrite"]:
        a += ["--rewrite", "--rewrites", rewrites]
    if cfg["rerank"]:
        a += ["--rerank"]
    return a


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", help="held-out question file, e.g. eval/test_v2.jsonl")
    parser.add_argument("--abstention", help="optional new out-of-corpus questions, e.g. eval/abstention_v2.jsonl")
    parser.add_argument("--tag", default="heldout_v2")
    parser.add_argument("--skip-git-check", action="store_true", help="only for a smoke test, never for the real run")
    args = parser.parse_args()
    files = [args.questions] + ([args.abstention] if args.abstention else [])

    # 1. validation
    run(["scripts/check_golden.py", args.questions])
    questions = load_golden(ROOT / args.questions)
    golden = load_golden(ROOT / "eval/golden.jsonl")
    clash = {q["id"] for q in questions} & {g["id"] for g in golden}
    same = {q["question"] for q in questions} & {g["question"] for g in golden}
    if clash or same:
        sys.exit(f"overlaps eval/golden.jsonl: ids {sorted(clash)}, questions {sorted(same)}")
    if not args.skip_git_check:
        for f in files:
            if not git("ls-files", f) or git("status", "--porcelain", f):
                sys.exit(f"{f} must be committed and unchanged before the held-out run")

    # 2. one run only
    existing = sorted(p.name for p in (ROOT / "results").glob(f"*{args.tag}*"))
    if existing:
        sys.exit(f"results for '{args.tag}' already exist: {existing}")

    # 3. rewrites, one LLM call per question
    rw_path = ROOT / f"results/rewrites_{args.tag}.jsonl"
    rewrites = {}
    for q in questions:
        rewrites[q["id"]] = cached_rewrite(q, rw_path, rewrites)["rewrite"]

    # 4. retrieval of every system
    texts = {p.stem: p.read_text(encoding="utf-8") for p in sorted((ROOT / "corpus/text").glob("*.txt"))}
    retrievers = {stem: HybridRetriever(texts, stem=stem) for stem in {s[0] for s in SYSTEMS.values()}}
    reranker = Reranker()
    rows = []
    for q in questions:
        row = {"id": q["id"], "type": q["type"]}
        for name, (stem, use_rw, rerank) in SYSTEMS.items():
            queries = [rewrites[q["id"]]] if use_rw is None else [q["question"]] + ([rewrites[q["id"]]] if use_rw else [])
            cands = retrievers[stem].search_multi(queries, k=N_CAND)
            if rerank:
                cands = reranker.rerank(q["question"], cands)
            for k in KS:
                row[f"{name} R@{k}"] = recall_at_k([c["text"] for c in cands], facts(q), k)
        rows.append(row)
    per_q = pd.DataFrame(rows)
    summary = pd.DataFrame([{"system": s, "n": len(per_q), **{f"R@{k}": per_q[f"{s} R@{k}"].mean() for k in KS}}
                            for s in SYSTEMS]).round(2)
    per_q.to_csv(ROOT / f"results/{args.tag}_retrieval_per_question.csv", index=False)
    summary.to_csv(ROOT / f"results/{args.tag}_retrieval_summary.csv", index=False)
    print(summary.to_string(index=False))
    del retrievers, reranker    # free GPU memory before the answer runs load their own models

    # 5. answers
    for name, cfg in (("v0", BASELINE), ("final", FINAL)):
        run(["scripts/run_answers.py", "--questions", args.questions, "--tag", f"{args.tag}_{name}",
             *answer_args(cfg, str(rw_path.relative_to(ROOT)))])
        if args.abstention:
            abs_rw = f"results/rewrites_{args.tag}_abstention.jsonl"
            run(["scripts/run_answers.py", "--questions", args.abstention, "--tag", f"{args.tag}_abstention_{name}",
                 *answer_args(cfg, abs_rw)])

    # 6. what was run
    config = {"date": datetime.now().isoformat(timespec="seconds"), "commit": git("rev-parse", "HEAD"),
              "questions": files, "final": FINAL, "baseline": BASELINE, "systems": list(SYSTEMS)}
    (ROOT / f"results/{args.tag}_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"done; grade results/answers_{args.tag}_*_grading.csv")


if __name__ == "__main__":
    main()
