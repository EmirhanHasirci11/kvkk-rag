"""Answer every question in a question file and write the answers plus a grading sheet.

Run from the repo root:
    python scripts/run_answers.py [--ids q004 q007] [--tag v0]
    python scripts/run_answers.py --questions eval/abstention.jsonl --tag abstention_v0
    python scripts/run_answers.py --rewrite --tag v1    (reuses results/rewrites_v1.jsonl)
    python scripts/run_answers.py --rewrite --rewrites results/rewrites_v1.jsonl --prompt v1 --tag v2
    python scripts/run_answers.py --rewrite --rewrites results/rewrites_v1.jsonl --prompt v1 --rerank --tag v3
    python scripts/run_answers.py --rewrite --rewrites results/rewrites_v1.jsonl --prompt v1 --rerank --stem snow --tag v4

Writes results/answers_<tag>.jsonl and results/answers_<tag>_grading.csv and refuses to
overwrite them. citation_check: "pass" when every gold fact (any alternative) is contained
in a chunk the answer cites, "fail" when not, "no_citation" when the answer cites nothing,
"n/a" for questions without evidence (eval/abstention.jsonl, graded against "expected").
A failed question is recorded with its error and the run continues.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from answer import PROMPTS, answer, load_retriever
from golden import facts, load_golden, norm
from llm import BUDGET_USD, BudgetExceeded, spent_usd

ROOT = Path(__file__).resolve().parents[1]


def citation_check(r: dict, fact_list: list[list[str]]) -> str:
    cited = {n for c in r["citations"] for n in c["chunk_ranks"]}
    if not r["citations"]:
        return "no_citation"
    texts = [norm(c["text"]) for c in r["retrieved"] if c["rank"] in cited]
    ok = all(any(norm(alt) in t for alt in alts for t in texts) for alts in fact_list)
    return "pass" if ok else "fail"


def evidence_text(fact_list: list[list[str]]) -> str:
    # facts separated by " | ", alternatives of one fact by " / "
    return " | ".join(" / ".join(alts) for alts in fact_list)


def cached_rewrite(g: dict, cache: Path, rewrites: dict) -> dict:
    """Rewrite from the cache file, or a new LLM call that is appended to it."""
    if g["id"] not in rewrites:
        from rewrite import rewrite
        out = rewrite(g["question"])
        rewrites[g["id"]] = {"id": g["id"], "question": g["question"], "rewrite": out["text"],
                             "input_tokens": out["input_tokens"], "output_tokens": out["output_tokens"],
                             "cost_usd": out["cost_usd"]}
        with cache.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rewrites[g["id"]], ensure_ascii=False) + "\n")
    rw = rewrites[g["id"]]
    assert rw["question"] == g["question"], f"{g['id']}: cached rewrite is for a different question"
    return rw


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval/golden.jsonl", help="question file, relative to the repo root")
    parser.add_argument("--ids", nargs="+", help="run only these question ids")
    parser.add_argument("--tag", default="v0", help="output file suffix")
    parser.add_argument("--rewrite", action="store_true",
                        help="also search with the LLM rewrite of each question, cached in results/rewrites_<tag>.jsonl")
    parser.add_argument("--rewrites", help="rewrite cache to use instead, e.g. results/rewrites_v1.jsonl")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="v0", help="system prompt version")
    parser.add_argument("--rerank", action="store_true", help="rerank the top-30 with bge-reranker-v2-m3")
    parser.add_argument("--stem", default="prefix5", help="BM25 stemmer, a key of textproc.STEMMERS")
    args = parser.parse_args()

    cache = ROOT / (args.rewrites or f"results/rewrites_{args.tag}.jsonl")
    rewrites = {}
    if args.rewrite and cache.exists():
        rewrites = {r["id"]: r for r in map(json.loads, cache.read_text(encoding="utf-8").splitlines())}

    out_jsonl = ROOT / f"results/answers_{args.tag}.jsonl"
    out_csv = ROOT / f"results/answers_{args.tag}_grading.csv"
    for p in (out_jsonl, out_csv):
        if p.exists():
            sys.exit(f"{p.relative_to(ROOT)} already exists; use a new --tag")

    questions = load_golden(ROOT / args.questions)
    if args.ids:
        questions = [g for g in questions if g["id"] in args.ids]
    retriever = load_retriever(args.stem)
    reranker = None
    if args.rerank:
        from reranker import Reranker
        reranker = Reranker()

    rows, errors = [], []
    for g in questions:
        try:
            rw = cached_rewrite(g, cache, rewrites) if args.rewrite else None
            r = answer(g["question"], retriever, rewritten=rw["rewrite"] if rw else None, prompt=args.prompt,
                       reranker=reranker)
            if rw:
                r["rewrite_cost_usd"] = rw["cost_usd"]
        except BudgetExceeded as e:
            # every later call would fail the same way
            errors += [(q["id"], str(e)) for q in questions[questions.index(g):]]
            print(f"{g['id']}: stopped, {e}")
            break
        except Exception as e:
            errors.append((g["id"], f"{type(e).__name__}: {e}"))
            rows.append({"id": g["id"], "question": g["question"], "error": f"{type(e).__name__}: {e}"})
            print(f"{g['id']}: ERROR {type(e).__name__}: {e}")
            continue
        check = citation_check(r, facts(g)) if "evidence" in g else "n/a"
        r = {"id": g["id"], **r, "citation_check": check}
        rows.append(r)
        print(f"{g['id']}: {r['citation_check']:<11} no_info={r['no_info']!s:<5} ${r['cost_usd']:.5f}")

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    gold = {g["id"]: g for g in questions}
    with_evidence = any("evidence" in g for g in questions)
    # utf-8-sig so Excel shows the Turkish characters correctly
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "answer", "gold_evidence" if with_evidence else "expected",
                    "citation_check", "grade", "note"])
        for r in rows:
            g = gold[r["id"]]
            w.writerow([r["id"], r["question"], r.get("answer", ""),
                        evidence_text(facts(g)) if "evidence" in g else g.get("expected", ""),
                        r.get("citation_check", "error"), "", ""])

    done = [r for r in rows if "error" not in r]
    checked = [r for r in done if r["citation_check"] != "n/a"]
    passed = sum(r["citation_check"] == "pass" for r in checked)
    print(f"\n{len(done)} answered, {len(errors)} errors")
    if checked:
        print(f"citation check pass: {passed}/{len(checked)} = {passed / len(checked):.2f}")
    if done:
        print(f"'bilgi yok' answers: {sum(r['no_info'] for r in done)}")
        print(f"answer cost: ${sum(r['cost_usd'] for r in done):.4f} (rewrites are logged in llm_usage.csv)")
    print(f"total spent: ${spent_usd():.4f} / ${BUDGET_USD:.2f}")
    if errors:
        print("errors:")
        for qid, msg in errors:
            print(f"  {qid}: {msg}")
    print(f"wrote {out_jsonl.relative_to(ROOT)} and {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
