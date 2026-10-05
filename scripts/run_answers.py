"""Answer every question in eval/golden.jsonl and write the answers plus a grading sheet.

Run from the repo root: python scripts/run_answers.py [--ids q004 q007] [--tag v0]

Writes results/answers_<tag>.jsonl and results/answers_<tag>_grading.csv and refuses to
overwrite them. citation_check: "pass" when every gold fact (any alternative) is contained
in a chunk the answer cites, "fail" when not, "no_citation" when the answer cites nothing.
A failed question is recorded with its error and the run continues.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from answer import answer, load_retriever
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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", help="run only these question ids")
    parser.add_argument("--tag", default="v0", help="output file suffix")
    args = parser.parse_args()

    out_jsonl = ROOT / f"results/answers_{args.tag}.jsonl"
    out_csv = ROOT / f"results/answers_{args.tag}_grading.csv"
    for p in (out_jsonl, out_csv):
        if p.exists():
            sys.exit(f"{p.relative_to(ROOT)} already exists; use a new --tag")

    questions = load_golden(ROOT / "eval/golden.jsonl")
    if args.ids:
        questions = [g for g in questions if g["id"] in args.ids]
    retriever = load_retriever()

    rows, errors = [], []
    for g in questions:
        fl = facts(g)
        try:
            r = answer(g["question"], retriever)
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
        r = {"id": g["id"], **r, "citation_check": citation_check(r, fl)}
        rows.append(r)
        print(f"{g['id']}: {r['citation_check']:<11} no_info={r['no_info']!s:<5} ${r['cost_usd']:.5f}")

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    gold = {g["id"]: g for g in questions}
    # utf-8-sig so Excel shows the Turkish characters correctly
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "answer", "gold_evidence", "citation_check", "grade", "note"])
        for r in rows:
            w.writerow([r["id"], r["question"], r.get("answer", ""), evidence_text(facts(gold[r["id"]])),
                        r.get("citation_check", "error"), "", ""])

    done = [r for r in rows if "error" not in r]
    passed = sum(r["citation_check"] == "pass" for r in done)
    print(f"\n{len(done)} answered, {len(errors)} errors")
    if done:
        print(f"citation check pass: {passed}/{len(done)} = {passed / len(done):.2f}")
        print(f"'bilgi yok' answers: {sum(r['no_info'] for r in done)}")
        print(f"run cost: ${sum(r['cost_usd'] for r in done):.4f}")
    print(f"total spent: ${spent_usd():.4f} / ${BUDGET_USD:.2f}")
    if errors:
        print("errors:")
        for qid, msg in errors:
            print(f"  {qid}: {msg}")
    print(f"wrote {out_jsonl.relative_to(ROOT)} and {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
