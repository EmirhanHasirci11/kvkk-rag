"""Latency and cost of the API, per stage (ROADMAP step 7).

Run from the repo root: python scripts/bench_serving.py [--n 10] [--dense numpy|pgvector] [--tag v1]

Sends the first n golden questions through POST /ask in-process (FastAPI TestClient), one at a time.
A retrieval-only warm-up runs first (no LLM call) and is not counted. Each request makes two LLM
calls, so n = 10 costs about $0.08.
Writes results/serving_<tag>_<dense>_requests.csv and results/serving_<tag>_<dense>_summary.csv.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--dense", default="numpy", choices=["numpy", "pgvector"])
parser.add_argument("--tag", default="v1")
args = parser.parse_args()
os.environ["KVKK_DENSE"] = args.dense

from fastapi.testclient import TestClient    # noqa: E402  (KVKK_DENSE must be set first)

import serve    # noqa: E402
from golden import load_golden    # noqa: E402

out_req = ROOT / f"results/serving_{args.tag}_{args.dense}_requests.csv"
out_sum = ROOT / f"results/serving_{args.tag}_{args.dense}_summary.csv"
if out_req.exists() or out_sum.exists():
    sys.exit(f"{out_req.name} or {out_sum.name} already exists; use a new --tag")

questions = load_golden(ROOT / "eval/golden.jsonl")[:args.n]
t0 = time.perf_counter()
with TestClient(serve.app) as client:
    startup_s = time.perf_counter() - t0
    serve.state["pipeline"].retrieve("Kişisel veri nedir?", "Kişisel veri tanımı")    # warm-up, not counted
    rows = []
    for g in questions:
        t = time.perf_counter()
        r = client.post("/ask", json={"question": g["question"]})
        http_ms = round((time.perf_counter() - t) * 1000)
        r.raise_for_status()
        d = r.json()
        rows.append({"id": g["id"], "http_ms": http_ms, **{f"{k}_ms": v for k, v in d["timings_ms"].items()},
                     "input_tokens": d["input_tokens"], "output_tokens": d["output_tokens"], "cost_usd": d["cost_usd"]})
        print(f"{g['id']}: {http_ms} ms, ${d['cost_usd']:.5f}")

req = pd.DataFrame(rows)
stages = ["rewrite_ms", "retrieval_ms", "rerank_ms", "generation_ms", "total_ms", "http_ms"]
summary = pd.DataFrame([{"stage": s, "p50": req[s].median(), "p95": req[s].quantile(0.95), "mean": req[s].mean()}
                        for s in stages]).round(0)
summary.loc[len(summary)] = {"stage": "cost_usd_per_question", "p50": round(req["cost_usd"].median(), 5),
                             "p95": round(req["cost_usd"].quantile(0.95), 5), "mean": round(req["cost_usd"].mean(), 5)}
summary.loc[len(summary)] = {"stage": "startup_s", "p50": round(startup_s, 1), "p95": None, "mean": None}
req.to_csv(out_req, index=False)
summary.to_csv(out_sum, index=False)
print(f"\n{args.dense}, n = {len(req)}")
print(summary.to_string(index=False))
