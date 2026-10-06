"""The final pipeline (v4, frozen in scripts/run_heldout.py) as one object with per-stage timings (ROADMAP step 7).

question -> LLM rewrite -> fuse (question + rewrite) top-30, bm25-snow -> bge rerank -> top-5 -> prompt v1 -> answer
"""
import time
from pathlib import Path

from answer import PROMPTS, build_prompt, is_no_info, parse_citations
from llm import MODEL, generate
from reranker import Reranker
from retriever import HybridRetriever
from rewrite import rewrite

ROOT = Path(__file__).resolve().parents[1]
STEM, PROMPT, N_CANDIDATES, TOP_K = "snow", "v1", 30, 5


class Pipeline:
    def __init__(self, dense="numpy", dsn=None):
        t0 = time.perf_counter()
        texts = {p.stem: p.read_text(encoding="utf-8") for p in sorted((ROOT / "corpus/text").glob("*.txt"))}
        self.retriever = HybridRetriever(texts, stem=STEM, dense=dense, dsn=dsn)
        self.reranker = Reranker()
        self.dense = dense
        self.startup_s = time.perf_counter() - t0

    def retrieve(self, question: str, rewritten: str, ms: dict | None = None) -> list[dict]:
        """Top-5 chunks for a question and its rewrite; no LLM call."""
        ms = {} if ms is None else ms
        t = time.perf_counter()
        candidates = self.retriever.search_multi([question, rewritten], k=N_CANDIDATES)
        ms["retrieval"] = round((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        chunks = self.reranker.rerank(question, candidates, k=TOP_K)
        ms["rerank"] = round((time.perf_counter() - t) * 1000)
        return chunks

    def ask(self, question: str) -> dict:
        ms = {}

        def lap(name, t0):
            ms[name] = round((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        rw = rewrite(question)
        lap("rewrite", t0)
        chunks = self.retrieve(question, rw["text"], ms)
        t = time.perf_counter()
        out = generate(build_prompt(question, chunks), PROMPTS[PROMPT])
        lap("generation", t)
        lap("total", t0)
        return {
            "question": question,
            "answer": out["text"].strip(),
            "no_info": is_no_info(out["text"]),
            "citations": parse_citations(out["text"], chunks),
            "sources": [{"rank": c["rank"], "doc": c["doc"], "section": c["section"], "sections": c["sections"]}
                        for c in chunks],
            "rewrite": rw["text"],
            "model": MODEL,
            "input_tokens": rw["input_tokens"] + out["input_tokens"],
            "output_tokens": rw["output_tokens"] + out["output_tokens"],
            "cost_usd": rw["cost_usd"] + out["cost_usd"],
            "timings_ms": ms,
        }
