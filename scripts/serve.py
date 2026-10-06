"""HTTP API for the final pipeline (ROADMAP step 7).

Run from the repo root:
    uvicorn serve:app --app-dir scripts --port 8000
    KVKK_DENSE=pgvector uvicorn serve:app --app-dir scripts --port 8000   (needs `docker compose up -d`)

    curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "Açık rıza nedir?"}'

Every /ask makes two LLM calls (rewrite and answer); both count against the budget in scripts/llm.py.
"""
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llm import BUDGET_USD, BudgetExceeded, spent_usd
from pipeline import Pipeline

state = {}
lock = threading.Lock()    # one GPU, one request at a time


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["pipeline"] = Pipeline(dense=os.environ.get("KVKK_DENSE", "numpy"))
    yield
    state.clear()


app = FastAPI(title="kvkk-rag", lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/health")
def health():
    p = state.get("pipeline")
    return {"ready": p is not None, "dense": p.dense if p else None,
            "startup_s": round(p.startup_s, 1) if p else None,
            "spent_usd": round(spent_usd(), 4), "budget_usd": BUDGET_USD}


@app.post("/ask")
def ask(q: Question):
    with lock:
        try:
            return state["pipeline"].ask(q.question)
        except BudgetExceeded as e:
            raise HTTPException(status_code=503, detail=str(e))
