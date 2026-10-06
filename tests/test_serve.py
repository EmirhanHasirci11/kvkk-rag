"""Run from the repo root: python -m pytest tests -v

The pipeline is replaced by a stub, so no model is loaded and no LLM call is made.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pytest
from fastapi.testclient import TestClient

import serve
from llm import BudgetExceeded


class StubPipeline:
    dense, startup_s = "stub", 0.0

    def __init__(self, dense=None):
        pass

    def ask(self, question):
        if question == "bütçe bitti":
            raise BudgetExceeded("LLM budget reached")
        return {"question": question, "answer": "Bu konuda verilen metinlerde bilgi yok.", "timings_ms": {}}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(serve, "Pipeline", StubPipeline)
    with TestClient(serve.app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["ready"] and body["dense"] == "stub"


def test_ask(client):
    r = client.post("/ask", json={"question": "Açık rıza nedir?"})
    assert r.status_code == 200 and r.json()["question"] == "Açık rıza nedir?"


def test_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_budget_exceeded_is_503(client):
    assert client.post("/ask", json={"question": "bütçe bitti"}).status_code == 503
