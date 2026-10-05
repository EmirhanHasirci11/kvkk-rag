"""One small LLM interface: generate(prompt, system) -> text, token counts and cost.

Backend: Gemini via the google-genai SDK, with the GEMINI_API_KEY from .env
(the prepaid, paid-tier AI Studio project; free-tier data is used for training).
Every call is appended to results/llm_usage.csv, and no call is made once the
logged total reaches BUDGET_USD.
"""
import csv
import sys
from datetime import datetime
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
USAGE_CSV = ROOT / "results/llm_usage.csv"
USAGE_FIELDS = ["date", "script", "model", "input_tokens", "output_tokens", "cost_usd"]

MODEL = "gemini-3.8-flash"
# USD per 1M tokens, paid tier, valid through 2026-12-31 (both double on 2027-01-01).
# Thinking tokens are billed as output.
PRICES = {"gemini-3.8-flash": {"input": 0.75, "output": 3.75}}
BUDGET_USD = 8.0


class BudgetExceeded(RuntimeError):
    pass


def spent_usd() -> float:
    if not USAGE_CSV.exists():
        return 0.0
    with USAGE_CSV.open(encoding="utf-8", newline="") as f:
        return sum(float(row["cost_usd"]) for row in csv.DictReader(f))


def _log_usage(row: dict):
    new = not USAGE_CSV.exists()
    with USAGE_CSV.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=USAGE_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        # read only from .env, so a key set elsewhere in the environment is never picked up
        key = dotenv_values(ROOT / ".env").get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is missing in .env")
        _client = genai.Client(api_key=key)
    return _client


def generate(prompt: str, system: str) -> dict:
    spent = spent_usd()
    if spent >= BUDGET_USD:
        raise BudgetExceeded(f"LLM budget reached: ${spent:.4f} of ${BUDGET_USD:.2f} spent, see {USAGE_CSV.name}")

    from google.genai import types
    resp = _get_client().models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system, temperature=0,
            # no tools are passed; this only silences the SDK's AFC warning
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)),
    )
    u = resp.usage_metadata
    input_tokens = u.prompt_token_count or 0
    output_tokens = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
    price = PRICES[MODEL]
    cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000

    _log_usage({"date": datetime.now().isoformat(timespec="seconds"), "script": Path(sys.argv[0]).name,
                "model": MODEL, "input_tokens": input_tokens, "output_tokens": output_tokens,
                "cost_usd": f"{cost:.6f}"})
    return {"text": resp.text or "", "input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost}
