"""
Optional local-LLM categorizer, used as a comparison arm.

This talks to Ollama on localhost, so no data leaves the machine and there is
no API cost - the property that made local models worth evaluating for a
finance app in the first place.

It is deliberately *not* the primary categorizer. Published results on real
bank data show a zero-shot LLM trailing a trained classifier by a wide margin
on abbreviated merchant strings, and per-transaction latency is orders of
magnitude higher. The role it earns here is a fallback for the transactions
the trained model flags as uncertain.

If Ollama is not installed or not running, every function degrades to a clear
"unavailable" result instead of crashing, so the evaluation script still runs.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .categories import SOURCE_CATEGORIES, UNCERTAIN_LABEL, to_budget_category

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"
REQUEST_TIMEOUT = 60

PROMPT_TEMPLATE = """You are a transaction categorizer for a budgeting app.

Assign the transaction below to exactly one category from this list:
{categories}

Rules:
- "[debit]" means money left the account; "[credit]" means money came in.
- A peer-to-peer payment (Venmo, Zelle, Cash App) is Income when it is a
  credit and Transfer when it is a debit.
- Recurring software or media billing is Subscriptions, not Shopping.
- If the merchant is unrecognisable, choose the closest category anyway.

Transaction: {description}
Amount: ${amount}

Respond with ONLY a JSON object, no markdown fences and no explanation:
{{"category": "<one category from the list>", "confidence": <0.0-1.0>}}
"""


@dataclass
class LLMResult:
    source_category: str
    budget_category: str
    confidence: float
    latency_ms: float
    raw_response: str
    ok: bool


def is_available(model: str = DEFAULT_MODEL) -> bool:
    """True when an Ollama server is reachable on localhost."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = {m.get("name", "") for m in payload.get("models", [])}
        return any(name.startswith(model.split(":")[0]) for name in names)
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return False


def _post(prompt: str, model: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 80},
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("response", "")


def classify(description: str, amount: float = 0.0, model: str = DEFAULT_MODEL) -> LLMResult:
    """Categorize one transaction with a local model."""
    prompt = PROMPT_TEMPLATE.format(
        categories="\n".join(f"- {c}" for c in SOURCE_CATEGORIES),
        description=description,
        amount=f"{amount:.2f}",
    )

    start = time.perf_counter()
    try:
        raw = _post(prompt, model)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return LLMResult(UNCERTAIN_LABEL, UNCERTAIN_LABEL, 0.0, 0.0, f"unavailable: {exc}", False)
    latency_ms = (time.perf_counter() - start) * 1000

    category, confidence = _parse(raw)
    valid = category in SOURCE_CATEGORIES

    return LLMResult(
        source_category=category if valid else UNCERTAIN_LABEL,
        budget_category=to_budget_category(category) if valid else UNCERTAIN_LABEL,
        confidence=confidence,
        latency_ms=round(latency_ms, 1),
        raw_response=raw.strip()[:200],
        ok=valid,
    )


def _parse(raw: str) -> tuple[str, float]:
    """Pull a category out of the model's reply, tolerating stray formatting."""
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        payload = json.loads(cleaned)
        category = str(payload.get("category", "")).strip()
        confidence = float(payload.get("confidence", 0.0))
        return category, max(0.0, min(1.0, confidence))
    except (json.JSONDecodeError, TypeError, ValueError):
        # Fall back to a substring match: small models sometimes answer in prose.
        upper = cleaned.upper()
        for candidate in SOURCE_CATEGORIES:
            if candidate.upper() in upper:
                return candidate, 0.5
        return "", 0.0
