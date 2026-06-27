"""Day 39: Per-call LLM cost tracking in Redis.

Each LLM call's cost (tokens × price/1K) is accumulated in a Redis hash
keyed by date. A daily budget guard raises BudgetExceededError when
today's spend crosses the threshold.

Redis keys:
    cost:{YYYY-MM-DD} → HASH {total_usd, calls, input_tokens, output_tokens}
TTL: 31 days

Usage:
    from middleware.cost import track_cost, get_daily_stats, check_budget

    track_cost("gpt-4o-mini", input_tokens=1200, output_tokens=400)
    check_budget()           # raises BudgetExceededError if over $5
    stats = get_daily_stats("2026-06-28")
"""
from __future__ import annotations

import os
from datetime import date

import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DAILY_BUDGET_USD = float(os.getenv("DAILY_BUDGET_USD", "5.0"))
_TTL = 31 * 24 * 3600  # 31 days

# Prices in USD per 1 000 tokens (input / output)
_PRICE_PER_1K: dict[str, dict[str, float]] = {
    "gpt-4o-mini":                {"input": 0.00015,  "output": 0.0006},
    "gpt-4o":                     {"input": 0.0025,   "output": 0.01},
    "gpt-4.1-mini":               {"input": 0.0004,   "output": 0.0016},
    "gpt-4.1-nano":               {"input": 0.0001,   "output": 0.0004},
    "llama-3.3-70b-versatile":    {"input": 0.00059,  "output": 0.00079},
    "llama-3.1-8b-instant":       {"input": 0.00005,  "output": 0.00008},
    "claude-3-haiku-20240307":    {"input": 0.00025,  "output": 0.00125},
    "claude-3-5-sonnet-20241022": {"input": 0.003,    "output": 0.015},
    "claude-sonnet-4-6":          {"input": 0.003,    "output": 0.015},
}
_DEFAULT_PRICE: dict[str, float] = {"input": 0.001, "output": 0.002}


class BudgetExceededError(Exception):
    """Raised when today's LLM spend exceeds DAILY_BUDGET_USD."""


def _today() -> str:
    return date.today().isoformat()


def _client() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)


def _key(day: str) -> str:
    return f"cost:{day}"


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return USD cost for one LLM call."""
    prices = _PRICE_PER_1K.get(model, _DEFAULT_PRICE)
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1000.0


def track_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    day: str | None = None,
) -> float:
    """Accumulate cost for one LLM call in Redis. Returns cost in USD.

    If only total_tokens is known (no per-direction breakdown), the function
    estimates a 70/30 input/output split — typical for tool-use agents.
    """
    if not input_tokens and not output_tokens and total_tokens:
        input_tokens = int(total_tokens * 0.7)
        output_tokens = total_tokens - input_tokens

    day = day or _today()
    cost_usd = compute_cost(model, input_tokens, output_tokens)
    key = _key(day)

    try:
        r = _client()
        pipe = r.pipeline()
        pipe.hincrbyfloat(key, "total_usd", cost_usd)
        pipe.hincrby(key, "calls", 1)
        pipe.hincrby(key, "input_tokens", input_tokens)
        pipe.hincrby(key, "output_tokens", output_tokens)
        pipe.expire(key, _TTL)
        pipe.execute()
    except Exception:
        pass  # Redis down → don't crash the request

    return cost_usd


def get_daily_stats(day: str | None = None) -> dict:
    """Return accumulated cost stats for a given day (default: today)."""
    day = day or _today()
    try:
        raw = _client().hgetall(_key(day))
    except Exception:
        return {"day": day, "total_usd": 0.0, "calls": 0,
                "input_tokens": 0, "output_tokens": 0, "error": "redis_unavailable"}

    if not raw:
        return {"day": day, "total_usd": 0.0, "calls": 0,
                "input_tokens": 0, "output_tokens": 0}

    return {
        "day": day,
        "total_usd": round(float(raw.get("total_usd", 0.0)), 6),
        "calls": int(raw.get("calls", 0)),
        "input_tokens": int(raw.get("input_tokens", 0)),
        "output_tokens": int(raw.get("output_tokens", 0)),
        "budget_usd": DAILY_BUDGET_USD,
        "budget_remaining_usd": round(
            max(0.0, DAILY_BUDGET_USD - float(raw.get("total_usd", 0.0))), 6
        ),
    }


def check_budget(day: str | None = None, limit_usd: float = DAILY_BUDGET_USD) -> None:
    """Raise BudgetExceededError if today's spend is at or above limit_usd."""
    stats = get_daily_stats(day)
    if stats["total_usd"] >= limit_usd:
        raise BudgetExceededError(
            f"Daily budget of ${limit_usd:.2f} exceeded "
            f"(spent ${stats['total_usd']:.4f} today — {stats['calls']} calls)"
        )
