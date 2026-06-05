"""Day 20 tests: error recovery, structured logging, token budget, trace JSON."""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agent.tracing import configure_logging

configure_logging()  # structlog JSON from the start


# ── TEST 1: broken tool — exception returned to LLM, agent recovers ──────────
def test_broken_tool_recovery():
    print("\n" + "=" * 70)
    print("TEST 1: Broken tool — exception recovery (invalid ticker)")
    print("=" * 70)

    from pydantic import BaseModel
    from agent.tools.base import Tool
    from agent.tools.finance import get_stock_data
    from agent.registry import ToolRegistry
    from agent.loop import run_agent
    from providers.factory import get_provider

    # A tool that always raises — simulates a network outage or bad API
    class _BrokenInput(BaseModel):
        ticker: str

    def _broken_handler(inputs: _BrokenInput) -> dict:
        raise RuntimeError(f"Simulated API failure for ticker '{inputs.ticker}'")

    broken_tool = Tool(
        name="get_stock_data",  # replace the real one so LLM calls it
        description="Fetches stock price data for a given ticker.",
        input_schema=_BrokenInput,
        handler=_broken_handler,
    )

    provider = get_provider("openai", "gpt-4o-mini")
    registry = ToolRegistry()
    registry.register(broken_tool)

    result = run_agent(
        query="What is the current price of AAPL?",
        provider=provider,
        registry=registry,
        max_iterations=5,
        write_trace_file=True,
    )

    # Agent must produce an answer — not crash
    assert result.answer and len(result.answer) > 10, f"Expected non-empty answer, got: {result.answer!r}"

    # At least one tool call must have status="error"
    all_calls = [tc for log in result.logs for tc in log.tool_calls]
    error_calls = [tc for tc in all_calls if tc.status == "error"]
    assert len(error_calls) >= 1, "Expected at least one error tool call"
    assert "RuntimeError" in error_calls[0].error, f"Expected RuntimeError in error, got: {error_calls[0].error}"

    # Trace file must exist
    assert result.trace_path and Path(result.trace_path).exists(), "Trace file not written"

    print(f"  Tool errors recorded: {len(error_calls)}")
    print(f"  Error message: {error_calls[0].error}")
    print(f"  LLM answer: {result.answer[:300]}")
    print(f"  Trace file: {result.trace_path}")
    print("  PASSED")
    return result


# ── TEST 2: trace JSON schema ─────────────────────────────────────────────────
def test_trace_json_structure(trace_path: str):
    print("\n" + "=" * 70)
    print("TEST 2: Trace JSON — correct schema")
    print("=" * 70)

    with open(trace_path) as f:
        trace = json.load(f)

    required_keys = {"run_id", "query", "total_tokens", "iterations", "total_duration_ms", "iterations_detail", "answer_snippet"}
    missing = required_keys - trace.keys()
    assert not missing, f"Missing trace keys: {missing}"

    # Each iteration must have tool_calls list
    for it in trace["iterations_detail"]:
        assert "iteration" in it
        assert "tokens_used" in it
        assert "tool_calls" in it
        for tc in it["tool_calls"]:
            assert "name" in tc
            assert "duration_ms" in tc
            assert "status" in tc

    print(f"  run_id: {trace['run_id']}")
    print(f"  total_tokens: {trace['total_tokens']}")
    print(f"  total_duration_ms: {trace['total_duration_ms']}")
    print(f"  iterations: {trace['iterations']}")
    print("  PASSED")


# ── TEST 3: duration_ms and status logged per tool call ───────────────────────
def test_tool_call_log_fields():
    print("\n" + "=" * 70)
    print("TEST 3: ToolCallLog has duration_ms and status fields")
    print("=" * 70)

    from agent.tools.finance import get_stock_data
    from agent.registry import ToolRegistry
    from agent.loop import run_agent
    from providers.factory import get_provider

    provider = get_provider("openai", "gpt-4o-mini")
    registry = ToolRegistry()
    registry.register(get_stock_data)

    result = run_agent(
        query="What is the current price of NVDA?",
        provider=provider,
        registry=registry,
        max_iterations=5,
        write_trace_file=True,
    )

    all_calls = [tc for log in result.logs for tc in log.tool_calls]
    assert len(all_calls) >= 1, "Expected at least one tool call"

    for tc in all_calls:
        assert tc.duration_ms >= 0, f"duration_ms must be non-negative, got {tc.duration_ms}"
        assert tc.status in ("success", "error"), f"status must be success/error, got {tc.status}"
        assert isinstance(tc.args, dict), "args must be a dict"

    success_calls = [tc for tc in all_calls if tc.status == "success"]
    assert len(success_calls) >= 1, "Expected at least one successful tool call"

    for tc in success_calls:
        assert tc.error is None, "Successful call should have no error"

    print(f"  Tool calls: {[(tc.name, tc.status, tc.duration_ms) for tc in all_calls]}")
    print("  PASSED")


# ── TEST 4: token budget guard ────────────────────────────────────────────────
def test_token_budget_guard():
    print("\n" + "=" * 70)
    print("TEST 4: Token budget guard — stops when budget exceeded")
    print("=" * 70)

    from agent.tools.finance import get_stock_data
    from agent.registry import ToolRegistry
    from agent.loop import run_agent
    from providers.factory import get_provider

    provider = get_provider("openai", "gpt-4o-mini")
    registry = ToolRegistry()
    registry.register(get_stock_data)

    # Tiny budget (100 tokens) to force early stop
    result = run_agent(
        query="What is the price of NVDA?",
        provider=provider,
        registry=registry,
        max_iterations=10,
        token_budget=100,
        write_trace_file=True,
    )

    # Should stop after first iteration due to budget
    assert result.total_tokens > 100, "total_tokens should exceed the budget"
    assert result.iterations <= 2, f"Should stop early, got {result.iterations} iterations"

    print(f"  total_tokens={result.total_tokens} > budget=100 → stopped at iteration {result.iterations}")
    print("  PASSED")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("DAY 20 TESTS — Error Recovery + Structured Logging")
    print("=" * 70)

    result1 = test_broken_tool_recovery()
    test_trace_json_structure(result1.trace_path)
    test_tool_call_log_fields()
    test_token_budget_guard()

    print("\n" + "=" * 70)
    print("ALL DAY 20 TESTS PASSED")
    print("agent/tracing.py: structlog JSON + write_trace()")
    print("agent/loop.py:    ToolCallLog(duration_ms, status) + error recovery + 100k budget")
    print("=" * 70)
