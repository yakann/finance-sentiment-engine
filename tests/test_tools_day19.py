"""Day 19 tests: 4-tool agent with 10-K RAG integration."""

import os
import sys
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)

# ── 1. Unit test: query_10k tool in isolation ─────────────────────────────────
def test_query_10k_direct():
    print("\n" + "=" * 70)
    print("TEST 1: query_10k tool — direct dispatch (NVDA autonomous vehicles)")
    print("=" * 70)

    from agent.tools.rag import _query_10k, _Query10KInput

    result = _query_10k(_Query10KInput(
        ticker="NVDA",
        question="What is NVDA's strategy in autonomous vehicles according to 10-K?",
    ))

    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert "answer" in result and result["answer"], "Missing answer"
    assert "sources" in result and len(result["sources"]) > 0, "No sources returned"

    print(f"\n  Answer ({len(result['answer'])} chars):\n  {result['answer'][:500]}...")
    print(f"\n  Sources ({len(result['sources'])} chunks):")
    for s in result["sources"][:3]:
        print(f"    [{s['section']}] {s['snippet'][:100]}...")

    print("\n  ✅ query_10k direct dispatch — PASSED")
    return result


# ── 2. Unsupported ticker graceful error ──────────────────────────────────────
def test_unsupported_ticker():
    print("\n" + "=" * 70)
    print("TEST 2: query_10k — unsupported ticker returns error dict")
    print("=" * 70)

    from agent.tools.rag import _query_10k, _Query10KInput

    result = _query_10k(_Query10KInput(ticker="TSLA", question="What is their EV strategy?"))

    assert "error" in result, "Expected error key for unsupported ticker"
    print(f"  Error message: {result['error']}")
    print("  ✅ Unsupported ticker graceful error — PASSED")


# ── 3. 4-tool agent loop: full integration ────────────────────────────────────
def test_four_tool_agent():
    print("\n" + "=" * 70)
    print("TEST 3: 4-tool agent — search + price + sentiment + 10-K")
    print("=" * 70)

    from providers.factory import get_provider
    from agent.registry import ToolRegistry
    from agent.loop import run_agent
    from agent.tools.finance import get_stock_data
    from agent.tools.search import web_search
    from agent.tools.sentiment import analyze_news_sentiment
    from agent.tools.rag import query_10k

    provider = get_provider("openai", "gpt-4o-mini")

    registry = ToolRegistry()
    registry.register(web_search)
    registry.register(get_stock_data)
    registry.register(analyze_news_sentiment)
    registry.register(query_10k)

    query = "What is NVDA's strategy in autonomous vehicles according to 10-K?"

    print(f"\n  Query: {query!r}")
    print(f"  Tools: {[t.name for t in registry.all_tools()]}")
    print()

    result = run_agent(query, provider=provider, registry=registry, max_iterations=5)

    assert result.answer and len(result.answer) > 50, "Answer too short"
    assert result.iterations >= 1, "Expected at least 1 iteration"

    # Verify that query_10k was actually called
    all_tool_calls = [
        call["name"]
        for log in result.logs
        for call in log.tool_calls
    ]
    assert "query_10k" in all_tool_calls, (
        f"Expected query_10k to be called, got: {all_tool_calls}"
    )

    print(f"  Iterations: {result.iterations}")
    print(f"  Total tokens: {result.total_tokens}")
    print(f"  Tools called: {all_tool_calls}")
    print(f"\n  Final answer:\n  {result.answer[:600]}")
    print("\n  ✅ 4-tool agent integration — PASSED")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("DAY 19 TESTS — RAG as Tool (10-K integration)")
    print("=" * 70)

    test_query_10k_direct()
    test_unsupported_ticker()
    test_four_tool_agent()

    print("\n" + "=" * 70)
    print("ALL DAY 19 TESTS PASSED ✅")
    print("4-tool agent: web_search + get_stock_data + analyze_news_sentiment + query_10k")
    print("=" * 70)
