"""Day 18 — test: composite agent (search + price + sentiment).

Query: "Should I be bullish on TSLA based on recent news?"
Expected: agent calls analyze_news_sentiment (+ optionally get_stock_data),
          then delivers a bullish/bearish recommendation grounded in real headlines.
"""
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

from providers.openai_provider import OpenAIProvider
from agent.registry import ToolRegistry
from agent.tools.builtins import get_current_time
from agent.tools.finance import get_stock_data
from agent.tools.search import web_search
from agent.tools.sentiment import analyze_news_sentiment
from agent.loop import run_agent

# ── registry: composite agent has all four tools ───────────────────────────
registry = ToolRegistry()
registry.register(get_current_time)
registry.register(get_stock_data)
registry.register(web_search)
registry.register(analyze_news_sentiment)

# ── provider ───────────────────────────────────────────────────────────────
provider = OpenAIProvider(model="gpt-4o-mini")

# ── run ────────────────────────────────────────────────────────────────────
query = "Should I be bullish on TSLA based on recent news?"
print(f"\n{'='*60}")
print(f"Query: {query}")
print("=" * 60)

result = run_agent(query, provider, registry, max_iterations=10)

# ── iteration log ──────────────────────────────────────────────────────────
print(f"\n--- Iteration log ({result.iterations} iterations, {result.total_tokens} tokens) ---")
for log in result.logs:
    if log.tool_calls:
        for tc in log.tool_calls:
            print(f"  [iter {log.iteration}] tool={tc.name}  status={tc.status}")
    else:
        print(f"  [iter {log.iteration}] (no tool calls — final answer)")

# ── final answer ───────────────────────────────────────────────────────────
print(f"\n--- Final Answer ---\n{result.answer}")

assert result.answer and result.answer not in (
    "[max iterations reached]", "[token budget exceeded]"
), "Agent did not produce a final answer!"
assert result.iterations >= 2, "Expected at least 2 iterations (tool call + answer)"

# Sentiment tool should have been used
used_tools = {tc["name"] for log in result.logs for tc in log.tool_calls}
assert "analyze_news_sentiment" in used_tools, (
    f"Expected analyze_news_sentiment to be called, got: {used_tools}"
)

print("\nAll assertions passed.")
