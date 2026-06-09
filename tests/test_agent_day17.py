"""Day 17 — test: multi-turn tool-use agent loop.

Query: "Latest news and price for NVDA"
Expected: 2-3 tool calls (get_stock_data + web_search), then a coherent final answer.
"""
import logging
import os
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
from agent.loop import run_agent

# ── registry ──────────────────────────────────────────────────────────────
registry = ToolRegistry()
registry.register(get_current_time)
registry.register(get_stock_data)
registry.register(web_search)

# ── provider ──────────────────────────────────────────────────────────────
provider = OpenAIProvider(model="gpt-4o-mini")

# ── run ───────────────────────────────────────────────────────────────────
query = "Latest news and price for NVDA"
print(f"\n{'='*60}")
print(f"Query: {query}")
print("=" * 60)

result = run_agent(query, provider, registry, max_iterations=10)

# ── iteration log ─────────────────────────────────────────────────────────
print(f"\n--- Iteration log ({result.iterations} iterations, {result.total_tokens} tokens) ---")
for log in result.logs:
    if log.tool_calls:
        for tc in log.tool_calls:
            print(f"  [iter {log.iteration}] tool={tc.name}  status={tc.status}")
    else:
        print(f"  [iter {log.iteration}] (no tool calls — final answer)")

# ── final answer ──────────────────────────────────────────────────────────
print(f"\n--- Final Answer ---\n{result.answer}")

assert result.answer and result.answer not in ("[max iterations reached]", "[token budget exceeded]"), \
    "Agent did not produce a final answer!"
assert result.iterations >= 2, "Expected at least 2 iterations (tool call + answer)"

print("\nAll assertions passed.")
