"""Day 16 — test: yfinance + tavily tools registered and dispatched."""
import os
from dotenv import load_dotenv

load_dotenv()

from agent.registry import ToolRegistry
from agent.tools.builtins import get_current_time
from agent.tools.finance import get_stock_data
from agent.tools.search import web_search

registry = ToolRegistry()
registry.register(get_current_time)
registry.register(get_stock_data)
registry.register(web_search)

print("=== Registered tools ===")
for t in registry.all_tools():
    print(f"  {t.name}")

print("\n=== Test: get_stock_data → NVDA ===")
result = registry.dispatch("get_stock_data", {"ticker": "NVDA", "period": "1mo"})
for k, v in result.items():
    print(f"  {k}: {v}")

# Only test web_search if the key is set
if os.environ.get("TAVILY_API_KEY"):
    print("\n=== Test: web_search → NVDA latest news ===")
    results = registry.dispatch("web_search", {"query": "NVDA stock news today", "max_results": 3})
    for r in results:
        print(f"  [{r['title']}] {r['url']}")
else:
    print("\n  (Skipping web_search — TAVILY_API_KEY not set)")

print("\nAll tool dispatch tests passed.")
