"""Day 29 — LangSmith Tracing Demo

Hafta 4 LangGraph agent'ını 5 kez çalıştırır; her çalışma LangSmith'te
ayrı bir trace olarak görünür. Custom metadata (ticker, provider, model)
her trace'e eklenir.

Kullanım:
    python scripts/run_day29.py

Ön koşul: .env dosyasında şunlar dolu olmalı:
    LANGSMITH_API_KEY=lsv2_...
    LANGSMITH_PROJECT=finance-agent
    LANGCHAIN_TRACING_V2=true
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Proje root'unu sys.path'e ekle
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

# ── Tracing kontrol ────────────────────────────────────────────────────────────

def _check_tracing() -> bool:
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    project = os.getenv("LANGSMITH_PROJECT", "finance-agent")

    if not tracing or not api_key:
        print("[UYARI] LangSmith tracing devre dışı veya API key eksik.")
        print("  .env içinde şunları ayarlayın:")
        print("    LANGSMITH_API_KEY=lsv2_...")
        print("    LANGCHAIN_TRACING_V2=true")
        print("  Devam ediliyor (trace gönderilmeyecek)...\n")
        return False

    print(f"[OK] LangSmith tracing aktif — project: {project}")
    return True


# ── Hafta 3 agent loop (brief.py tabanlı) ─────────────────────────────────────

def run_week3_traces() -> None:
    """agent/loop.py tabanlı 3 trace — NVDA, TSLA, MSFT."""
    from agent.loop import run_agent
    from agent.registry import ToolRegistry
    from agent.tools.finance import get_stock_data
    from agent.tools.search import web_search
    from agent.tools.sentiment import analyze_news_sentiment
    from agent.tracing import configure_logging
    from providers.factory import get_provider

    configure_logging()

    provider = get_provider("openai", "gpt-4o-mini")
    registry = ToolRegistry()
    registry.register(get_stock_data)
    registry.register(web_search)
    registry.register(analyze_news_sentiment)

    cases = [
        ("NVDA", "What is the current stock price and recent news sentiment for NVDA?"),
        ("TSLA", "Is TSLA bullish or bearish based on recent news?"),
        ("MSFT", "Summarize recent analyst outlook for MSFT."),
    ]

    for ticker, query in cases:
        print(f"[agent] Running Week-3 agent for {ticker}...")
        t0 = time.perf_counter()
        result = run_agent(
            query=query,
            provider=provider,
            registry=registry,
            max_iterations=5,
            metadata={
                "ticker": ticker,
                "provider": "openai",
                "model": "gpt-4o-mini",
                "week": 3,
                "day": 29,
            },
        )
        elapsed = round((time.perf_counter() - t0) * 1000)
        print(f"  Done: {result.iterations} iter, {result.total_tokens} tokens, {elapsed}ms")
        print(f"  Answer: {result.answer[:120]}...\n")


# ── Hafta 4 LangGraph graph ────────────────────────────────────────────────────

def run_week4_traces() -> None:
    """graph/finance_graph.py tabanlı 2 trace — NVDA, AAPL."""
    from graph.finance_graph import build_finance_graph
    from graph.checkpointer import make_checkpointer

    cases = [
        ("NVDA", "openai", "gpt-4o-mini"),
        ("AAPL", "openai", "gpt-4o-mini"),
    ]

    with make_checkpointer("day29_traces.db") as cp:
        graph = build_finance_graph(checkpointer=cp)

        for ticker, provider_name, model_name in cases:
            thread_id = f"day29-{ticker}-{int(time.time())}"
            config = {
                "configurable": {"thread_id": thread_id},
                # LangGraph bu metadata'yı her node'un trace'ine ekler
                "metadata": {
                    "ticker": ticker,
                    "provider": provider_name,
                    "model": model_name,
                    "week": 4,
                    "day": 29,
                    "thread_id": thread_id,
                },
            }

            print(f"[graph] Running Week-4 graph for {ticker} (thread={thread_id})...")
            t0 = time.perf_counter()
            state = graph.invoke({"ticker": ticker, "messages": []}, config)
            elapsed = round((time.perf_counter() - t0) * 1000)

            draft = state.get("draft", "")[:150]
            sentiment = state.get("sentiment_summary", "N/A")
            print(f"  Done: {elapsed}ms | sentiment={sentiment}")
            print(f"  Draft: {draft}...\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Day 29 — LangSmith Tracing Demo")
    print("=" * 60)
    _check_tracing()

    print("\n--- Hafta 3: agent/loop.py traces (3 runs) ---")
    run_week3_traces()

    print("\n--- Hafta 4: LangGraph traces (2 runs) ---")
    run_week4_traces()

    project = os.getenv("LANGSMITH_PROJECT", "finance-agent")
    print("\n" + "=" * 60)
    print(f"Tamamlandı. LangSmith dashboard'da kontrol edin:")
    print(f"  https://smith.langchain.com  →  project: {project}")
    print("=" * 60)


if __name__ == "__main__":
    main()
