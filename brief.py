"""Day 21: `brief <TICKER>` CLI — generates a 2-page structured investment brief.

Usage:
    python brief.py NVDA
    python brief.py TSLA --print
    brief MSFT            # after `uv pip install -e .`

The agent calls all 4 tools (stock data, news sentiment, web search, 10-K RAG),
then formats the results as a structured Markdown document saved to briefs/<TICKER>.md.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agent.prompts import BRIEF_PROMPT as _BRIEF_PROMPT


def _build_registry():
    from agent.tools.finance import get_stock_data
    from agent.tools.search import web_search
    from agent.tools.sentiment import analyze_news_sentiment
    from agent.tools.rag import query_10k
    from agent.tools.financial_metrics import get_financial_metrics
    from agent.tools.valuation import get_valuation
    from agent.tools.competitor import get_competitor_analysis
    from agent.tools.earnings import get_earnings
    from agent.tools.technical import get_technical_analysis
    from agent.registry import ToolRegistry

    registry = ToolRegistry()
    for tool in [
        get_stock_data, web_search, analyze_news_sentiment, query_10k,
        get_financial_metrics, get_valuation, get_competitor_analysis,
        get_earnings, get_technical_analysis,
    ]:
        registry.register(tool)
    return registry


def generate_brief(ticker: str) -> str:
    """Run the 4-tool agent and return the formatted Markdown brief."""
    from agent.loop import run_agent
    from providers.factory import get_provider
    from agent.tracing import configure_logging

    configure_logging()

    provider = get_provider("openai", "gpt-4o-mini")
    registry = _build_registry()

    today = date.today().strftime("%B %d, %Y")
    from agent.data_quality import compute_dq_block
    dq_precheck = compute_dq_block(ticker.upper()).replace("{", "[").replace("}", "]")
    prompt = _BRIEF_PROMPT.format(ticker=ticker.upper(), date=today, dq_precheck=dq_precheck)

    result = run_agent(
        query=prompt,
        provider=provider,
        registry=registry,
        max_iterations=10,
    )
    return result.answer


def save_brief(ticker: str, content: str) -> Path:
    """Write brief to briefs/<TICKER>.md and return the path."""
    out_dir = Path(__file__).parent / "briefs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{ticker.upper()}.md"
    out_path.write_text(content, encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="brief",
        description="Generate a 2-page structured investment brief for a stock ticker.",
    )
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. NVDA, TSLA, MSFT)")
    parser.add_argument(
        "--print", dest="print_output", action="store_true",
        help="Also print the brief to stdout",
    )
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"[brief] Generating brief for {ticker}…", file=sys.stderr)

    content = generate_brief(ticker)
    path = save_brief(ticker, content)

    print(f"[brief] Saved → {path}", file=sys.stderr)

    if args.print_output:
        print(content)


if __name__ == "__main__":
    main()
