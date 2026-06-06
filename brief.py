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

# ── Brief prompt template ─────────────────────────────────────────────────────
_BRIEF_PROMPT = """\
Produce a 2-page structured investment brief for {ticker} dated {date}.

STEP 1 — Call these tools to gather data (as many in parallel as possible):
  • get_stock_data: ticker="{ticker}", period="1mo"
  • analyze_news_sentiment: ticker="{ticker}", top_n=5
  • web_search: query="{ticker} latest news analyst outlook 2025"
  • query_10k: ticker="{ticker}", question="What are the top risk factors and key business uncertainties?"

If query_10k returns an error (ticker not indexed), immediately call:
  • web_search: query="{ticker} 10-K annual report key risk factors 2024"

STEP 2 — Using ONLY the data returned by those tools, output this EXACT Markdown document.
Output the Markdown directly — no preamble, no code fences, no extra commentary.

# {ticker} — Investment Brief
**Date:** {date}
**Analyst Engine:** Finance Sentiment Engine v0.2.0

---

## 1. Company Snapshot

| Metric | Value |
|--------|-------|
| Current Price | $[price from get_stock_data] |
| Market Cap | $[market_cap formatted as $XB] |
| 1-Month Return | [pct_change]% |

[One paragraph (2–3 sentences) describing what {ticker} does and its market position.]

---

## 2. Recent News & Sentiment

[For each news article from analyze_news_sentiment, write one bullet:]
- [🟢 bullish / 🔴 bearish / ⚪ neutral] **[key_event]** — [one-line summary from the article]

**Overall Sentiment:** [bullish/bearish/neutral] — [One sentence justification based on the sentiment distribution above.]

---

## 3. Key Risk Factors

[List exactly 5 risk factors from query_10k answer or web_search. Use bullet points with a bold risk name followed by a brief description.]

---

## 4. Analyst Verdict

| | |
|---|---|
| **Recommendation** | Buy / Hold / Watch |
| **Key Opportunity** | [from news/sentiment data — one line] |
| **Key Risk** | [top risk from section 3 — one line] |

> [One-sentence analyst summary tying together the sentiment, risks, and price action.]

---

## 5. Sources

[Bullet list of all URLs returned by web_search, plus:]
- Stock data: Yahoo Finance via yfinance (get_stock_data)
- Sentiment: Yahoo Finance RSS + GPT-4 analysis (analyze_news_sentiment)
[If query_10k was used: - 10-K: SEC EDGAR via Qdrant RAG (query_10k)]
"""


def _build_registry():
    from agent.tools.finance import get_stock_data
    from agent.tools.search import web_search
    from agent.tools.sentiment import analyze_news_sentiment
    from agent.tools.rag import query_10k
    from agent.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(get_stock_data)
    registry.register(web_search)
    registry.register(analyze_news_sentiment)
    registry.register(query_10k)
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
    prompt = _BRIEF_PROMPT.format(ticker=ticker.upper(), date=today)

    result = run_agent(
        query=prompt,
        provider=provider,
        registry=registry,
        max_iterations=8,
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
