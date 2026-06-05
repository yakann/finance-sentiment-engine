"""Day 18: Sentiment tool — wraps Week-1 analyzer as an agent-callable tool."""

import json
from pathlib import Path

from pydantic import BaseModel

from agent.tools.base import Tool
from schemas import NewsAnalysis

_CACHE_PATH = Path(__file__).parent.parent.parent / "cache" / "news.jsonl"


def _load_cached_news(ticker: str, limit: int) -> list[dict]:
    """Read most-recent news for ticker from the JSONL cache (fallback path)."""
    if not _CACHE_PATH.exists():
        return []
    items: list[dict] = []
    with _CACHE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("ticker", "").upper() == ticker.upper():
                    items.append(item)
            except json.JSONDecodeError:
                pass
    return items[-limit:]


class _AnalyzeNewsSentimentInput(BaseModel):
    ticker: str
    top_n: int = 5


def _analyze_news_sentiment(inputs: _AnalyzeNewsSentimentInput) -> list[dict]:
    from scraper.yahoo import fetch_news, NewsItem
    from analyzer.openai_analyzer import analyze_news

    ticker = inputs.ticker.upper()

    # Try live RSS first; fall back to cache if all articles already seen
    items = fetch_news([ticker], limit=inputs.top_n)
    if not items:
        raw = _load_cached_news(ticker, inputs.top_n)
        items = [NewsItem(**r) for r in raw]

    if not items:
        return [{"error": f"No news found for {ticker}"}]

    results: list[dict] = []
    for item in items[: inputs.top_n]:
        analysis: NewsAnalysis = analyze_news(item.title, item.summary, ticker)
        results.append(analysis.model_dump())

    return results


analyze_news_sentiment = Tool(
    name="analyze_news_sentiment",
    description=(
        "Fetches recent financial news for a stock ticker and runs LLM sentiment analysis "
        "on each article. Returns a list of analyses, each with: sentiment "
        "(bullish/bearish/neutral), urgency (high/medium/low), key_event category, "
        "and a short summary. Use this to assess market sentiment before making a "
        "bullish/bearish recommendation on a stock."
    ),
    input_schema=_AnalyzeNewsSentimentInput,
    handler=_analyze_news_sentiment,
)
