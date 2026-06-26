"""Day 36: FastAPI wrapper for the finance-sentiment agent.

Usage:
    uvicorn api.main:app --reload

Endpoints:
    POST /run   — runs the 4-tool agent for a given ticker
    GET  /health — liveness check
"""
from __future__ import annotations

from datetime import date

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(
    title="Finance Sentiment Engine",
    description="HTTP interface for the finance-sentiment agent pipeline",
    version="1.0.0",
)


# ── Request / Response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. NVDA")


class AgentResponse(BaseModel):
    ticker: str
    brief: str = Field(..., description="Markdown investment brief produced by the agent")
    iterations: int
    total_tokens: int
    trace_path: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/run", response_model=AgentResponse)
def run(req: RunRequest) -> AgentResponse:
    """Run the 4-tool agent for the given ticker and return the investment brief."""
    from agent.loop import run_agent
    from agent.tracing import configure_logging
    from agent.tools.finance import get_stock_data
    from agent.tools.search import web_search
    from agent.tools.sentiment import analyze_news_sentiment
    from agent.tools.rag import query_10k
    from agent.registry import ToolRegistry
    from providers.factory import get_provider

    configure_logging()

    ticker = req.ticker.upper()

    registry = ToolRegistry()
    registry.register(get_stock_data)
    registry.register(web_search)
    registry.register(analyze_news_sentiment)
    registry.register(query_10k)

    provider = get_provider("openai", "gpt-4o-mini")

    today = date.today().strftime("%B %d, %Y")
    prompt = _BRIEF_PROMPT.format(ticker=ticker, date=today)

    try:
        result = run_agent(
            query=prompt,
            provider=provider,
            registry=registry,
            max_iterations=8,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AgentResponse(
        ticker=ticker,
        brief=result.answer,
        iterations=result.iterations,
        total_tokens=result.total_tokens,
        trace_path=result.trace_path,
    )


# ── Brief prompt (shared with brief.py CLI) ───────────────────────────────────

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
