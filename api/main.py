"""Day 36 + Day 37: FastAPI wrapper for the finance-sentiment agent.

Usage:
    uvicorn api.main:app --reload

Endpoints:
    POST /run    — runs the 4-tool agent for a given ticker (blocking, JSON)
    POST /stream — runs the LangGraph pipeline and streams progress as SSE
    GET  /health — liveness check

Day 37 — SSE streaming (POST /stream):
    The LangGraph finance pipeline is driven with `astream_events`. Each graph
    node start/end becomes a `tool_start` / `tool_end` Server-Sent Event so a UI
    can show live progress ("collecting news…", "fetching price…"). When the
    pipeline finishes, the assembled `draft` is streamed back word-by-word as
    `token` events (the "typewriter" effect the UI wants), followed by a `final`
    event carrying the full text.

    Note on real LLM tokens: this project's providers (`provider.generate()`) are
    custom synchronous SDK wrappers, NOT LangChain chat models, so `astream_events`
    never emits `on_chat_model_stream` events — and the visible answer (`draft`) is
    assembled deterministically in `write_draft`, not produced by an LLM token
    stream. Word-by-word chunking of the final draft is therefore the correct way
    to deliver the typewriter UX here. True per-LLM-token streaming would require
    porting the providers to streaming LangChain `ChatModel`s (a later day).
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import date
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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


class StreamRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. TSLA")


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


# ── Day 37: SSE streaming endpoint ────────────────────────────────────────────

# Yalnızca bu node'lar için tool_start/tool_end yayınlanır. LangGraph'ın iç
# Runnable'ları (RunnableSequence, ChannelWrite, ...) gürültü yaratmasın diye
# ana pipeline node'larını + research subgraph'ın tool-loop node'larını süzeriz.
_STREAM_NODES = {
    "collect_news",      # research subgraph (haber + sentiment toplama)
    "call_model",        # subgraph: LLM tool-use turu
    "dispatch_tools",    # subgraph: analyze_news_sentiment çalıştırma
    "analyze_sentiment", # bullish/bearish/neutral özeti
    "deep_analysis",     # 10-K RAG dalı
    "short_brief",       # neutral hızlı özet dalı
    "fetch_price",       # yfinance fiyat verisi
    "draft",             # final brief taslağı
}

# Her token arasına minik bir gecikme — UI'da "yazılıyor" hissini verir.
# 0 yaparsan akış olabildiğince hızlı boşalır.
_TOKEN_DELAY_S = 0.02


def _sse(event: str, data: dict) -> str:
    """Bir Server-Sent Event çerçevesi biçimlendirir (event + JSON data + boş satır)."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _chunk_tokens(text: str) -> list[str]:
    """Metni, boşlukları koruyan kelime parçalarına böler (typewriter için).

    re ile her parça 'kelime + ardındaki boşluk(lar)' olur; böylece parçaları
    ardışık birleştirince orijinal metin (newline'lar dahil) aynen geri gelir.
    """
    return re.findall(r"\S+\s*", text) or ([text] if text else [])


async def _event_stream(ticker: str) -> AsyncIterator[str]:
    """LangGraph pipeline'ını `astream_events` ile sürer ve SSE çerçeveleri üretir.

    Yayınlanan event tipleri:
        tool_start : bir node çalışmaya başladı   → {"node": "..."}
        tool_end   : bir node tamamlandı           → {"node": "..."}
        token      : final draft'ın bir parçası    → {"text": "..."}
        final      : tüm draft tek seferde          → {"text": "...", "ticker": "..."}
        error      : pipeline patladı               → {"message": "..."}
    """
    # Lazy import: app başlangıcını hafif tutar ve test'te patch'lenebilir kılar.
    from graph.finance_graph import build_finance_graph

    # interrupt yok, checkpointer yok → graf END'e kadar kesintisiz koşar.
    graph = build_finance_graph()

    draft = ""
    try:
        async for event in graph.astream_events(
            {"ticker": ticker, "messages": []},
            version="v2",
        ):
            kind = event.get("event")
            name = event.get("name")

            if name not in _STREAM_NODES:
                continue

            if kind == "on_chain_start":
                yield _sse("tool_start", {"node": name})
            elif kind == "on_chain_end":
                yield _sse("tool_end", {"node": name})
                # draft node'unun çıktısından final metni yakala.
                output = event.get("data", {}).get("output")
                if name == "draft" and isinstance(output, dict) and output.get("draft"):
                    draft = output["draft"]
    except Exception as exc:  # pipeline hatası → istemciye structured error
        yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})
        return

    # ── Typewriter: final draft'ı parça parça token olarak yolla ───────────────
    if not draft:
        draft = f"[no draft produced for {ticker}]"

    for piece in _chunk_tokens(draft):
        yield _sse("token", {"text": piece})
        if _TOKEN_DELAY_S:
            await asyncio.sleep(_TOKEN_DELAY_S)

    # ── Bitiş: tüm metni tek seferde gönder (UI son halini doğrulasın) ─────────
    yield _sse("final", {"ticker": ticker, "text": draft})


@app.post("/stream")
async def stream(req: StreamRequest) -> StreamingResponse:
    """LangGraph pipeline'ını çalıştırır ve ilerlemeyi SSE olarak akıtır.

    Örnek:
        curl -N localhost:8000/stream \\
             -d '{"ticker":"TSLA"}' -H 'Content-Type: application/json'
    """
    ticker = req.ticker.upper()
    return StreamingResponse(
        _event_stream(ticker),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx buffer'lamasın → anında flush
        },
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
