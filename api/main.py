"""Day 36 + Day 37 + Day 39: FastAPI wrapper for the finance-sentiment agent.

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
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Finance Sentiment Engine",
    description="HTTP interface for the finance-sentiment agent pipeline",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]


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


from cache.cache_utils import get_cache as _get_cache


@app.post("/run", response_model=AgentResponse)
@limiter.limit("10/minute")
def run(request: Request, req: RunRequest) -> AgentResponse:
    """Run the 4-tool agent for the given ticker and return the investment brief.

    Day 38: results are cached in Redis using semantic vector similarity so that
    identical (or near-identical) queries are answered in < 100 ms without
    re-running the full agent pipeline.

    Day 39: rate-limited to 10 req/min per IP; daily budget guard blocks requests
    once $5 of LLM spend is reached for the calendar day.
    """
    from agent.loop import run_agent
    from agent.tracing import configure_logging
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
    from providers.factory import get_provider
    from middleware.cost import track_cost, check_budget, BudgetExceededError

    configure_logging()

    # ── Daily budget guard ─────────────────────────────────────────────────────
    try:
        check_budget()
    except BudgetExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    ticker = req.ticker.upper()
    today = date.today().strftime("%B %d, %Y")
    from agent.data_quality import compute_dq_block
    dq_precheck = compute_dq_block(ticker).replace("{", "[").replace("}", "]")
    prompt = _BRIEF_PROMPT.format(ticker=ticker, date=today, dq_precheck=dq_precheck)

    # ── Semantic cache lookup (free — no LLM cost) ────────────────────────────
    sc = _get_cache()
    if sc is not None:
        cached = sc.get(prompt)
        if cached is not None:
            return AgentResponse(
                ticker=ticker,
                brief=cached,
                iterations=0,
                total_tokens=0,
                trace_path=None,
            )

    # ── Cache miss → run agent ─────────────────────────────────────────────────
    registry = ToolRegistry()
    registry.register(get_stock_data)
    registry.register(web_search)
    registry.register(analyze_news_sentiment)
    registry.register(query_10k)
    registry.register(get_financial_metrics)
    registry.register(get_valuation)
    registry.register(get_competitor_analysis)
    registry.register(get_earnings)
    registry.register(get_technical_analysis)

    _MODEL = "gpt-4o-mini"
    provider = get_provider("openai", _MODEL)

    try:
        result = run_agent(
            query=prompt,
            provider=provider,
            registry=registry,
            max_iterations=10,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ── Track cost ─────────────────────────────────────────────────────────────
    track_cost(_MODEL, total_tokens=result.total_tokens)

    # ── Store result in cache ──────────────────────────────────────────────────
    if sc is not None:
        try:
            sc.set(prompt, result.answer)
        except Exception:
            pass

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
@limiter.limit("10/minute")
async def stream(request: Request, req: StreamRequest) -> StreamingResponse:  # noqa: ARG001
    """LangGraph pipeline'ını çalıştırır ve ilerlemeyi SSE olarak akıtır.

    Day 39: rate-limited to 10 req/min per IP.

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


# ── Day 39: Cost stats endpoint ───────────────────────────────────────────────

@app.get("/stats")
def stats(day: str = Query(default=None, description="Date in YYYY-MM-DD format (default: today)")) -> dict:
    """Return daily LLM cost summary from Redis.

    Example:
        GET /stats
        GET /stats?day=2026-06-28
    """
    from middleware.cost import get_daily_stats
    return get_daily_stats(day)


# ── Brief prompt (canonical version lives in agent/prompts.py) ────────────────

from agent.prompts import BRIEF_PROMPT as _BRIEF_PROMPT  # noqa: E402
