"""Day 41: Streamlit demo UI for Finance Sentiment Engine."""
from __future__ import annotations

import re
import time
from datetime import date

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Finance Sentiment Engine",
    page_icon="📈",
    layout="centered",
)

# Yahoo Finance exchange suffix map
EXCHANGES: dict[str, str] = {
    "🇺🇸 US (NYSE / NASDAQ)": "",
    "🇹🇷 Turkey (BIST)": ".IS",
    "🇩🇪 Germany (XETRA)": ".DE",
    "🇬🇧 UK (LSE)": ".L",
    "🇯🇵 Japan (TSE)": ".T",
    "🇫🇷 France (Euronext Paris)": ".PA",
    "🇳🇱 Netherlands (Euronext Amsterdam)": ".AS",
    "🇧🇪 Belgium (Euronext Brussels)": ".BR",
    "🇮🇹 Italy (Borsa Italiana)": ".MI",
    "🇪🇸 Spain (BME)": ".MC",
    "🇨🇭 Switzerland (SIX)": ".SW",
    "🇭🇰 Hong Kong (HKEX)": ".HK",
    "🇨🇳 China Shanghai (SSE)": ".SS",
    "🇨🇳 China Shenzhen (SZSE)": ".SZ",
    "🇮🇳 India NSE": ".NS",
    "🇮🇳 India BSE": ".BO",
    "🇧🇷 Brazil (B3)": ".SA",
    "🇨🇦 Canada (TSX)": ".TO",
    "🇦🇺 Australia (ASX)": ".AX",
    "🇰🇷 South Korea (KRX)": ".KS",
    "🇸🇪 Sweden (Nasdaq Stockholm)": ".ST",
    "🇳🇴 Norway (Oslo Børs)": ".OL",
    "🇩🇰 Denmark (Nasdaq Copenhagen)": ".CO",
    "🇸🇬 Singapore (SGX)": ".SI",
    "🇿🇦 South Africa (JSE)": ".JO",
    "🇲🇽 Mexico (BMV)": ".MX",
}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Finance Sentiment Engine")
st.caption("AI-powered investment briefs · LangGraph + GPT-4o-mini · Day 41")

# ── Input row ─────────────────────────────────────────────────────────────────
col_ticker, col_exchange = st.columns([2, 3])

with col_ticker:
    ticker_raw = st.text_input(
        "Stock ticker",
        placeholder="NVDA · THYAO · BMW · 7203",
        label_visibility="visible",
    )

with col_exchange:
    exchange_label = st.selectbox(
        "Exchange",
        options=list(EXCHANGES.keys()),
        index=0,
    )

suffix = EXCHANGES[exchange_label]
ticker_base = ticker_raw.upper().strip()
# Only append suffix if user hasn't already typed it (e.g. "THYAO.IS")
if suffix and not ticker_base.endswith(suffix):
    ticker = ticker_base + suffix
else:
    ticker = ticker_base

if ticker_base and suffix:
    st.caption(f"Resolved Yahoo Finance symbol: **{ticker}**")

col_run, col_clear = st.columns([3, 2])

with col_run:
    run = st.button("🚀 Run Analysis", type="primary", disabled=not ticker_base, use_container_width=True)

with col_clear:
    clear = st.button("🗑️ Clear cache", disabled=not ticker_base, use_container_width=True)

if clear and ticker_base:
    from cache.cache_utils import get_cache as _get_cache_for_clear
    from datetime import date as _date
    _sc = _get_cache_for_clear()
    if _sc:
        _key = f"{ticker}::{_date.today().strftime('%B %d, %Y')}"
        deleted = _sc.delete(_key)
        if deleted:
            st.success(f"Cache cleared for **{ticker}**.")
        else:
            st.info(f"No cache entry found for **{ticker}** today.")
    else:
        st.warning("Redis not reachable — nothing to clear.")

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run and ticker_base:
    from agent.loop import run_agent
    from agent.registry import ToolRegistry
    from agent.tools.finance import _GetStockDataInput, _get_stock_data, get_stock_data
    from agent.tools.rag import query_10k
    from agent.tools.search import web_search
    from agent.tools.sentiment import analyze_news_sentiment
    from agent.tools.financial_metrics import get_financial_metrics
    from agent.tools.valuation import get_valuation
    from agent.tools.competitor import get_competitor_analysis
    from agent.tools.earnings import get_earnings
    from agent.tools.technical import get_technical_analysis
    from agent.tracing import configure_logging
    from providers.factory import get_provider

    # ── Ticker validation (early exit) ───────────────────────────────────────
    with st.spinner(f"Validating ticker **{ticker}**…"):
        probe = _get_stock_data(_GetStockDataInput(ticker=ticker))
    if "error" in probe:
        st.error(
            f"**Ticker not found:** `{ticker}`\n\n"
            f"{probe['error']}\n\n"
            f"Check the symbol and exchange — e.g. for BIST use `THYAO` + Turkey (BIST)."
        )
        st.stop()

    configure_logging()

    registry = ToolRegistry()
    for tool in [
        get_stock_data, web_search, analyze_news_sentiment, query_10k,
        get_financial_metrics, get_valuation, get_competitor_analysis,
        get_earnings, get_technical_analysis,
    ]:
        registry.register(tool)

    provider = get_provider("openai", "gpt-4o-mini")
    today = date.today().strftime("%B %d, %Y")

    from agent.prompts import BRIEF_PROMPT as _BRIEF_PROMPT
    from cache.cache_utils import get_cache as _get_cache
    from agent.data_quality import compute_dq_block as _compute_dq
    dq_precheck = _compute_dq(ticker).replace("{", "[").replace("}", "]")
    prompt = _BRIEF_PROMPT.format(ticker=ticker, date=today, dq_precheck=dq_precheck)

    # ── Semantic cache lookup ─────────────────────────────────────────────────
    # Cache key = ticker::date (NOT the full prompt).
    # The prompt template is ~3 KB of nearly identical text across tickers;
    # embedding similarity easily exceeds 0.95, causing cross-ticker false hits.
    # A short, specific key like "THYAO.IS::June 29, 2026" is unambiguous.
    sc = _get_cache()
    _cache_key = f"{ticker}::{today}"
    cached_brief: str | None = sc.get(_cache_key) if sc else None

    t0 = time.perf_counter()

    if cached_brief:
        # Cache hit — skip agent entirely
        elapsed = time.perf_counter() - t0
        st.success(f"⚡ Cache hit — served instantly for **{ticker}**")
        answer = cached_brief
        iterations = 0
        total_tokens = 0
    else:
        # Cache miss — run the full pipeline
        with st.status(f"Analyzing **{ticker}**…", expanded=True) as pipeline_status:
            st.write("📰 Collecting news & running sentiment analysis…")
            st.write("💹 Fetching live price data…")
            st.write("📄 Querying 10-K risk factors via RAG…")
            st.write("✍️ Generating investment brief…")

            try:
                result = run_agent(
                    query=prompt,
                    provider=provider,
                    registry=registry,
                    max_iterations=10,
                )
                pipeline_status.update(
                    label=f"✅ {ticker} brief ready!", state="complete"
                )
            except Exception as exc:
                pipeline_status.update(
                    label=f"❌ Pipeline error: {type(exc).__name__}", state="error"
                )
                st.exception(exc)
                st.stop()

        elapsed = time.perf_counter() - t0
        answer = result.answer
        iterations = result.iterations
        total_tokens = result.total_tokens

        # Store in cache for next time
        if sc and answer:
            try:
                sc.set(_cache_key, answer)
            except Exception:
                pass

    # ── Cost tracking ─────────────────────────────────────────────────────────
    daily_spend: float | None = None
    try:
        from middleware.cost import get_daily_stats, track_cost
        if not cached_brief:
            track_cost("gpt-4o-mini", total_tokens=total_tokens)
        daily_spend = get_daily_stats().get("total_usd")
    except Exception:
        pass

    # ── Metric badges ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("⏱️ Latency", f"{elapsed:.1f}s")
    c2.metric("🔄 Iterations", iterations if not cached_brief else "cached")
    c3.metric(
        "💰 Daily Spend",
        f"${daily_spend:.4f}" if daily_spend is not None else "—",
    )

    st.divider()

    # ── Streaming output (typewriter) ─────────────────────────────────────────
    if answer:
        def _word_stream():
            for chunk in re.findall(r"\S+\s*", answer):
                time.sleep(0.015)
                yield chunk

        st.write_stream(_word_stream())
    else:
        st.warning(f"No brief generated for **{ticker}**. Try a major ticker like NVDA.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
        1. **Type** a ticker (NVDA, TSLA, AAPL…)
        2. **Click Run** — 4-tool agent starts
        3. **Watch** the pipeline steps
        4. **Read** the brief streamed word by word

        ---

        **Tools per run:**
        | Tool | Source |
        |------|--------|
        | 🔍 Web search | Tavily |
        | 📰 Sentiment | Yahoo RSS + GPT |
        | 💹 Price | yfinance |
        | 📄 10-K RAG | Qdrant + Cohere |

        ---
        **Stack:** LangGraph · OpenAI · Redis cache
        **Built in:** 41 days of daily AI practice
        """
    )

    st.divider()

    try:
        from middleware.cost import get_daily_stats
        s = get_daily_stats()
        st.metric("Today's API spend", f"${s.get('total_usd', 0):.4f}")
        total_tok = s.get("input_tokens", 0) + s.get("output_tokens", 0)
        st.caption(f"{s.get('calls', 0)} calls · {total_tok:,} tokens")
    except Exception:
        st.caption("Cost tracking offline (no Redis)")
