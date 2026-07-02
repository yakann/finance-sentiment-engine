"""Streamlit demo UI for Finance Sentiment Engine."""
from __future__ import annotations

import os
import re
import time

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "https://finance-sentiment-engine.fly.dev")

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
    try:
        r = requests.delete(f"{API_URL}/cache/{ticker}", timeout=10)
        if r.status_code == 200:
            st.success(f"Cache cleared for **{ticker}**.")
        else:
            st.info(f"No cache entry found for **{ticker}** today.")
    except Exception:
        st.warning("Cache service not reachable.")

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run and ticker_base:
    t0 = time.perf_counter()
    cached_brief = False

    with st.status(f"Analyzing **{ticker}**…", expanded=True) as pipeline_status:
        st.write("📰 Collecting news & running sentiment analysis…")
        st.write("💹 Fetching live price data…")
        st.write("📄 Querying 10-K risk factors via RAG…")
        st.write("✍️ Generating investment brief…")

        try:
            resp = requests.post(
                f"{API_URL}/run",
                json={"ticker": ticker},
                timeout=300,
            )
            if resp.status_code == 404:
                pipeline_status.update(label="❌ Ticker not found", state="error")
                st.error(
                    f"**Ticker not found:** `{ticker}`\n\n"
                    "Check the symbol and exchange — e.g. for BIST use `THYAO` + Turkey (BIST)."
                )
                st.stop()
            resp.raise_for_status()
            data = resp.json()
            pipeline_status.update(label=f"✅ {ticker} brief ready!", state="complete")
        except requests.exceptions.Timeout:
            pipeline_status.update(label="❌ Request timed out", state="error")
            st.error("The analysis took too long (>5 min). Try again or use a major ticker like NVDA.")
            st.stop()
        except Exception as exc:
            pipeline_status.update(label=f"❌ Error: {type(exc).__name__}", state="error")
            st.exception(exc)
            st.stop()

    elapsed = time.perf_counter() - t0
    answer = data.get("brief", "")
    iterations = data.get("iterations", 0)
    total_tokens = data.get("total_tokens", 0)
    cached_brief = iterations == 0 and total_tokens == 0

    daily_spend: float | None = None

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
