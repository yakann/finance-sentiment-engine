"""Day 41: Streamlit demo UI for Finance Sentiment Engine."""
from __future__ import annotations

import asyncio
import re
import time
from datetime import date

import nest_asyncio
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit's async runtime may have a loop already running — allow nesting.
nest_asyncio.apply()

st.set_page_config(
    page_title="Finance Sentiment Engine",
    page_icon="📈",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Finance Sentiment Engine")
st.caption(
    "AI-powered investment briefs · LangGraph + GPT-4o-mini · "
    "[Source](https://github.com) · Day 41"
)

# ── Input row ─────────────────────────────────────────────────────────────────
ticker_raw = st.text_input(
    "Stock ticker",
    placeholder="NVDA · TSLA · AAPL · MSFT",
    label_visibility="collapsed",
)
ticker = ticker_raw.upper().strip()

run = st.button("🚀 Run Analysis", type="primary", disabled=not ticker)

# ── Pipeline execution ─────────────────────────────────────────────────────────
if run and ticker:
    _STREAM_NODES = {
        "collect_news",
        "call_model",
        "dispatch_tools",
        "analyze_sentiment",
        "deep_analysis",
        "short_brief",
        "fetch_price",
        "draft",
    }

    _NODE_LABELS: dict[str, str] = {
        "collect_news": "📰 Collecting news & running sentiment…",
        "call_model": "🤖 LLM calling tools…",
        "dispatch_tools": "🔧 Dispatching tools…",
        "analyze_sentiment": "📊 Aggregating sentiment…",
        "deep_analysis": "📄 10-K RAG deep dive…",
        "short_brief": "✍️ Drafting quick brief…",
        "fetch_price": "💹 Fetching live price…",
        "draft": "📝 Writing investment brief…",
    }

    completed: list[str] = []
    draft_text: str = ""

    t0 = time.perf_counter()

    # ── Async graph streaming ─────────────────────────────────────────────────
    with st.status(f"Analyzing **{ticker}**…", expanded=True) as pipeline_status:
        node_slot = st.empty()

        async def _stream_graph() -> str:
            from graph.finance_graph import build_finance_graph
            graph = build_finance_graph()
            result_draft = ""
            async for event in graph.astream_events(
                {"ticker": ticker, "messages": []}, version="v2"
            ):
                kind = event.get("event")
                name = event.get("name")
                if name not in _STREAM_NODES:
                    continue
                if kind == "on_chain_start":
                    node_slot.markdown(_NODE_LABELS.get(name, f"⚙️ {name}…"))
                elif kind == "on_chain_end":
                    completed.append(name)
                    node_slot.markdown(
                        " → ".join(f"✅ `{n}`" for n in completed[-4:])
                    )
                    out = event.get("data", {}).get("output")
                    if name == "draft" and isinstance(out, dict) and out.get("draft"):
                        result_draft = out["draft"]
            return result_draft

        try:
            draft_text = asyncio.run(_stream_graph())
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

    # ── Cost tracking ─────────────────────────────────────────────────────────
    daily_spend: float | None = None
    try:
        from middleware.cost import get_daily_stats
        daily_spend = get_daily_stats().get("total_usd")
    except Exception:
        pass

    # ── Metric badges ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("⏱️ Latency", f"{elapsed:.1f}s")
    c2.metric("✅ Steps", len(completed))
    if daily_spend is not None:
        c3.metric("💰 Daily Spend", f"${daily_spend:.4f}")
    else:
        c3.metric("💰 Cost Tracking", "—")

    st.divider()

    # ── Streaming output ──────────────────────────────────────────────────────
    if draft_text:
        def _word_stream():
            for chunk in re.findall(r"\S+\s*", draft_text):
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
        2. **Click Run** — LangGraph pipeline starts
        3. **Watch** live node progress
        4. **Read** the brief — streamed word by word

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

    # Live cost widget (gracefully absent if Redis is offline)
    try:
        from middleware.cost import get_daily_stats
        s = get_daily_stats()
        st.metric("Today's API spend", f"${s.get('total_usd', 0):.4f}")
        st.caption(f"{s.get('calls', 0)} calls · {s.get('input_tokens', 0) + s.get('output_tokens', 0):,} tokens")
    except Exception:
        st.caption("Cost tracking offline (Redis not running)")
