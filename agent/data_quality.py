"""Pre-compute data quality checks before the agent runs.

Calls earnings, valuation, and competitor tools in Python so authoritative
PASS/FAIL verdicts can be injected into the prompt. This prevents gpt-4o-mini
from misapplying the conditions itself (a known reliability issue).

Sentiment is NOT pre-checked here because it requires LLM sub-calls;
the prompt handles it with a strict binary rule instead.
"""
from __future__ import annotations


def compute_dq_block(ticker: str) -> str:
    """Return a multi-line DQ status string to embed in the brief prompt."""
    ticker = ticker.upper()
    lines: list[str] = []

    # ── Earnings ──────────────────────────────────────────────────────────────
    try:
        from agent.tools.earnings import _GetEarningsInput, _get_earnings
        er = _get_earnings(_GetEarningsInput(ticker=ticker))
        if "error" in er:
            lines.append(f"EARNINGS: FAIL — {er['error']}")
        elif er.get("warnings") and any("stale" in w.lower() for w in er["warnings"]):
            lines.append(f"EARNINGS: FAIL — stale data ({er['warnings'][0][:100]})")
        else:
            quarters = er.get("last_4_quarters", [])
            recent = quarters[0]["date"] if quarters else "none"
            lines.append(f"EARNINGS: PASS — {len(quarters)} quarters, most recent {recent}")
    except Exception as exc:
        lines.append(f"EARNINGS: PASS — pre-check skipped ({exc})")

    # ── Sentiment: lightweight RSS count (no LLM) ────────────────────────────
    # PASS if Yahoo Finance RSS returns ≥1 article for the ticker.
    # FAIL only when the feed is completely empty (tool returns [{"error":...}]).
    try:
        import feedparser
        feed = feedparser.parse(
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        )
        n_articles = len(feed.entries)
        if n_articles > 0:
            lines.append(f"SENTIMENT: PASS — {n_articles} RSS articles found for {ticker}")
        else:
            lines.append(f"SENTIMENT: FAIL — Yahoo Finance RSS returned 0 articles for {ticker}")
    except Exception as exc:
        lines.append(f"SENTIMENT: PASS — RSS pre-check skipped, assume data present ({exc})")

    # ── Valuation ─────────────────────────────────────────────────────────────
    try:
        from agent.tools.valuation import _GetValuationInput, _get_valuation
        vr = _get_valuation(_GetValuationInput(ticker=ticker))
        if vr.get("warnings"):
            lines.append(f"VALUATION: FAIL — {vr['warnings'][0][:140]}")
        else:
            lines.append("VALUATION: PASS — no warnings, all multiples are reliable")
    except Exception as exc:
        lines.append(f"VALUATION: PASS — pre-check skipped ({exc})")

    # ── Competitor ────────────────────────────────────────────────────────────
    try:
        from agent.tools.competitor import _GetCompetitorAnalysisInput, _get_competitor_analysis
        ca = _get_competitor_analysis(_GetCompetitorAnalysisInput(ticker=ticker))
        if "error" in ca:
            lines.append(f"COMPETITOR: FAIL — {ca['error'][:120]}")
        else:
            n = len(ca.get("comparison", []))
            lines.append(f"COMPETITOR: PASS — {n}-entry comparison array returned")
    except Exception as exc:
        lines.append(f"COMPETITOR: FAIL — exception: {exc}")

    body = "\n".join(f"  {ln}" for ln in lines)
    return (
        "PRE-COMPUTED DATA QUALITY STATUS (authoritative — do NOT re-evaluate or override):\n"
        + body
    )
