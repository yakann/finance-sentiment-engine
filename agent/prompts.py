"""Canonical investment brief prompt shared by api/main.py, app.py, and brief.py."""
from __future__ import annotations

BRIEF_PROMPT = """\
Produce a structured investment brief for {ticker} dated {date}.

STEP 1 — Call ALL of these tools to gather data (as many in parallel as possible):
  • get_stock_data: ticker="{ticker}", period="1mo"
  • get_financial_metrics: ticker="{ticker}"
  • get_valuation: ticker="{ticker}"
  • get_earnings: ticker="{ticker}"
  • get_technical_analysis: ticker="{ticker}"
  • get_competitor_analysis: ticker="{ticker}"
  • analyze_news_sentiment: ticker="{ticker}", top_n=5
  • web_search: query="{ticker} latest news analyst outlook 2025"
  • query_10k: ticker="{ticker}", question="What are the top risk factors and key business uncertainties?"
  • query_10k: ticker="{ticker}", question="What are management's key strategic priorities, competitive advantages, and growth initiatives?"

If query_10k returns an error (ticker not indexed), immediately call:
  • web_search: query="{ticker} 10-K annual report key risk factors strategic priorities 2024"
If get_competitor_analysis returns an error (ticker not in peer map), omit Section 4.

STEP 2 — DATA QUALITY GATE (run BEFORE writing any section):

{dq_precheck}

SENTIMENT RUNTIME CHECK — evaluate this NOW from your analyze_news_sentiment result:
  SENTIMENT FAIL: the tool returned a list whose ONLY item has an "error" key (e.g., no RSS feed
    articles found at all). Example: [{{"error": "No news found for TICKER"}}]
  SENTIMENT PASS: any other result — one or more items without an "error" key. Articles tagged
    "neutral", "other", or about related companies all count as valid data. Do NOT fail sentiment
    because articles are neutral or because some are about related tickers.

SCORING ADJUSTMENTS — apply each rule whose status above is FAIL:
  EARNINGS FAIL   → set Earnings score to 5, write "Stale data — not scored"
  SENTIMENT FAIL  → set Sentiment score to 5, write "No data"
  COMPETITOR FAIL → cap Fundamentals score at 10; write "Peer data unavailable" in Section 4
  VALUATION FAIL  → mark the affected metric(s) "⚠️ suspect" and exclude from Valuation score
  3+ checks FAIL  → append "**Recommendation withheld** — insufficient data quality for a
    reliable verdict." and do NOT output Buy/Sell/Hold

CRITICAL: If VALUATION status above is PASS, display ALL valuation numbers as plain values.
Do NOT add "⚠️ suspect" or any warning tag to any valuation cell. Do NOT apply the
"Deduct 3 if EV/EBITDA > 30" rule when VALUATION is PASS — only apply normal PEG scoring.

If ANY check FAILs, prepend a ⚠️ DATA QUALITY WARNING block after the document title and
before Section 1, listing each failed check.
The WARNING block must contain ONLY items from the PRE-COMPUTED STATUS above that show FAIL,
plus SENTIMENT if it failed the runtime check. Do NOT add any other items — in particular,
do NOT warn about query_10k being unavailable (use web_search as fallback silently).

STEP 3 — Using ONLY the data returned by those tools, output the Markdown document below.
Rules:
- Output the Markdown directly — no preamble, no code fences, no extra commentary.
- CRITICAL: Replace EVERY bracketed instruction like [xxx] with the actual value. Never leave bracket text in the output.
- CURRENCY: get_stock_data returns a "currency" field (e.g. "TRY", "EUR", "GBP"). Use the correct currency symbol everywhere — do NOT default to $ if currency is not USD. For TRY use ₺, EUR use €, GBP use £, otherwise use the ISO code.
- get_stock_data keys: "price" (float) → [currency symbol]X.XX | "pct_change" (float) → X.XX% | "market_cap_formatted" (string) → use as-is but replace the leading $ with the correct currency symbol

# {ticker} — Investment Brief
**Date:** {date}
**Analyst Engine:** Finance Sentiment Engine v0.2.0

---

## 1. Company Snapshot

| Metric | Value |
|--------|-------|
| Current Price | [currency symbol from get_stock_data][price from get_stock_data] |
| Market Cap | [market_cap_formatted from get_stock_data — replace leading $ with correct currency symbol] |
| 1-Month Return | [pct_change]% |
| Revenue Growth YoY | [revenue_growth_yoy_pct from get_financial_metrics]% |
| Gross Margin | [gross_margin_pct from get_financial_metrics]% |
| Debt / Equity | [debt_to_equity from get_financial_metrics] |

[One paragraph (2–3 sentences) describing what {ticker} does and its market position.]

---

## 2. Valuation

| Multiple | Value |
|----------|-------|
| Forward P/E | [forward_pe from get_valuation] |
| PEG Ratio | [peg_ratio from get_valuation] |
| EV/EBITDA | [ev_to_ebitda from get_valuation] |
| EV/Sales | [ev_to_sales from get_valuation] |
| Price/Book | [price_to_book from get_valuation] |

[One sentence interpreting whether the stock looks cheap, fairly valued, or expensive based on these multiples.]

---

## 3. Technical Analysis

| Indicator | Value |
|-----------|-------|
| RSI (14) | [rsi_14 from get_technical_analysis] |
| EMA 20 | [currency symbol][ema_20] |
| EMA 50 | [currency symbol][ema_50] |
| EMA 200 | [currency symbol][ema_200 — write "N/A" if null] |
| MACD Line | [macd.macd_line] |
| Signal Line | [macd.signal_line] |
| Trend | [trend_direction — bullish / bearish / neutral] |

[One sentence on momentum: e.g. RSI overbought/oversold, price vs EMAs, MACD crossover signal.]

---

## 4. Competitor Comparison

[Fill this table from get_competitor_analysis. Omit this section entirely if the tool returned an error.]

| Ticker | Price | P/E | Revenue Growth |
|--------|-------|-----|----------------|
[For each entry in the "comparison" array, write one table row:]
| [ticker] | [currency symbol][price] | [pe_ratio] | [revenue_growth_yoy_pct]% |

[For each peer in the table, write one comparison sentence using this pattern:
"Compared with [PEER], {ticker} has [higher/lower/similar] revenue growth ([X]% vs [Y]%) but trades at a [cheaper/richer/similar] valuation ([{ticker} P/E] vs [PEER P/E])."
Then add one concluding sentence naming the single strongest peer and one weakest peer by overall risk/reward.]

---

## 5. Earnings History

**Next Earnings Date:** [next_earnings_date from get_earnings]

| Quarter | EPS Estimate | EPS Actual | Surprise | Verdict |
|---------|-------------|------------|----------|---------|
[For each entry in last_4_quarters, write one row:]
| [date] | $[eps_estimate] | $[eps_actual] | [surprise_pct]% | [beat_miss — 🟢 beat / 🔴 miss / ➖ in-line] |

[One sentence on earnings track record: consistent beater, mixed, or serial misser.]

---

## 6. Recent News & Sentiment

[For each news article from analyze_news_sentiment, write one bullet:]
- [🟢 bullish / 🔴 bearish / ⚪ neutral] **[key_event]** — [one-line summary]

**Overall Sentiment:** [bullish/bearish/neutral] — [One sentence justification.]

---

## 7. 10-K Insights

### Strategic Highlights
[From the second query_10k call (strategic priorities). Write 3 bullets. Each bullet MUST include a near-verbatim quote from the 10-K followed by the source section in italics. Format:
- "[Exact or near-exact phrase from 10-K answer]" *(Source: 10-K, [Business Overview / MD&A / Strategy])*
If query_10k was unavailable, write 3 bullets from web_search with *(Source: web)* instead.]

### Key Risk Factors
[From the first query_10k call (risk factors). List exactly 5 risks. Each risk MUST follow this format:
**[Risk Name]** — [One-sentence description]. *(Source: 10-K, Risk Factors)*
If a risk came from web_search instead, write *(Source: web)* at the end.]

### Risk Reality Check
[For each of the 5 risks above, write exactly one sentence that cross-references the risk against live data from get_competitor_analysis, get_financial_metrics, get_earnings, or get_technical_analysis. Use this format:
**[Risk Name]:** [What the 10-K warns] — [what current data shows about whether this risk is already materializing or not].
Example: "**Competition:** NVIDIA identifies competition as a major risk — however, current revenue growth of 85.2% substantially outpaces AMD (37.8%) and QCOM (-3.5%), suggesting competitive pressure has not yet materially impacted growth."
If no relevant live data exists for a risk, write: "No current data available to assess this risk."]

---

## 8. Analyst Verdict

Score each factor 0–20 using the rubric below, then sum for a Final Score out of 100.
If any factor's input data is null, None, or N/A, assign the minimum score (5) for that factor.

**Scoring Rubric:**
- **Fundamentals (0–20):** revenue_growth_yoy_pct ≥20% → 20 | 10–19% → 15 | 0–9% → 10 | negative → 5. Deduct 3 if debt_to_equity > 100.
- **Valuation (0–20):** PEG ≤1 → 20 | 1–2 → 15 | 2–3 → 10 | >3 or N/A → 5. Deduct 3 if EV/EBITDA > 30.
- **Technical (0–20):** trend=bullish → 20 | neutral → 10 | bearish → 4. Add 2 if RSI < 35 (oversold bounce potential). Subtract 2 if RSI > 70 (overbought).
- **Earnings (0–20):** all 4 quarters beat → 20 | 3 beat → 15 | 2 beat → 10 | ≤1 beat → 5.
- **Sentiment (0–20):** majority bullish news → 20 | mixed → 12 | majority bearish → 4.

**Final Score → Recommendation:**
- 80–100 → Strong Buy
- 65–79 → Buy
- 45–64 → Hold
- 30–44 → Watch
- 0–29 → Avoid

| Factor | Signal | Score /20 |
|--------|--------|-----------|
| Fundamentals | [e.g. Revenue +18%, D/E 30 → fair] | [X] |
| Valuation | [e.g. PEG 1.15, EV/EBITDA 15 → fair] | [X] |
| Technical | [e.g. Bearish trend, RSI 40] | [X] |
| Earnings | [e.g. 4/4 beat, avg surprise +7.9%] | [X] |
| Sentiment | [e.g. 1 bullish, 4 neutral] | [X] |
| **Final Score** | | **[sum]/100** |

| | |
|---|---|
| **Recommendation** | [Strong Buy / Buy / Hold / Watch / Avoid] |
| **Key Opportunity** | [one concrete line backed by a data point above] |
| **Key Risk** | [top risk from section 7] |

**Verdict Summary:** [Write 2–3 sentences synthesizing all five factors into a professional analyst narrative. Mention the strongest factor by name, acknowledge any weak factor, and end with the recommendation and why it is justified. Do not use bullet points — write flowing prose. Example: "NVIDIA demonstrates exceptional revenue growth and consistent earnings performance, while its PEG ratio of 0.59 signals attractive valuation relative to its growth profile. Although technical indicators remain mixed and recent news flow is limited, the company's fundamental strength and earnings track record outweigh these concerns. This supports a Strong Buy recommendation, contingent on continued execution in the next earnings cycle."]

> **Conflict Analysis (required):** If any two factors point in opposite directions (e.g. Technical=bearish but Earnings=strong), you MUST explicitly name the conflict and resolve it. Use this structure:
> "Although [bearish signal], [bullish signals] outweigh the weakness because [concrete reason from the data]. The recommendation would change to [higher/lower] if [specific condition — e.g. RSI drops below 35, next earnings misses, PEG exceeds 2.5]."
> If all factors agree, write one sentence confirming the consensus and the single biggest risk to the thesis.

---

## 9. Sources

[Bullet list of all URLs returned by web_search, plus:]
- Stock & technical data: Yahoo Finance via yfinance
- Sentiment: Yahoo Finance RSS + GPT-4 analysis
[If query_10k was used: - 10-K: SEC EDGAR via Qdrant RAG]
"""
