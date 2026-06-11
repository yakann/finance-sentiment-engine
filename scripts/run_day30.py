"""Day 30 — Golden Dataset for Sentiment Evaluation

50 finans haberi başlığı (25 clear + 25 edge case) elle etiketlendi.
LangSmith'e 'finance-sentiment-v1' dataset olarak yüklenir.

Kullanım:
    python scripts/run_day30.py

Ön koşul: .env dosyasında şunlar dolu olmalı:
    LANGSMITH_API_KEY=lsv2_...
    LANGSMITH_PROJECT=finance-agent
    LANGCHAIN_TRACING_V2=true
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

# ── Golden Dataset ─────────────────────────────────────────────────────────────
# Her örnek: inputs (ticker, title, summary) + outputs (expected_sentiment,
# expected_urgency, expected_key_event).
#
# expected_sentiment: "bullish" | "bearish" | "neutral"
# expected_urgency:   "high" | "medium" | "low"
# expected_key_event: schemas.KeyEvent literal

GOLDEN_EXAMPLES: list[dict] = [
    # ── CLEAR BULLISH (10) ─────────────────────────────────────────────────────
    {
        "inputs": {
            "ticker": "NVDA",
            "title": "Nvidia Reports Record Q4 Revenue of $22B, Blowing Past Estimates",
            "summary": "Nvidia Q4 2024 revenue hit $22B, 8% above the $20.4B consensus. "
                       "Data center segment doubled YoY. EPS of $5.16 beat by $0.52. "
                       "CEO Jensen Huang called it 'an exceptional quarter.'",
        },
        "outputs": {
            "expected_sentiment": "bullish",
            "expected_urgency": "high",
            "expected_key_event": "earnings",
        },
    },
    {
        "inputs": {
            "ticker": "MSFT",
            "title": "Morgan Stanley Upgrades Microsoft to Overweight, Raises Target to $500",
            "summary": "Morgan Stanley upgraded Microsoft from Equal Weight to Overweight, "
                       "lifting its 12-month price target from $420 to $500, citing accelerating "
                       "Azure AI workloads and Copilot subscription momentum.",
        },
        "outputs": {
            "expected_sentiment": "bullish",
            "expected_urgency": "medium",
            "expected_key_event": "analyst_action",
        },
    },
    {
        "inputs": {
            "ticker": "AMZN",
            "title": "Amazon Prime Day 2024 Breaks Records with $14B in Global Sales",
            "summary": "Amazon's two-day Prime Day event generated an estimated $14.2B in "
                       "global sales, up 11% from 2023. Third-party seller GMV hit a record "
                       "$7.5B. Stock closed up 2.3% on the news.",
        },
        "outputs": {
            "expected_sentiment": "bullish",
            "expected_urgency": "high",
            "expected_key_event": "product_event",
        },
    },
    {
        "inputs": {
            "ticker": "AAPL",
            "title": "Apple Unveils iPhone 16 with On-Device AI; Analysts See Once-in-a-Decade Upgrade Cycle",
            "summary": "Apple introduced iPhone 16 with Apple Intelligence, a suite of on-device "
                       "AI features. Wedbush analyst Dan Ives called it a 'once-in-a-decade upgrade "
                       "supercycle,' estimating 300M potential upgraders.",
        },
        "outputs": {
            "expected_sentiment": "bullish",
            "expected_urgency": "medium",
            "expected_key_event": "product_event",
        },
    },
    {
        "inputs": {
            "ticker": "WMT",
            "title": "Walmart Raises Full-Year Guidance After Beating Q2 EPS and Revenue",
            "summary": "Walmart Q2 EPS of $0.67 beat the $0.65 estimate. Revenue rose 4.8% to "
                       "$169.3B. Management raised FY2025 net sales growth guidance to 3.75–4.75% "
                       "from 3–4%. Comparable-store sales grew 4.2% in the US.",
        },
        "outputs": {
            "expected_sentiment": "bullish",
            "expected_urgency": "high",
            "expected_key_event": "earnings",
        },
    },
    {
        "inputs": {
            "ticker": "DIS",
            "title": "Disney+ Streaming Segment Reaches Profitability for First Time, Up $300M",
            "summary": "Disney reported its combined streaming business (Disney+, Hulu, ESPN+) "
                       "generated $300M in operating income in Q4, turning profitable for the "
                       "first time. Subscriber base grew to 153M globally.",
        },
        "outputs": {
            "expected_sentiment": "bullish",
            "expected_urgency": "high",
            "expected_key_event": "earnings",
        },
    },
    {
        "inputs": {
            "ticker": "GOOGL",
            "title": "Alphabet Q3 Earnings Beat on Strong Ad Revenue Recovery, Stock Rises 6%",
            "summary": "Alphabet posted Q3 EPS of $1.55, beating the $1.45 consensus. Google "
                       "Search revenue grew 12% YoY, recovering faster than peers. YouTube ads "
                       "rose 18%. Cloud segment grew 28%, topping $10B for the first time.",
        },
        "outputs": {
            "expected_sentiment": "bullish",
            "expected_urgency": "high",
            "expected_key_event": "earnings",
        },
    },
    {
        "inputs": {
            "ticker": "META",
            "title": "Meta Reports Best Quarter Since 2021 with Revenue Up 27% Year-Over-Year",
            "summary": "Meta Q2 revenue hit $39.1B, up 27% YoY, crushing the $38.3B estimate. "
                       "EPS of $5.16 beat by $0.49. Daily active people reached 3.27B. "
                       "CFO guided Q3 revenue of $38.5–41B, above the $39B consensus.",
        },
        "outputs": {
            "expected_sentiment": "bullish",
            "expected_urgency": "high",
            "expected_key_event": "earnings",
        },
    },
    {
        "inputs": {
            "ticker": "TSLA",
            "title": "Tesla Reports Record 484,000 Vehicle Deliveries in Q4, Beats Estimates",
            "summary": "Tesla delivered 484,507 vehicles in Q4 2023, beating the 476,000 analyst "
                       "consensus. Production reached 494,989 units. Cybertruck production "
                       "ramped to 2,242 units. Full-year deliveries hit 1.81M, up 38%.",
        },
        "outputs": {
            "expected_sentiment": "bullish",
            "expected_urgency": "high",
            "expected_key_event": "earnings",
        },
    },
    {
        "inputs": {
            "ticker": "COIN",
            "title": "Coinbase Shares Surge 28% as Bitcoin Breaks $100,000 for the First Time",
            "summary": "Bitcoin crossed $100,000 for the first time, lifting Coinbase shares 28% "
                       "to $340 in a single session. Trading volumes spiked to 4x the 30-day "
                       "average. Analysts raised targets, citing revenue leverage to crypto prices.",
        },
        "outputs": {
            "expected_sentiment": "bullish",
            "expected_urgency": "high",
            "expected_key_event": "market_dynamics",
        },
    },

    # ── CLEAR BEARISH (10) ─────────────────────────────────────────────────────
    {
        "inputs": {
            "ticker": "AMZN",
            "title": "Amazon Misses Q3 Revenue Estimates by $2B; Guidance Disappoints Wall Street",
            "summary": "Amazon Q3 revenue of $127.1B came in $2B below the $129B consensus. "
                       "AWS growth decelerated to 12%. Q4 guidance of $140–148B midpoint "
                       "disappointed vs. the $155B Street estimate. Shares fell 9%.",
        },
        "outputs": {
            "expected_sentiment": "bearish",
            "expected_urgency": "high",
            "expected_key_event": "earnings",
        },
    },
    {
        "inputs": {
            "ticker": "TSLA",
            "title": "Tesla Q3 Gross Margin Collapses to 17.9%, Worst Level in Three Years",
            "summary": "Tesla's Q3 automotive gross margin fell to 17.9%, down from 25.1% a year "
                       "earlier, as aggressive price cuts eroded profitability. Net income fell "
                       "44% YoY. CEO Elon Musk warned of 'difficult times' ahead.",
        },
        "outputs": {
            "expected_sentiment": "bearish",
            "expected_urgency": "high",
            "expected_key_event": "earnings",
        },
    },
    {
        "inputs": {
            "ticker": "F",
            "title": "Ford Recalls 650,000 Mustang Mach-E Vehicles Over Steering Defect",
            "summary": "Ford issued a safety recall for 650,000 Mustang Mach-E EVs due to a "
                       "potential loss-of-steering control defect. NHTSA opened a probe. "
                       "Repair costs estimated at $800M. Shares fell 4.2% on the announcement.",
        },
        "outputs": {
            "expected_sentiment": "bearish",
            "expected_urgency": "high",
            "expected_key_event": "product_event",
        },
    },
    {
        "inputs": {
            "ticker": "NVDA",
            "title": "US Expands AI Chip Export Restrictions, Blocking Nvidia H100 and A100 Sales to China",
            "summary": "The Biden administration tightened export controls, adding Nvidia's H100 "
                       "and A100 chips to the restricted list for China sales. Analysts estimate "
                       "this could cost Nvidia $5B in annual revenue. Shares dropped 7%.",
        },
        "outputs": {
            "expected_sentiment": "bearish",
            "expected_urgency": "high",
            "expected_key_event": "policy_geopolitical",
        },
    },
    {
        "inputs": {
            "ticker": "META",
            "title": "FTC Files Federal Antitrust Lawsuit Seeking to Break Up Meta's Instagram and WhatsApp",
            "summary": "The FTC filed suit to unwind Meta's acquisitions of Instagram and WhatsApp, "
                       "arguing they eliminated competitive threats. The case could force Meta to "
                       "divest both platforms. Legal costs and uncertainty sent shares down 5%.",
        },
        "outputs": {
            "expected_sentiment": "bearish",
            "expected_urgency": "high",
            "expected_key_event": "policy_geopolitical",
        },
    },
    {
        "inputs": {
            "ticker": "AAPL",
            "title": "Warren Buffett Discloses Berkshire Hathaway Has Cut Apple Stake by 49%",
            "summary": "Berkshire Hathaway's 13-F filing showed it sold approximately 389M Apple "
                       "shares in Q2, reducing its stake by 49% to about 400M shares. Buffett "
                       "cited tax considerations, but the scale of selling alarmed markets.",
        },
        "outputs": {
            "expected_sentiment": "bearish",
            "expected_urgency": "high",
            "expected_key_event": "insider_activity",
        },
    },
    {
        "inputs": {
            "ticker": "BA",
            "title": "Boeing Faces Federal Criminal Probe After Second 737 Max Door Plug Incident",
            "summary": "The DOJ reopened a criminal investigation into Boeing after an Alaska "
                       "Airlines 737 Max 9 door plug blew out mid-flight. The FAA capped 737 Max "
                       "production. CEO Dave Calhoun faced calls to resign. Shares fell 8%.",
        },
        "outputs": {
            "expected_sentiment": "bearish",
            "expected_urgency": "high",
            "expected_key_event": "policy_geopolitical",
        },
    },
    {
        "inputs": {
            "ticker": "PYPL",
            "title": "PayPal Loses eBay Checkout Partnership; $1.5B Annual Revenue at Risk",
            "summary": "eBay officially completed its transition away from PayPal to managed "
                       "payments, ending a 15-year partnership. Analysts estimated the loss at "
                       "$1.5B in annual revenue. PayPal shares fell 6% on the formal confirmation.",
        },
        "outputs": {
            "expected_sentiment": "bearish",
            "expected_urgency": "high",
            "expected_key_event": "business_action",
        },
    },
    {
        "inputs": {
            "ticker": "INTC",
            "title": "Intel Reports $1.6B Net Loss, Cuts Quarterly Dividend and Announces 15,000 Layoffs",
            "summary": "Intel posted a Q2 net loss of $1.6B vs. expected profit of $100M. Revenue "
                       "fell 1% to $12.8B, missing estimates. Management suspended the dividend "
                       "and announced a restructuring cutting 15% of the global workforce.",
        },
        "outputs": {
            "expected_sentiment": "bearish",
            "expected_urgency": "high",
            "expected_key_event": "earnings",
        },
    },
    {
        "inputs": {
            "ticker": "GM",
            "title": "General Motors Delays Ultium EV Platform Rollout by Two Years Amid Weak Demand",
            "summary": "GM pushed back its next-generation Ultium EV platform rollout from 2025 "
                       "to 2027, citing soft consumer EV demand and battery cost challenges. "
                       "The company also cut EV production targets by 30% for 2024.",
        },
        "outputs": {
            "expected_sentiment": "bearish",
            "expected_urgency": "medium",
            "expected_key_event": "business_action",
        },
    },

    # ── CLEAR NEUTRAL (5) ─────────────────────────────────────────────────────
    {
        "inputs": {
            "ticker": "JNJ",
            "title": "Johnson & Johnson Q2 Earnings In Line With Consensus Estimates",
            "summary": "J&J reported Q2 EPS of $2.82 vs. the $2.83 consensus. Revenue of $22.4B "
                       "matched the $22.3B estimate. Management reiterated full-year guidance. "
                       "Pharmaceutical segment grew 6.5%, MedTech grew 5.8%.",
        },
        "outputs": {
            "expected_sentiment": "neutral",
            "expected_urgency": "medium",
            "expected_key_event": "earnings",
        },
    },
    {
        "inputs": {
            "ticker": "XOM",
            "title": "ExxonMobil Reiterates $25B Full-Year Capital Expenditure Guidance",
            "summary": "ExxonMobil's investor day presentation reiterated its FY2024 capex plan "
                       "of $23–25B, unchanged from prior guidance. No new projects announced. "
                       "Management provided standard operational efficiency updates.",
        },
        "outputs": {
            "expected_sentiment": "neutral",
            "expected_urgency": "low",
            "expected_key_event": "company_communication",
        },
    },
    {
        "inputs": {
            "ticker": "T",
            "title": "AT&T Declares Regular Quarterly Dividend of $0.2775 Per Share",
            "summary": "AT&T's board declared its regular quarterly cash dividend of $0.2775 "
                       "per share, payable August 1 to shareholders of record as of July 10. "
                       "No change from the prior quarter's dividend amount.",
        },
        "outputs": {
            "expected_sentiment": "neutral",
            "expected_urgency": "low",
            "expected_key_event": "business_action",
        },
    },
    {
        "inputs": {
            "ticker": "WBA",
            "title": "Walgreens Provides Update on Ongoing Store Closure and Restructuring Program",
            "summary": "Walgreens confirmed it closed 150 of the previously announced 300 "
                       "underperforming stores under its multi-year restructuring plan. The "
                       "company said the program remains on schedule and within budget estimates.",
        },
        "outputs": {
            "expected_sentiment": "neutral",
            "expected_urgency": "medium",
            "expected_key_event": "business_action",
        },
    },
    {
        "inputs": {
            "ticker": "GS",
            "title": "Goldman Sachs Economists Note Mixed Signals in Q3 GDP Data",
            "summary": "Goldman Sachs research published a note calling Q3 GDP data 'mixed,' "
                       "with consumer spending resilient but business investment softening. "
                       "The team maintained its 2024 GDP growth forecast of 2.1%.",
        },
        "outputs": {
            "expected_sentiment": "neutral",
            "expected_urgency": "low",
            "expected_key_event": "other",
        },
    },

    # ── EDGE CASES (25) ────────────────────────────────────────────────────────
    # Edge 1: CEO resignation / transition → market rejoices (ironic bullish)
    {
        "inputs": {
            "ticker": "NFLX",
            "title": "Netflix Stock Surges 12% After Co-CEO Reed Hastings Steps Down, Transitions to Chairman",
            "summary": "Netflix co-founder Reed Hastings announced he would step down as co-CEO "
                       "and transition to Executive Chairman. Greg Peters takes the co-CEO role "
                       "with Ted Sarandos. Wall Street cheered the management clarity, sending "
                       "shares up 12% — the surface read of 'CEO quits' is misleading.",
        },
        "outputs": {
            "expected_sentiment": "bullish",
            "expected_urgency": "medium",
            "expected_key_event": "company_communication",
        },
    },
    # Edge 2: Beat current quarter but cautious forward guidance (mixed)
    {
        "inputs": {
            "ticker": "MSFT",
            "title": "Microsoft Beats Q4 Estimates but Issues Below-Consensus Q1 Revenue Guidance",
            "summary": "Microsoft Q4 EPS of $3.30 beat the $3.10 estimate. Revenue of $64.7B "
                       "topped $64.3B consensus. However, Q1 FY2025 guidance of $63.8–64.8B "
                       "midpoint came in below the $65.1B analyst consensus, weighing on the stock.",
        },
        "outputs": {
            "expected_sentiment": "neutral",
            "expected_urgency": "high",
            "expected_key_event": "earnings",
        },
    },
    # Edge 3: Layoffs cheered as cost discipline (ironic bullish)
    {
        "inputs": {
            "ticker": "META",
            "title": "Meta Announces 11,000 Layoffs; Wall Street Cheers 'Year of Efficiency' Cost Discipline",
            "summary": "Meta cut 13% of its workforce (11,000 employees). CEO Zuckerberg called "
                       "it a 'year of efficiency.' Analysts immediately raised EPS estimates by "
                       "10–15% on the cost savings. Stock rose 4% despite the human impact. "
                       "The layoff read as bullish to the market.",
        },
        "outputs": {
            "expected_sentiment": "bullish",
            "expected_urgency": "high",
            "expected_key_event": "business_action",
        },
    },
    # Edge 4: Acquisition blocked by regulator — acquirer stock rises (deal collapse = relief)
    {
        "inputs": {
            "ticker": "ADBE",
            "title": "Adobe Terminates $20B Figma Acquisition After Regulatory Block; Shares Rise on Capital Return Hope",
            "summary": "Adobe called off its $20B Figma acquisition after the UK CMA blocked it. "
                       "Adobe paid Figma a $1B termination fee. Surprisingly, ADBE shares rose 3% "
                       "as investors preferred share buybacks to the expensive, dilutive acquisition.",
        },
        "outputs": {
            "expected_sentiment": "bullish",
            "expected_urgency": "high",
            "expected_key_event": "policy_geopolitical",
        },
    },
    # Edge 5: Miss + insider buy (contradictory signals)
    {
        "inputs": {
            "ticker": "TSLA",
            "title": "Tesla Misses Q3 Revenue Estimates; Elon Musk Personally Buys $20M in Shares",
            "summary": "Tesla Q3 revenue of $23.4B missed the $24.1B estimate by 3%. Automotive "
                       "gross margin fell to 18.7%. However, Elon Musk disclosed a personal "
                       "purchase of $20M in Tesla shares, his first open-market buy in two years.",
        },
        "outputs": {
            "expected_sentiment": "neutral",
            "expected_urgency": "high",
            "expected_key_event": "earnings",
        },
    },
    # Edge 6: Bland headline hiding a deceleration story (deceptively bearish phrasing)
    {
        "inputs": {
            "ticker": "AAPL",
            "title": "Apple Revenue Growth Moderates to 2% for Third Consecutive Quarter",
            "summary": "Apple reported revenue of $89.5B, up just 2% YoY, matching the consensus. "
                       "iPhone revenue was flat. Services grew 16% but couldn't offset hardware "
                       "weakness. China revenue fell 8% YoY for the second consecutive quarter.",
        },
        "outputs": {
            "expected_sentiment": "bearish",
            "expected_urgency": "medium",
            "expected_key_event": "earnings",
        },
    },
    # Edge 7: Debt swap avoids bankruptcy — relief rally but still distressed (neutral)
    {
        "inputs": {
            "ticker": "AMC",
            "title": "AMC Entertainment Completes Debt-for-Equity Swap, Narrowly Avoids Chapter 11",
            "summary": "AMC converted $2.4B in debt to equity, averting an imminent bankruptcy "
                       "filing. The deal dilutes existing shareholders by 60% but eliminates "
                       "near-term default risk. Analysts called it 'avoiding the worst outcome "
                       "while creating a new set of problems.'",
        },
        "outputs": {
            "expected_sentiment": "neutral",
            "expected_urgency": "high",
            "expected_key_event": "business_action",
        },
    },
    # Edge 8: Tariff hits domestic brand's manufacturing (bearish despite nationalistic framing)
    {
        "inputs": {
            "ticker": "AAPL",
            "title": "Trump Administration Imposes 25% Tariff on Chinese Electronics; Apple Seeks Exemption",
            "summary": "The White House announced 25% tariffs on Chinese-made consumer electronics. "
                       "Apple, which manufactures 85% of iPhones in China, immediately lobbied for "
                       "an exemption. Analysts estimated tariffs could add $100–150 to iPhone "
                       "production costs if no exemption is granted.",
        },
        "outputs": {
            "expected_sentiment": "bearish",
            "expected_urgency": "high",
            "expected_key_event": "policy_geopolitical",
        },
    },
    # Edge 9: Corporate breakup — not clearly good or bad (neutral transformation)
    {
        "inputs": {
            "ticker": "GE",
            "title": "GE Completes Three-Way Split Into GE Aerospace, GE Vernova, and GE HealthCare",
            "summary": "General Electric officially completed its split into three independent "
                       "publicly traded companies: GE Aerospace, GE Vernova (energy), and "
                       "GE HealthCare. The move ends the 130-year-old conglomerate. "
                       "Analysts are split on whether the sum of parts exceeds the whole.",
        },
        "outputs": {
            "expected_sentiment": "neutral",
            "expected_urgency": "medium",
            "expected_key_event": "business_action",
        },
    },
    # Edge 10: Legal settlement framed as good news (legal resolution = bullish)
    {
        "inputs": {
            "ticker": "GOOG",
            "title": "Google Agrees to $5B Privacy Settlement; Analysts Say Worst-Case Liability Now Capped",
            "summary": "Alphabet agreed to settle a $5B class-action privacy lawsuit over Incognito "
                       "mode data collection. While $5B is large, analysts noted it removes an "
                       "open-ended liability overhang. Stock rose 2% on settlement certainty.",
        },
        "outputs": {
            "expected_sentiment": "bullish",
            "expected_urgency": "medium",
            "expected_key_event": "policy_geopolitical",
        },
    },
    # Edge 11: Record loss but losses narrowed / beat estimates (tricky neutral)
    {
        "inputs": {
            "ticker": "LYFT",
            "title": "Lyft Reports Record $1.2B Annual Loss, But Loss Narrowed 35% Versus Prior Year",
            "summary": "Lyft posted a full-year net loss of $1.2B, its largest ever in absolute "
                       "terms. However, the loss narrowed 35% from the prior year's $1.85B. "
                       "Adjusted EBITDA turned positive at $340M. Analysts focused on the "
                       "trajectory toward profitability.",
        },
        "outputs": {
            "expected_sentiment": "neutral",
            "expected_urgency": "medium",
            "expected_key_event": "earnings",
        },
    },
    # Edge 12: Recall AND record deliveries in the same headline (opposing signals)
    {
        "inputs": {
            "ticker": "TSLA",
            "title": "Tesla Recalls 200,000 Vehicles Over Autopilot Bug While Recording Its Best Delivery Quarter",
            "summary": "Tesla issued a software recall for 200,000 vehicles over an Autopilot "
                       "detection failure, the same week it announced record quarterly deliveries "
                       "of 484,000 vehicles. NHTSA is monitoring the recall. The opposing signals "
                       "left analysts uncertain about the net sentiment.",
        },
        "outputs": {
            "expected_sentiment": "neutral",
            "expected_urgency": "high",
            "expected_key_event": "product_event",
        },
    },
    # Edge 13: Short seller attack that the market ultimately rejects (stock ends higher)
    {
        "inputs": {
            "ticker": "HIMS",
            "title": "Hindenburg Research Short Report Sends HIMS Stock Down 15%; Shares Recover to Close 7% Higher",
            "summary": "Hindenburg Research published a short-seller report alleging HIMS used "
                       "misleading subscriber metrics. Shares dropped 15% at open. After management "
                       "rebutted key claims in an investor call, shares reversed to close 7% above "
                       "the prior day's close. Short interest spiked but the market rejected the thesis.",
        },
        "outputs": {
            "expected_sentiment": "neutral",
            "expected_urgency": "medium",
            "expected_key_event": "market_dynamics",
        },
    },
    # Edge 14: False M&A rumor → stock pops then reverses on denial (net bearish once corrected)
    {
        "inputs": {
            "ticker": "AMZN",
            "title": "Report of Amazon Acquiring TikTok Sends Shares Up 5%; Amazon Denies Any Discussions",
            "summary": "A tweet claiming Amazon was in advanced talks to acquire TikTok's US "
                       "operations sent AMZN shares up 5% in minutes. Amazon's PR team issued "
                       "a flat denial 40 minutes later. The stock retraced the entire gain and "
                       "closed down 1%. No fundamental basis to the original report.",
        },
        "outputs": {
            "expected_sentiment": "bearish",
            "expected_urgency": "low",
            "expected_key_event": "other",
        },
    },
    # Edge 15: Current quarter miss + forward guidance beat (classic mixed signal)
    {
        "inputs": {
            "ticker": "SNAP",
            "title": "Snap Misses Q2 Revenue Estimates by 4% but Issues Above-Consensus Q3 Guidance",
            "summary": "Snap Q2 revenue of $1.24B missed the $1.29B estimate. However, Q3 "
                       "revenue guidance of $1.33–1.37B midpoint beat the $1.31B consensus, "
                       "suggesting recovery. Daily active users grew 10% to 432M. "
                       "Shares swung between -8% and +6% during extended trading.",
        },
        "outputs": {
            "expected_sentiment": "neutral",
            "expected_urgency": "medium",
            "expected_key_event": "earnings",
        },
    },
    # Edge 16: Regulatory win + macro concern in same headline (neutral net)
    {
        "inputs": {
            "ticker": "NVDA",
            "title": "Nvidia Wins EU Clearance for ARM Acquisition Attempt While Analysts Flag AI Capex Slowdown Risk",
            "summary": "The EU cleared Nvidia's attempted acquisition of ARM, removing a major "
                       "regulatory hurdle. However, in the same week, three sell-side analysts "
                       "published notes warning that hyperscaler AI capex growth may peak in "
                       "2025, creating a ceiling for GPU demand beyond current backlog.",
        },
        "outputs": {
            "expected_sentiment": "neutral",
            "expected_urgency": "medium",
            "expected_key_event": "policy_geopolitical",
        },
    },
    # Edge 17: 'Better than feared' framing of declining results (neutral despite negative numbers)
    {
        "inputs": {
            "ticker": "INTC",
            "title": "Intel Q3 Revenue Decline Narrows to 8%, Far Better Than the 14% Feared by Analysts",
            "summary": "Intel reported Q3 revenue down 8% YoY — below zero, but much better than "
                       "the 14% decline analysts had penciled in after profit warnings. Shares "
                       "rose 9% on the 'less bad than expected' result, a classic relief rally "
                       "despite fundamentally negative results.",
        },
        "outputs": {
            "expected_sentiment": "neutral",
            "expected_urgency": "medium",
            "expected_key_event": "earnings",
        },
    },
    # Edge 18: Fed policy hurts growth stocks (macro bearish for tech)
    {
        "inputs": {
            "ticker": "AMZN",
            "title": "Fed Signals Rates Staying Higher for Longer; Amazon CFO Warns of Enterprise Cloud Cost Optimization",
            "summary": "The Fed's dot plot showed rates staying above 5% through 2025. Amazon's "
                       "CFO warned that enterprise customers are 'optimizing cloud spend' — "
                       "a phrase that historically precedes AWS deceleration. Both signals are "
                       "negative for Amazon's premium growth multiple.",
        },
        "outputs": {
            "expected_sentiment": "bearish",
            "expected_urgency": "medium",
            "expected_key_event": "policy_geopolitical",
        },
    },
    # Edge 19: Workforce cuts cheered — Spotify efficiency (ironic bullish)
    {
        "inputs": {
            "ticker": "SPOT",
            "title": "Spotify Cuts 17% of Global Workforce; Shares Jump 8% as Analysts Applaud Efficiency",
            "summary": "Spotify announced its third round of layoffs in a year, cutting 17% of "
                       "staff (1,500 employees). CEO Daniel Ek cited 'overhiring.' Shares rose 8% "
                       "as analysts raised EPS estimates by 20%, projecting Spotify's path to "
                       "profitability accelerated by 18 months. The headline reads grim; the "
                       "market reaction was decidedly bullish.",
        },
        "outputs": {
            "expected_sentiment": "bullish",
            "expected_urgency": "high",
            "expected_key_event": "business_action",
        },
    },
    # Edge 20: Beat subs, miss ARPU (good user growth, weak monetization)
    {
        "inputs": {
            "ticker": "NFLX",
            "title": "Netflix Adds 9.3M Subscribers Beating Estimates but Average Revenue Per Member Falls Short",
            "summary": "Netflix Q2 net subscriber additions of 9.3M beat the 8.2M estimate. "
                       "However, average revenue per membership of $16.62 missed the $16.80 "
                       "estimate. The ad-supported tier grew faster than premium, diluting ARPU. "
                       "Revenue of $9.56B matched consensus exactly.",
        },
        "outputs": {
            "expected_sentiment": "neutral",
            "expected_urgency": "medium",
            "expected_key_event": "earnings",
        },
    },
    # Edge 21: CEO retirement speculation (uncertainty = bearish)
    {
        "inputs": {
            "ticker": "AAPL",
            "title": "Analysts Speculate Tim Cook Planning 2025 Exit as Apple Shuffles C-Suite Roles",
            "summary": "Multiple media reports cited anonymous sources suggesting Apple CEO "
                       "Tim Cook may announce retirement by end-2025. Apple reshuffled three "
                       "executive roles, adding fuel to succession speculation. No official "
                       "comment from Apple. Analysts called the uncertainty 'an overhang.'",
        },
        "outputs": {
            "expected_sentiment": "bearish",
            "expected_urgency": "low",
            "expected_key_event": "company_communication",
        },
    },
    # Edge 22: Slow growth framed positively in headline (ironic bullish framing)
    {
        "inputs": {
            "ticker": "AMZN",
            "title": "Amazon Web Services Growth 'Only' 17% as Bears Had Predicted Single Digits; Stock Rallies",
            "summary": "AWS Q3 growth of 17% YoY was framed negatively by headline writers, "
                       "but bears had forecast 8–12% growth following Microsoft Azure's weak "
                       "quarter. The 'only 17%' beat expectations significantly. Shares rallied "
                       "4% as cloud spending held up better than feared.",
        },
        "outputs": {
            "expected_sentiment": "bullish",
            "expected_urgency": "medium",
            "expected_key_event": "earnings",
        },
    },
    # Edge 23: Analyst upgrade with simultaneous price target cut (contradictory)
    {
        "inputs": {
            "ticker": "TSLA",
            "title": "Morgan Stanley Upgrades Tesla to Overweight but Cuts 12-Month Price Target by $50 to $350",
            "summary": "Morgan Stanley upgraded Tesla from Equal Weight to Overweight but "
                       "simultaneously cut its 12-month price target from $400 to $350 due to "
                       "near-term margin pressure. The upgrade signals long-term conviction while "
                       "the target cut acknowledges short-term headwinds.",
        },
        "outputs": {
            "expected_sentiment": "neutral",
            "expected_urgency": "medium",
            "expected_key_event": "analyst_action",
        },
    },
    # Edge 24: ESG miss, zero market impact (fundamentals vs. non-financial metrics)
    {
        "inputs": {
            "ticker": "AAPL",
            "title": "Apple Misses Carbon Neutrality Deadline by Two Years; ESG Fund Flows Into Stock Unchanged",
            "summary": "Apple acknowledged it will not meet its 2025 carbon neutrality pledge, "
                       "pushing the deadline to 2027. Environmental advocates expressed "
                       "disappointment. However, major ESG funds maintained their Apple "
                       "allocations unchanged, citing the company's overall sustainability trajectory.",
        },
        "outputs": {
            "expected_sentiment": "neutral",
            "expected_urgency": "low",
            "expected_key_event": "policy_geopolitical",
        },
    },
    # Edge 25: 'Sell the news' — blockbuster earnings but stock falls (fundamental vs. market reaction)
    {
        "inputs": {
            "ticker": "NVDA",
            "title": "Nvidia Shatters Revenue Records at $35.1B, Tripling YoY — Stock Falls 6% on 'Priced In' Concerns",
            "summary": "Nvidia Q2 FY2025 revenue of $35.1B tripled YoY and beat the $33.8B "
                       "estimate. EPS of $0.68 beat by $0.05. Data center revenue hit $26.3B. "
                       "Yet shares fell 6% as investors had pre-positioned aggressively and "
                       "took profits. The underlying business is exceptional; the market reaction "
                       "reflects sentiment dynamics, not fundamentals.",
        },
        "outputs": {
            "expected_sentiment": "bullish",
            "expected_urgency": "high",
            "expected_key_event": "earnings",
        },
    },
]


# ── LangSmith Upload ───────────────────────────────────────────────────────────

DATASET_NAME = "finance-sentiment-v1"
DATASET_DESCRIPTION = (
    "50 finance news headlines for sentiment evaluation. "
    "25 clear-signal examples (bullish/bearish/neutral) + "
    "25 edge cases including ironic bullish, mixed signals, sell-the-news, "
    "and deceptive framing. Day 30 golden set."
)


def _check_env() -> bool:
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] LANGSMITH_API_KEY not set in .env")
        return False
    endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    print(f"[OK] LangSmith API key found — endpoint: {endpoint}")
    return True


def upload_dataset() -> None:
    from langsmith import Client

    client = Client()

    # Check if dataset already exists and remove it to allow re-upload
    existing = [d for d in client.list_datasets() if d.name == DATASET_NAME]
    if existing:
        print(f"[INFO] Dataset '{DATASET_NAME}' already exists — deleting for fresh upload...")
        client.delete_dataset(dataset_id=existing[0].id)

    print(f"[+] Creating dataset '{DATASET_NAME}'...")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=DATASET_DESCRIPTION,
    )
    print(f"    Dataset ID: {dataset.id}")

    inputs = [ex["inputs"] for ex in GOLDEN_EXAMPLES]
    outputs = [ex["outputs"] for ex in GOLDEN_EXAMPLES]

    print(f"[+] Uploading {len(GOLDEN_EXAMPLES)} examples...")
    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        dataset_id=dataset.id,
    )
    print(f"[OK] {len(GOLDEN_EXAMPLES)} examples uploaded successfully.")

    # Summary stats
    sentiments = [o["expected_sentiment"] for o in outputs]
    urgencies = [o["expected_urgency"] for o in outputs]
    print("\n── Dataset Summary ───────────────────────────────")
    for s in ("bullish", "bearish", "neutral"):
        print(f"  {s:10s}: {sentiments.count(s):2d}")
    print()
    for u in ("high", "medium", "low"):
        print(f"  {u:10s}: {urgencies.count(u):2d}")
    print("──────────────────────────────────────────────────")

    endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://smith.langchain.com")
    base = endpoint.replace("/api", "").rstrip("/")
    # EU endpoint
    if "eu.api" in base:
        base = "https://eu.smith.langchain.com"
    print(f"\nDataset live at: {base}/datasets/{dataset.id}")


def main() -> None:
    print("=" * 60)
    print("Day 30 — Golden Dataset Upload")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Examples: {len(GOLDEN_EXAMPLES)} (25 clear + 25 edge cases)")
    print("=" * 60)

    if not _check_env():
        sys.exit(1)

    upload_dataset()
    print("\nDone. Run the evaluator against this dataset to measure model accuracy.")


if __name__ == "__main__":
    main()
