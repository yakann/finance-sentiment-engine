from typing import Literal
from pydantic import BaseModel, Field

# 10 consolidated event buckets. Keep this list stable — models are prompted
# with a 1-line definition for each so they can pick the right bucket.
KeyEvent = Literal[
    "earnings",             # Earnings previews, results, expectations
    "analyst_action",       # Analyst target changes, predictions, commentary
    "company_communication",# CEO/exec statements, internal product narratives
    "product_event",        # Launches, safety incidents, recalls, concrete product news
    "competitive_dynamics", # Competitor moves, market share shifts, customer churn
    "policy_geopolitical",  # Regulation, trade, government action, public sentiment
    "business_action",      # M&A, partnerships, pricing changes, supply chain
    "market_dynamics",      # Market cap milestones, sector moves, trading patterns
    "insider_activity",     # Insider/institutional buys, sells, speculation
    "other",                # Comparative analysis, mixed signals, off-topic
]


class NewsAnalysis(BaseModel):
    ticker: str
    sentiment: Literal["bullish", "bearish", "neutral"]
    urgency: Literal["high", "medium", "low"]
    key_event: KeyEvent
    summary: str = Field(max_length=200)
