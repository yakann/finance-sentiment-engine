from typing import Literal
from pydantic import BaseModel, Field


class NewsAnalysis(BaseModel):
    ticker: str
    sentiment: Literal["bullish", "bearish", "neutral"]
    urgency: Literal["high", "medium", "low"]
    key_event: str
    summary: str = Field(max_length=200)
