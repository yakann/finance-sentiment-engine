from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from schemas import NewsAnalysis


class FinanceState(TypedDict, total=False):
    ticker: str
    news: list[NewsAnalysis]
    sentiment_summary: str
    price_data: dict
    risks: list[str]
    draft: str
    messages: Annotated[list[BaseMessage], add_messages]
