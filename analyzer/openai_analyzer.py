import json
from schemas import NewsAnalysis
from providers.factory import get_provider

_SYSTEM = (
    "You are a financial analyst. Analyze the news and return structured JSON.\n\n"
    "Urgency criteria:\n"
    "- high: earnings miss/beat + stock move >5%, major layoffs (>10% workforce), "
    "regulatory action, M&A announcement\n"
    "- medium: product launch, executive change, analyst upgrade/downgrade, "
    "earnings roughly in-line\n"
    "- low: general market commentary, speculative articles, minor news"
)

_FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Ticker: NVDA\n"
            "Title: Nvidia Reports Record Revenue, Stock Jumps 8%\n"
            "Summary: Nvidia posted Q3 earnings of $18.1B, beating analyst expectations "
            "by 12%. Data center revenue surged 200% YoY."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "ticker": "NVDA",
            "sentiment": "bullish",
            "urgency": "high",
            "key_event": "earnings_beat",
            "summary": "Nvidia Q3 revenue $18.1B, beat by 12%. Data center up 200% YoY. Stock +8%.",
        }),
    },
    {
        "role": "user",
        "content": (
            "Ticker: META\n"
            "Title: Meta Announces 11,000 Layoffs Amid Cost-Cutting Push\n"
            "Summary: Meta will lay off 13% of its workforce as CEO Mark Zuckerberg "
            "calls it a year of efficiency."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "ticker": "META",
            "sentiment": "bearish",
            "urgency": "high",
            "key_event": "layoffs",
            "summary": "Meta cuts 11,000 jobs (13% of workforce) in major restructuring.",
        }),
    },
    {
        "role": "user",
        "content": (
            "Ticker: AAPL\n"
            "Title: Apple Analyst Raises Price Target to $220\n"
            "Summary: Wedbush analyst Dan Ives raised his Apple price target from $200 "
            "to $220, citing strong iPhone 15 demand."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "ticker": "AAPL",
            "sentiment": "bullish",
            "urgency": "medium",
            "key_event": "analyst_upgrade",
            "summary": "Wedbush raises AAPL target to $220 on strong iPhone 15 demand.",
        }),
    },
]


def analyze_news(title: str, summary: str, ticker: str) -> NewsAnalysis:
    provider = get_provider("openai", "gpt-4.1-mini")
    messages = (
        [{"role": "system", "content": _SYSTEM}]
        + _FEW_SHOT
        + [{"role": "user", "content": f"Ticker: {ticker}\nTitle: {title}\nSummary: {summary}"}]
    )
    return provider.generate_structured(messages=messages, schema=NewsAnalysis)
