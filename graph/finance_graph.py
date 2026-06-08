from langgraph.graph import END, START, StateGraph

from agent.tools.finance import _GetStockDataInput, _get_stock_data
from agent.tools.sentiment import _AnalyzeNewsSentimentInput, _analyze_news_sentiment
from graph.state import FinanceState
from schemas import NewsAnalysis


def collect_news(state: FinanceState) -> dict:
    ticker = state["ticker"]
    raw = _analyze_news_sentiment(_AnalyzeNewsSentimentInput(ticker=ticker, top_n=5))
    news = [NewsAnalysis(**item) for item in raw if "error" not in item]
    return {"news": news}


def analyze_sentiment(state: FinanceState) -> dict:
    news: list[NewsAnalysis] = state.get("news", [])
    if not news:
        return {"sentiment_summary": "No news available."}
    counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    for item in news:
        counts[item.sentiment] = counts.get(item.sentiment, 0) + 1
    dominant = max(counts, key=counts.get)
    total = len(news)
    summary = (
        f"{dominant.upper()} — "
        f"{counts['bullish']} bullish, {counts['bearish']} bearish, "
        f"{counts['neutral']} neutral ({total} articles)"
    )
    return {"sentiment_summary": summary}


def fetch_price(state: FinanceState) -> dict:
    ticker = state["ticker"]
    price_data = _get_stock_data(_GetStockDataInput(ticker=ticker))
    return {"price_data": price_data}


def build_finance_graph():
    builder = StateGraph(FinanceState)
    builder.add_node("collect_news", collect_news)
    builder.add_node("analyze_sentiment", analyze_sentiment)
    builder.add_node("fetch_price", fetch_price)
    builder.add_edge(START, "collect_news")
    builder.add_edge("collect_news", "analyze_sentiment")
    builder.add_edge("analyze_sentiment", "fetch_price")
    builder.add_edge("fetch_price", END)
    return builder.compile()


if __name__ == "__main__":
    graph = build_finance_graph()
    result = graph.invoke({"ticker": "NVDA", "messages": []})

    print("=== Finance Graph Output ===")
    print(f"Ticker     : {result['ticker']}")
    print(f"News count : {len(result.get('news', []))}")
    print(f"Sentiment  : {result.get('sentiment_summary')}")
    price = result.get("price_data", {})
    print(f"Price      : ${price.get('price')}  ({price.get('pct_change', 0):+.2f}% 1mo)")

    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("graph/finance_graph.png", "wb") as f:
        f.write(png_bytes)
    print("\nGraph PNG saved → graph/finance_graph.png")
