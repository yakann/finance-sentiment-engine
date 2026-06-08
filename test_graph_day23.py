"""Day 23: FinanceState schema + 3-node linear LangGraph."""

import pytest
from graph.finance_graph import (
    analyze_sentiment,
    build_finance_graph,
    collect_news,
    fetch_price,
)
from graph.state import FinanceState
from schemas import NewsAnalysis


# ---------------------------------------------------------------------------
# State schema tests
# ---------------------------------------------------------------------------

def test_state_has_required_keys():
    from typing import get_type_hints
    hints = get_type_hints(FinanceState, include_extras=True)
    for key in ("ticker", "news", "price_data", "risks", "draft", "messages"):
        assert key in hints, f"FinanceState missing field: {key}"


def test_state_sentiment_summary_key():
    from typing import get_type_hints
    hints = get_type_hints(FinanceState, include_extras=True)
    assert "sentiment_summary" in hints


# ---------------------------------------------------------------------------
# Node unit tests (no external calls)
# ---------------------------------------------------------------------------

_SAMPLE_NEWS = [
    NewsAnalysis(ticker="TST", sentiment="bullish", urgency="high",
                 key_event="earnings", summary="Beat estimates"),
    NewsAnalysis(ticker="TST", sentiment="bullish", urgency="medium",
                 key_event="analyst_action", summary="Target raised"),
    NewsAnalysis(ticker="TST", sentiment="bearish", urgency="low",
                 key_event="policy_geopolitical", summary="Tariff risk"),
]


def test_analyze_sentiment_dominant():
    state: FinanceState = {"ticker": "TST", "news": _SAMPLE_NEWS}
    result = analyze_sentiment(state)
    assert "sentiment_summary" in result
    assert "BULLISH" in result["sentiment_summary"]


def test_analyze_sentiment_counts():
    state: FinanceState = {"ticker": "TST", "news": _SAMPLE_NEWS}
    result = analyze_sentiment(state)
    summary = result["sentiment_summary"]
    assert "2 bullish" in summary
    assert "1 bearish" in summary
    assert "0 neutral" in summary


def test_analyze_sentiment_empty_news():
    state: FinanceState = {"ticker": "TST", "news": []}
    result = analyze_sentiment(state)
    assert result["sentiment_summary"] == "No news available."


# ---------------------------------------------------------------------------
# Graph structure test (no external calls)
# ---------------------------------------------------------------------------

def test_graph_nodes_and_edges():
    graph = build_finance_graph()
    schema = graph.get_graph()
    node_names = set(schema.nodes.keys())
    assert "collect_news" in node_names
    assert "analyze_sentiment" in node_names
    assert "fetch_price" in node_names


def test_graph_edge_order():
    graph = build_finance_graph()
    schema = graph.get_graph()
    edge_pairs = {(e.source, e.target) for e in schema.edges}
    assert ("__start__", "collect_news") in edge_pairs
    assert ("collect_news", "analyze_sentiment") in edge_pairs
    assert ("analyze_sentiment", "fetch_price") in edge_pairs
    assert ("fetch_price", "__end__") in edge_pairs


# ---------------------------------------------------------------------------
# Integration test (live network call — mark slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_full_graph_nvda():
    graph = build_finance_graph()
    result = graph.invoke({"ticker": "NVDA", "messages": []})

    assert result["ticker"] == "NVDA"
    assert isinstance(result.get("news"), list)
    assert isinstance(result.get("sentiment_summary"), str)
    assert len(result["sentiment_summary"]) > 0
    assert isinstance(result.get("price_data"), dict)
    assert result["price_data"].get("price", 0) > 0
