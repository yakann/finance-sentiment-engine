"""Day 24: Conditional routing by sentiment — route_by_sentiment + new nodes."""

import pytest
from unittest.mock import patch

from graph.finance_graph import (
    analyze_sentiment,
    build_finance_graph,
    deep_analysis,
    route_by_sentiment,
    short_brief,
)
from graph.state import FinanceState
from schemas import NewsAnalysis

# ---------------------------------------------------------------------------
# Test verisi
# ---------------------------------------------------------------------------

_BULLISH_NEWS = [
    NewsAnalysis(ticker="TST", sentiment="bullish", urgency="high",
                 key_event="earnings", summary="Beat estimates"),
    NewsAnalysis(ticker="TST", sentiment="bullish", urgency="medium",
                 key_event="analyst_action", summary="Price target raised"),
    NewsAnalysis(ticker="TST", sentiment="neutral", urgency="low",
                 key_event="market_dynamics", summary="Sector rotation noted"),
]

_BEARISH_NEWS = [
    NewsAnalysis(ticker="TST", sentiment="bearish", urgency="high",
                 key_event="earnings", summary="Missed estimates"),
    NewsAnalysis(ticker="TST", sentiment="bearish", urgency="medium",
                 key_event="policy_geopolitical", summary="Tariff headwinds"),
    NewsAnalysis(ticker="TST", sentiment="neutral", urgency="low",
                 key_event="market_dynamics", summary="Market mixed"),
]

_NEUTRAL_NEWS = [
    NewsAnalysis(ticker="TST", sentiment="neutral", urgency="low",
                 key_event="market_dynamics", summary="Sideways movement"),
    NewsAnalysis(ticker="TST", sentiment="bullish", urgency="low",
                 key_event="analyst_action", summary="Hold rating maintained"),
    NewsAnalysis(ticker="TST", sentiment="bearish", urgency="low",
                 key_event="market_dynamics", summary="Caution advised"),
]


# ---------------------------------------------------------------------------
# route_by_sentiment — birim testler
# ---------------------------------------------------------------------------

def test_route_bullish_goes_to_deep_analysis():
    state: FinanceState = {
        "ticker": "TST",
        "news": _BULLISH_NEWS,
        "sentiment_summary": "BULLISH — 2 bullish, 0 bearish, 1 neutral (3 articles)",
    }
    assert route_by_sentiment(state) == "deep_analysis"


def test_route_bearish_goes_to_deep_analysis():
    state: FinanceState = {
        "ticker": "TST",
        "news": _BEARISH_NEWS,
        "sentiment_summary": "BEARISH — 0 bullish, 2 bearish, 1 neutral (3 articles)",
    }
    assert route_by_sentiment(state) == "deep_analysis"


def test_route_neutral_goes_to_short_brief():
    state: FinanceState = {
        "ticker": "TST",
        "news": _NEUTRAL_NEWS,
        "sentiment_summary": "NEUTRAL — 1 bullish, 1 bearish, 1 neutral (3 articles)",
    }
    assert route_by_sentiment(state) == "short_brief"


def test_route_empty_summary_goes_to_short_brief():
    state: FinanceState = {"ticker": "TST"}
    assert route_by_sentiment(state) == "short_brief"


def test_route_no_news_summary_goes_to_short_brief():
    state: FinanceState = {"ticker": "TST", "sentiment_summary": "No news available."}
    assert route_by_sentiment(state) == "short_brief"


# ---------------------------------------------------------------------------
# short_brief node — birim testler
# ---------------------------------------------------------------------------

def test_short_brief_sets_draft():
    state: FinanceState = {
        "ticker": "TST",
        "news": _NEUTRAL_NEWS,
        "sentiment_summary": "NEUTRAL — 1 bullish, 1 bearish, 1 neutral (3 articles)",
    }
    result = short_brief(state)
    assert "draft" in result
    assert "TST" in result["draft"]
    assert "NEUTRAL" in result["draft"]


def test_short_brief_includes_headlines():
    state: FinanceState = {
        "ticker": "TST",
        "news": _NEUTRAL_NEWS,
        "sentiment_summary": "NEUTRAL — 1 bullish, 1 bearish, 1 neutral (3 articles)",
    }
    result = short_brief(state)
    # İlk haber özetinin draft'a girdiğini doğrula
    assert "Sideways movement" in result["draft"]


def test_short_brief_no_news():
    state: FinanceState = {"ticker": "TST", "news": [], "sentiment_summary": "No news available."}
    result = short_brief(state)
    assert "draft" in result
    assert "No headlines available" in result["draft"]


# ---------------------------------------------------------------------------
# deep_analysis node — mock'lu birim testler
# ---------------------------------------------------------------------------

def test_deep_analysis_sets_risks():
    state: FinanceState = {
        "ticker": "NVDA",
        "sentiment_summary": "BULLISH — 3 bullish, 0 bearish, 0 neutral (3 articles)",
    }
    with patch("graph.finance_graph._query_10k") as mock_q:
        mock_q.return_value = {
            "answer": "Key risks include supply chain disruptions and increased competition.",
            "sources": [],
        }
        result = deep_analysis(state)

    assert "risks" in result
    assert isinstance(result["risks"], list)
    assert len(result["risks"]) == 1
    assert "supply chain" in result["risks"][0]


def test_deep_analysis_unsupported_ticker():
    state: FinanceState = {
        "ticker": "FAKE",
        "sentiment_summary": "BEARISH — 0 bullish, 3 bearish, 0 neutral (3 articles)",
    }
    with patch("graph.finance_graph._query_10k") as mock_q:
        mock_q.return_value = {"error": "No 10-K index for 'FAKE'."}
        result = deep_analysis(state)

    assert "risks" in result
    assert "No 10-K index" in result["risks"][0]


# ---------------------------------------------------------------------------
# Graf topolojisi testler
# ---------------------------------------------------------------------------

def test_graph_has_branch_nodes():
    graph = build_finance_graph()
    node_names = set(graph.get_graph().nodes.keys())
    assert "deep_analysis" in node_names
    assert "short_brief" in node_names
    assert "fetch_price" in node_names


def test_graph_no_direct_sentiment_to_price_edge():
    """Day 24'te bu kenar artık conditional edges'e bırakıldı."""
    graph = build_finance_graph()
    edge_pairs = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("analyze_sentiment", "fetch_price") not in edge_pairs


def test_graph_branches_converge_at_fetch_price():
    graph = build_finance_graph()
    edge_pairs = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("deep_analysis", "fetch_price") in edge_pairs
    assert ("short_brief", "fetch_price") in edge_pairs


def test_graph_mermaid_contains_branch_nodes():
    graph = build_finance_graph()
    mermaid_text = graph.get_graph().draw_mermaid()
    assert "deep_analysis" in mermaid_text
    assert "short_brief" in mermaid_text
    assert "fetch_price" in mermaid_text


# ---------------------------------------------------------------------------
# Integration test (live ağ çağrısı — yavaş)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_full_graph_nvda_conditional():
    graph = build_finance_graph()
    result = graph.invoke({"ticker": "NVDA", "messages": []})

    assert result["ticker"] == "NVDA"
    assert isinstance(result.get("news"), list)
    assert isinstance(result.get("sentiment_summary"), str)
    assert isinstance(result.get("price_data"), dict)
    assert result["price_data"].get("price", 0) > 0

    # Sentiment'e göre ya risks ya draft dolu olmalı
    has_deep = bool(result.get("risks"))
    has_brief = bool(result.get("draft"))
    assert has_deep or has_brief, "Neither deep_analysis nor short_brief ran"
