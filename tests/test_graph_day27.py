"""
Day 27 Tests — Research Subgraph

Test edilen özellikler:
    1. Subgraph import ve compile (hata yok)
    2. should_continue: pending_tool_calls var → dispatch_tools
    3. should_continue: pending_tool_calls yok → __end__
    4. dispatch_tools: news doğru çıkarılıyor (NewsAnalysis dönüşümü)
    5. dispatch_tools: hatalı tool sonucu tolere ediliyor
    6. Ana graph build_research_subgraph() node'u ile derleniyor
    7. Subgraph call_model → dispatch_tools döngüsü (mocked provider)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from graph.research_subgraph import (
    ResearchState,
    build_research_subgraph,
    dispatch_tools,
    should_continue,
)
from schemas import NewsAnalysis


# ---------------------------------------------------------------------------
# Test 1: Subgraph compile
# ---------------------------------------------------------------------------

def test_subgraph_compiles():
    """build_research_subgraph() exception olmadan derlenebilmeli."""
    sg = build_research_subgraph()
    assert sg is not None


# ---------------------------------------------------------------------------
# Test 2 & 3: should_continue routing
# ---------------------------------------------------------------------------

def test_should_continue_with_tool_calls():
    """pending_tool_calls dolu ise dispatch_tools'a yönlenmeli."""
    state: ResearchState = {
        "ticker": "NVDA",
        "loop_messages": [],
        "pending_tool_calls": [{"id": "c1", "name": "analyze_news_sentiment", "input": {"ticker": "NVDA"}}],
        "news": [],
    }
    assert should_continue(state) == "dispatch_tools"


def test_should_continue_without_tool_calls():
    """pending_tool_calls boş ise __end__'e yönlenmeli."""
    state: ResearchState = {
        "ticker": "NVDA",
        "loop_messages": [],
        "pending_tool_calls": [],
        "news": [],
    }
    assert should_continue(state) == "__end__"


def test_should_continue_missing_key():
    """pending_tool_calls key'i hiç yoksa __end__'e yönlenmeli."""
    state: ResearchState = {"ticker": "TSLA"}  # type: ignore
    assert should_continue(state) == "__end__"


# ---------------------------------------------------------------------------
# Test 4: dispatch_tools — NewsAnalysis dönüşümü
# ---------------------------------------------------------------------------

def test_dispatch_tools_extracts_news():
    """
    analyze_news_sentiment tool'u doğru veri döndürdüğünde
    dispatch_tools NewsAnalysis nesneleri üretmeli.
    """
    fake_results = [
        {
            "ticker": "NVDA",
            "sentiment": "bullish",
            "urgency": "high",
            "key_event": "earnings",
            "summary": "NVDA beat expectations.",
        },
        {
            "ticker": "NVDA",
            "sentiment": "neutral",
            "urgency": "low",
            "key_event": "other",
            "summary": "No major events.",
        },
    ]

    state: ResearchState = {
        "ticker": "NVDA",
        "loop_messages": [],
        "pending_tool_calls": [
            {"id": "c1", "name": "analyze_news_sentiment", "input": {"ticker": "NVDA", "top_n": 5}},
        ],
        "news": [],
    }

    with patch("graph.research_subgraph._make_registry") as mock_reg_fn:
        mock_registry = MagicMock()
        mock_registry.dispatch.return_value = fake_results
        mock_reg_fn.return_value = mock_registry

        result = dispatch_tools(state)

    news = result["news"]
    assert len(news) == 2
    assert all(isinstance(n, NewsAnalysis) for n in news)
    assert news[0].sentiment == "bullish"
    assert news[1].sentiment == "neutral"
    assert result["pending_tool_calls"] == []


# ---------------------------------------------------------------------------
# Test 5: dispatch_tools — hata toleransı
# ---------------------------------------------------------------------------

def test_dispatch_tools_error_tolerance():
    """
    Tool exception fırlatırsa dispatch_tools hata yaymadan devam etmeli;
    news boş olmalı, loop_messages error içermeli.
    """
    state: ResearchState = {
        "ticker": "FAIL",
        "loop_messages": [],
        "pending_tool_calls": [
            {"id": "c2", "name": "analyze_news_sentiment", "input": {"ticker": "FAIL"}},
        ],
        "news": [],
    }

    with patch("graph.research_subgraph._make_registry") as mock_reg_fn:
        mock_registry = MagicMock()
        mock_registry.dispatch.side_effect = RuntimeError("API down")
        mock_reg_fn.return_value = mock_registry

        result = dispatch_tools(state)

    # Hata mesajı loop_messages'e function_call_output olarak yazılmalı
    assert result["pending_tool_calls"] == []
    assert result["news"] == []
    assert len(result["loop_messages"]) == 1
    payload = json.loads(result["loop_messages"][0]["output"])
    assert "error" in payload


# ---------------------------------------------------------------------------
# Test 6: Ana graph subgraph node'u ile derleniyor
# ---------------------------------------------------------------------------

def test_main_graph_compiles_with_subgraph():
    """build_finance_graph() research_subgraph dahil exception olmadan derlenebilmeli."""
    from graph.finance_graph import build_finance_graph

    graph = build_finance_graph()
    assert graph is not None


# ---------------------------------------------------------------------------
# Test 7: Subgraph call_model → dispatch_tools döngüsü (mocked)
# ---------------------------------------------------------------------------

def test_subgraph_invoke_mocked():
    """
    Provider mock'lanarak subgraph'ın tool-use döngüsünü baştan sona test eder.

    Senaryo:
        1. call_model → LLM analyze_news_sentiment çağırır (tool_calls)
        2. dispatch_tools → tool çalışır, news çıkarılır
        3. call_model → LLM metin yanıt verir (text) → END
        4. Subgraph state'inde news dolu olmalı
    """
    from providers.base import LLMResponse, LLMUsage, NextAction, ToolCall

    fake_news_result = [
        {
            "ticker": "AAPL",
            "sentiment": "bullish",
            "urgency": "medium",
            "key_event": "product_event",
            "summary": "Apple launched new iPhone.",
        }
    ]

    # İlk LLM yanıtı: tool call talep et
    tool_response = LLMResponse(
        text="",
        usage=LLMUsage(input_tokens=50, output_tokens=20, total_tokens=70),
        next_action=NextAction(
            type="tool_calls",
            tool_calls=[ToolCall(id="call_abc", name="analyze_news_sentiment", input={"ticker": "AAPL", "top_n": 5})],
        ),
        raw_assistant_content=[{
            "type": "function_call",
            "id": "item_1",
            "call_id": "call_abc",
            "name": "analyze_news_sentiment",
            "arguments": '{"ticker": "AAPL", "top_n": 5}',
        }],
    )

    # İkinci LLM yanıtı: metin yanıt (döngü biter)
    text_response = LLMResponse(
        text="AAPL looks bullish based on recent news.",
        usage=LLMUsage(input_tokens=200, output_tokens=30, total_tokens=230),
        next_action=NextAction(type="text"),
        raw_assistant_content=None,
    )

    call_count = {"n": 0}

    def fake_generate(messages, tools=None, system=None):
        call_count["n"] += 1
        return tool_response if call_count["n"] == 1 else text_response

    with (
        patch("graph.research_subgraph.get_provider") as mock_prov_fn,
        patch("graph.research_subgraph._make_registry") as mock_reg_fn,
    ):
        mock_provider = MagicMock()
        mock_provider.generate.side_effect = fake_generate
        mock_prov_fn.return_value = mock_provider

        mock_registry = MagicMock()
        mock_registry.all_tools.return_value = []
        mock_registry.dispatch.return_value = fake_news_result
        mock_reg_fn.return_value = mock_registry

        sg = build_research_subgraph()
        final = sg.invoke({"ticker": "AAPL"})

    news = final.get("news", [])
    assert len(news) == 1
    assert isinstance(news[0], NewsAnalysis)
    assert news[0].ticker == "AAPL"
    assert news[0].sentiment == "bullish"
    # İki LLM çağrısı yapılmalı: 1 tool call turu + 1 metin turu
    assert call_count["n"] == 2
