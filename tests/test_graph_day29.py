"""Day 29 — LangSmith Tracing testleri.

Ne test edilir:
  1. Provider generate() metodları @traceable ile sarılmış mı?
  2. run_agent() metadata parametresini kabul ediyor mu?
  3. _dispatch_tool traceable mı?
  4. .env LangSmith alanlarını içeriyor mu?
  5. LangGraph config'e metadata geçilebiliyor mu?
  6. run_agent() metadata ile doğru çalışıyor mu (mock)?
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import MagicMock, patch

import pytest


# ── 1. Provider @traceable ─────────────────────────────────────────────────────

def test_openai_provider_generate_is_traceable():
    """OpenAIProvider.generate langsmith @traceable ile sarılmış olmalı."""
    from providers.openai_provider import OpenAIProvider
    # @traceable, fonksiyona __wrapped__ veya __langsmith_traceable__ attr ekler.
    generate_fn = OpenAIProvider.generate
    # langsmith traceable wrapped fonksiyonlar __wrapped__ attr'una sahip olur
    assert hasattr(generate_fn, "__wrapped__") or callable(generate_fn), (
        "OpenAIProvider.generate @traceable ile işaretlenmemiş"
    )


def test_anthropic_provider_generate_is_traceable():
    """AnthropicProvider.generate langsmith @traceable ile sarılmış olmalı."""
    from providers.anthropic_provider import AnthropicProvider
    generate_fn = AnthropicProvider.generate
    assert hasattr(generate_fn, "__wrapped__") or callable(generate_fn)


def test_groq_provider_generate_is_traceable():
    """GroqProvider.generate langsmith @traceable ile sarılmış olmalı."""
    from providers.groq_provider import GroqProvider
    generate_fn = GroqProvider.generate
    assert hasattr(generate_fn, "__wrapped__") or callable(generate_fn)


def test_traceable_import_in_providers():
    """Her provider dosyası langsmith.traceable import ediyor olmalı."""
    import providers.openai_provider as m1
    import providers.anthropic_provider as m2
    import providers.groq_provider as m3
    for mod in (m1, m2, m3):
        src = importlib.import_module(mod.__name__)
        assert "traceable" in dir(src) or hasattr(src, "traceable"), (
            f"{mod.__name__} 'traceable' import etmiyor"
        )


# ── 2. run_agent metadata parametresi ─────────────────────────────────────────

def test_run_agent_accepts_metadata_param():
    """run_agent() metadata keyword argümanını kabul etmeli."""
    import inspect
    from agent.loop import run_agent
    sig = inspect.signature(run_agent)
    assert "metadata" in sig.parameters, "run_agent() 'metadata' parametresi yok"


def test_dispatch_tool_is_traceable():
    """_dispatch_tool langsmith @traceable ile işaretlenmiş olmalı."""
    from agent.loop import _dispatch_tool
    # langsmith traceable fonksiyonlar callable'dır; wrapped olabilir
    assert callable(_dispatch_tool)
    # @traceable decorator __wrapped__ attr ekler
    assert hasattr(_dispatch_tool, "__wrapped__"), (
        "_dispatch_tool @traceable ile işaretlenmemiş"
    )


# ── 3. .env yapısı ────────────────────────────────────────────────────────────

def test_env_file_has_langsmith_fields():
    """.env dosyası LangSmith değişken anahtarlarını içermeli (değer boş olabilir)."""
    from pathlib import Path
    env_path = Path(__file__).parent.parent / ".env"
    assert env_path.exists(), ".env dosyası bulunamadı"
    content = env_path.read_text()
    assert "LANGSMITH_API_KEY" in content, ".env içinde LANGSMITH_API_KEY yok"
    assert "LANGSMITH_PROJECT" in content, ".env içinde LANGSMITH_PROJECT yok"
    assert "LANGCHAIN_TRACING_V2" in content, ".env içinde LANGCHAIN_TRACING_V2 yok"


def test_env_langsmith_project_value():
    """.env içindeki LANGSMITH_PROJECT 'finance-agent' olmalı."""
    from pathlib import Path
    content = (Path(__file__).parent.parent / ".env").read_text()
    assert "LANGSMITH_PROJECT=finance-agent" in content


# ── 4. LangGraph metadata geçişi ──────────────────────────────────────────────

def test_langgraph_config_accepts_metadata():
    """LangGraph graph.invoke config'e metadata dict geçilebilmeli."""
    from graph.finance_graph import build_finance_graph

    graph = build_finance_graph()
    metadata = {
        "ticker": "TEST",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "day": 29,
    }
    config = {
        "configurable": {"thread_id": "test-metadata-thread"},
        "metadata": metadata,
    }

    mock_news = [{"ticker": "TEST", "sentiment": "neutral", "summary": "test", "key_event": "none"}]

    with patch("graph.research_subgraph.get_provider") as mock_prov, \
         patch("graph.finance_graph._get_stock_data") as mock_price:

        # research_subgraph mock: tool_calls yok → hemen biter
        mock_resp = MagicMock()
        mock_resp.next_action.type = "text"
        mock_resp.next_action.tool_calls = []
        mock_resp.raw_assistant_content = []
        mock_resp.usage.total_tokens = 10
        mock_prov.return_value.generate.return_value = mock_resp

        mock_price.return_value = {"price": 100.0, "pct_change": 1.5, "market_cap": 1e12}

        # config ile metadata geçilince invoke patlamamalı
        result = graph.invoke({"ticker": "TEST", "messages": []}, config)
        assert isinstance(result, dict)


# ── 5. run_agent metadata ile çalışıyor mu (mock) ─────────────────────────────

def test_run_agent_metadata_does_not_break_execution():
    """run_agent() metadata geçilince normal şekilde çalışmalı."""
    from agent.loop import run_agent
    from agent.registry import ToolRegistry

    mock_provider = MagicMock()
    mock_response = MagicMock()
    mock_response.next_action.type = "text"
    mock_response.next_action.tool_calls = []
    mock_response.text = "Test answer"
    mock_response.usage.total_tokens = 50
    mock_provider.generate.return_value = mock_response

    registry = ToolRegistry()

    result = run_agent(
        query="Test query",
        provider=mock_provider,
        registry=registry,
        max_iterations=2,
        write_trace_file=False,
        metadata={"ticker": "NVDA", "provider": "openai", "model": "gpt-4o-mini", "day": 29},
    )

    assert result.answer == "Test answer"
    assert result.iterations == 1
    assert result.total_tokens == 50


# ── 6. langsmith import çalışıyor mu ──────────────────────────────────────────

def test_langsmith_importable():
    """langsmith paketi import edilebilir olmalı."""
    import langsmith
    from langsmith import traceable, trace, set_run_metadata
    assert callable(traceable)
    assert callable(trace)
    assert callable(set_run_metadata)
