"""
Day 28 — Full System Run: NVDA, TSLA, MSFT with Checkpointing

Test edilen özellikler:
    1. Üç ticker (NVDA, TSLA, MSFT) izole thread_id ile çalıştırılıyor
    2. Checkpointer: her run SQLite'a yazılıyor, get_state() geri okunabiliyor
    3. Thread izolasyonu: farklı ticker'ların state'i birbirine karışmıyor
    4. get_state_history(): her run ≥ 1 checkpoint adımı içeriyor
    5. Mocked E2E: tüm pipeline (subgraph dahil) baştan sona akıyor
    6. Draft üretildi: son state'te "draft" alanı dolu

Çalıştırma (canlı API):
    python test_graph_day28.py NVDA
    python test_graph_day28.py TSLA
    python test_graph_day28.py MSFT
    python test_graph_day28.py --all
"""

from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from graph.checkpointer import make_checkpointer
from graph.finance_graph import build_finance_graph
from schemas import NewsAnalysis

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

TICKERS = ["NVDA", "TSLA", "MSFT"]


def _fake_news(ticker: str) -> list[dict]:
    return [
        {
            "ticker": ticker,
            "sentiment": "bullish",
            "urgency": "high",
            "key_event": "earnings",
            "summary": f"{ticker} beat expectations this quarter.",
        },
        {
            "ticker": ticker,
            "sentiment": "neutral",
            "urgency": "low",
            "key_event": "other",
            "summary": f"{ticker} trading sideways amid market uncertainty.",
        },
    ]


def _build_mock_provider(ticker: str):
    """
    İki adımlı sahte provider:
        1. çağrı → tool_calls (analyze_news_sentiment)
        2. çağrı → text (döngü biter)
    """
    from providers.base import LLMResponse, LLMUsage, NextAction, ToolCall

    tool_resp = LLMResponse(
        text="",
        usage=LLMUsage(input_tokens=40, output_tokens=15, total_tokens=55),
        next_action=NextAction(
            type="tool_calls",
            tool_calls=[
                ToolCall(
                    id="call_01",
                    name="analyze_news_sentiment",
                    input={"ticker": ticker, "top_n": 5},
                )
            ],
        ),
        raw_assistant_content=[
            {
                "type": "function_call",
                "id": "item_01",
                "call_id": "call_01",
                "name": "analyze_news_sentiment",
                "arguments": f'{{"ticker": "{ticker}", "top_n": 5}}',
            }
        ],
    )

    text_resp = LLMResponse(
        text=f"{ticker} sentiment analysis complete.",
        usage=LLMUsage(input_tokens=150, output_tokens=25, total_tokens=175),
        next_action=NextAction(type="text"),
        raw_assistant_content=None,
    )

    call_count: dict[str, int] = {"n": 0}

    def fake_generate(messages, tools=None, system=None):
        call_count["n"] += 1
        return tool_resp if call_count["n"] == 1 else text_resp

    provider = MagicMock()
    provider.generate.side_effect = fake_generate
    return provider


def _run_graph_mocked(ticker: str, db_path: str) -> dict:
    """
    Tek bir ticker için mocked pipeline çalıştırır, son state'i döndürür.
    """
    fake_news = _fake_news(ticker)
    provider = _build_mock_provider(ticker)
    thread_id = f"day28-{ticker.lower()}-run"

    with (
        patch("graph.research_subgraph.get_provider", return_value=provider),
        patch("graph.research_subgraph._make_registry") as mock_reg_fn,
        patch("graph.finance_graph._query_10k") as mock_rag,
        patch("graph.finance_graph._get_stock_data") as mock_price,
    ):
        mock_registry = MagicMock()
        mock_registry.all_tools.return_value = []
        mock_registry.dispatch.return_value = fake_news
        mock_reg_fn.return_value = mock_registry

        mock_rag.return_value = {
            "answer": f"{ticker} key risks include competition and regulatory exposure.",
            "sources": [],
        }
        mock_price.return_value = {
            "ticker": ticker,
            "price": 150.0 + hash(ticker) % 100,
            "pct_change": 1.5,
        }

        with make_checkpointer(db_path) as cp:
            graph = build_finance_graph(checkpointer=cp)
            config = {"configurable": {"thread_id": thread_id}}
            result = graph.invoke({"ticker": ticker, "messages": []}, config)
            state = graph.get_state(config)
            history = list(graph.get_state_history(config))

    return {
        "ticker": ticker,
        "thread_id": thread_id,
        "result": result,
        "final_state": state,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Test 1: Üç ticker mocked run — draft üretildi
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ticker", TICKERS)
def test_run_produces_draft(ticker, tmp_path):
    """Her ticker için pipeline draft alanını doldurmalı."""
    db = str(tmp_path / f"{ticker.lower()}_day28.db")
    run = _run_graph_mocked(ticker, db)
    draft = run["result"].get("draft", "")
    assert draft, f"{ticker}: draft boş geldi"
    assert ticker in draft or "[DRAFT]" in draft or "[SHORT BRIEF]" in draft


# ---------------------------------------------------------------------------
# Test 2: Checkpointer — get_state() son değerleri döndürüyor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ticker", TICKERS)
def test_checkpointer_get_state(ticker, tmp_path):
    """
    get_state() run bittikten sonra:
        - doğru ticker döndürmeli
        - news alanı dolu olmalı
        - draft alanı dolu olmalı
    """
    db = str(tmp_path / f"{ticker.lower()}_cp.db")
    run = _run_graph_mocked(ticker, db)

    values = run["final_state"].values
    assert values.get("ticker") == ticker, f"{ticker}: ticker eşleşmedi"
    assert len(values.get("news", [])) > 0, f"{ticker}: news boş"
    assert values.get("draft"), f"{ticker}: checkpoint draft boş"


# ---------------------------------------------------------------------------
# Test 3: Checkpointer — get_state_history() ≥ 1 adım
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ticker", TICKERS)
def test_checkpointer_history_has_steps(ticker, tmp_path):
    """Her run SQLite'a en az 1 checkpoint adımı yazmalı."""
    db = str(tmp_path / f"{ticker.lower()}_hist.db")
    run = _run_graph_mocked(ticker, db)
    assert len(run["history"]) >= 1, f"{ticker}: history boş — checkpoint yazılmadı"


# ---------------------------------------------------------------------------
# Test 4: Thread izolasyonu
# ---------------------------------------------------------------------------

def test_thread_isolation(tmp_path):
    """
    Aynı DB'de NVDA ve TSLA ayrı thread_id ile çalıştırılırsa
    her birinin get_state() kendi ticker'ını döndürmeli.
    """
    db = str(tmp_path / "isolation.db")

    with (
        patch("graph.research_subgraph.get_provider") as mock_prov,
        patch("graph.research_subgraph._make_registry") as mock_reg_fn,
        patch("graph.finance_graph._query_10k") as mock_rag,
        patch("graph.finance_graph._get_stock_data") as mock_price,
    ):
        mock_rag.return_value = {"answer": "risk text", "sources": []}
        mock_price.return_value = {"ticker": "X", "price": 100.0, "pct_change": 0.5}

        def make_provider(ticker: str):
            p = _build_mock_provider(ticker)
            return p

        def make_registry(ticker: str):
            r = MagicMock()
            r.all_tools.return_value = []
            r.dispatch.return_value = _fake_news(ticker)
            return r

        with make_checkpointer(db) as cp:
            graph = build_finance_graph(checkpointer=cp)

            for ticker in ["NVDA", "TSLA"]:
                mock_prov.return_value = make_provider(ticker)
                mr = MagicMock()
                mr.all_tools.return_value = []
                mr.dispatch.return_value = _fake_news(ticker)
                mock_reg_fn.return_value = mr

                config = {"configurable": {"thread_id": f"isolation-{ticker.lower()}"}}
                graph.invoke({"ticker": ticker, "messages": []}, config)

            nvda_state = graph.get_state({"configurable": {"thread_id": "isolation-nvda"}})
            tsla_state = graph.get_state({"configurable": {"thread_id": "isolation-tsla"}})

    assert nvda_state.values.get("ticker") == "NVDA"
    assert tsla_state.values.get("ticker") == "TSLA"


# ---------------------------------------------------------------------------
# Test 5: Sentiment field dolu (analyze_sentiment çalıştı)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ticker", TICKERS)
def test_sentiment_summary_present(ticker, tmp_path):
    """analyze_sentiment node'u sentiment_summary alanını doldurmalı."""
    db = str(tmp_path / f"{ticker.lower()}_sent.db")
    run = _run_graph_mocked(ticker, db)
    summary = run["result"].get("sentiment_summary", "")
    assert summary, f"{ticker}: sentiment_summary boş"


# ---------------------------------------------------------------------------
# Test 6: price_data field dolu (fetch_price çalıştı)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ticker", TICKERS)
def test_price_data_present(ticker, tmp_path):
    """fetch_price node'u price_data alanını doldurmalı."""
    db = str(tmp_path / f"{ticker.lower()}_price.db")
    run = _run_graph_mocked(ticker, db)
    price_data = run["result"].get("price_data", {})
    assert price_data, f"{ticker}: price_data boş"
    assert "price" in price_data, f"{ticker}: price_data.price eksik"


# ---------------------------------------------------------------------------
# Canlı çalıştırma (python test_graph_day28.py NVDA | TSLA | MSFT | --all)
# ---------------------------------------------------------------------------

def _live_run(ticker: str):
    """Gerçek API çağrılarıyla tek ticker'ı çalıştırır."""
    db_path = f"day28_{ticker.lower()}.db"
    thread_id = f"v0.3.0-{ticker.lower()}"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'='*60}")
    print(f"  TICKER: {ticker}  |  thread_id: {thread_id}")
    print(f"  DB: {db_path}")
    print("=" * 60)

    with make_checkpointer(db_path) as cp:
        graph = build_finance_graph(checkpointer=cp)
        result = graph.invoke({"ticker": ticker, "messages": []}, config)

        state = graph.get_state(config)
        history = list(graph.get_state_history(config))

    news = result.get("news", [])
    draft = result.get("draft", "")
    sentiment = result.get("sentiment_summary", "")
    price = result.get("price_data", {})

    print(f"\n[sentiment] {sentiment}")
    print(f"[news]      {len(news)} articles")
    print(f"[price]     ${price.get('price', 'N/A')}")
    print(f"[draft]     {draft[:200]}...")
    print(f"\n[checkpoint] thread={thread_id}, steps={len(history)}, db={db_path}")
    print(f"\n[DONE] {ticker} run successful.")
    return result


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python test_graph_day28.py NVDA | TSLA | MSFT | --all")
        sys.exit(1)

    targets = TICKERS if "--all" in args else [a.upper() for a in args]

    for t in targets:
        if t not in TICKERS:
            print(f"Unknown ticker: {t}")
            continue
        _live_run(t)

    print(f"\n[ALL DONE] {len(targets)} run(s) completed. Check day28_*.db files.")
