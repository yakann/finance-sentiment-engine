"""
Day 25 — SQLite Checkpointing Testleri

Test senaryoları:
1. Checkpointer ile graf derlenebilir (compile)
2. İlk run'dan sonra checkpoint SQLite'ta yazılır
3. Aynı thread_id ile get_state → persist edilmiş state okunur
4. Farklı thread_id'ler birbirini etkilemez (izolasyon)
5. Checkpoint'ten state inspect edilebilir (debug altın kural)
"""

import os
import tempfile
import pytest
from unittest.mock import patch

from graph.checkpointer import make_checkpointer
from graph.finance_graph import build_finance_graph
from schemas import NewsAnalysis

# ---------------------------------------------------------------------------
# Yardımcı sabitler
# ---------------------------------------------------------------------------

_MOCK_NEWS_RESULT = [
    {"ticker": "TST", "sentiment": "neutral", "urgency": "low",
     "key_event": "market_dynamics", "summary": "Sideways movement"},
]

_MOCK_PRICE_RESULT = {
    "ticker": "TST", "price": 100.0, "pct_change": 0.5,
    "volume": 1_000_000, "market_cap": 500_000_000,
}


def _mock_patches():
    """Tüm network çağrılarını mock'la."""
    return [
        patch("graph.finance_graph._analyze_news_sentiment", return_value=_MOCK_NEWS_RESULT),
        patch("graph.finance_graph._get_stock_data", return_value=_MOCK_PRICE_RESULT),
    ]


# ---------------------------------------------------------------------------
# Test 1: checkpointer ile graf derlenir
# ---------------------------------------------------------------------------

def test_graph_compiles_with_checkpointer():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        with make_checkpointer(db_path) as cp:
            graph = build_finance_graph(checkpointer=cp)
            assert graph is not None
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 2: run sonrası checkpoint SQLite dosyasında oluşur
# ---------------------------------------------------------------------------

def test_checkpoint_file_created_after_run():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        patches = _mock_patches()
        for p in patches:
            p.start()

        with make_checkpointer(db_path) as cp:
            graph = build_finance_graph(checkpointer=cp)
            config = {"configurable": {"thread_id": "test-file-created"}}
            graph.invoke({"ticker": "TST", "messages": []}, config)

        for p in patches:
            p.stop()

        assert os.path.exists(db_path)
        assert os.path.getsize(db_path) > 0
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 3: get_state — persist edilmiş state okunabilir
# ---------------------------------------------------------------------------

def test_get_state_after_run():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        patches = _mock_patches()
        for p in patches:
            p.start()

        with make_checkpointer(db_path) as cp:
            graph = build_finance_graph(checkpointer=cp)
            config = {"configurable": {"thread_id": "test-get-state"}}
            graph.invoke({"ticker": "TST", "messages": []}, config)

            # Run bittikten sonra state'i oku
            saved_state = graph.get_state(config)

        for p in patches:
            p.stop()

        assert saved_state is not None
        assert saved_state.values["ticker"] == "TST"
        assert saved_state.values["price_data"]["price"] == 100.0
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 4: farklı thread_id'ler izole çalışır
# ---------------------------------------------------------------------------

def test_thread_isolation():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        patches = _mock_patches()
        for p in patches:
            p.start()

        with make_checkpointer(db_path) as cp:
            graph = build_finance_graph(checkpointer=cp)

            config_a = {"configurable": {"thread_id": "thread-NVDA"}}
            config_b = {"configurable": {"thread_id": "thread-TSLA"}}

            graph.invoke({"ticker": "NVDA", "messages": []}, config_a)
            graph.invoke({"ticker": "TSLA", "messages": []}, config_b)

            state_a = graph.get_state(config_a)
            state_b = graph.get_state(config_b)

        for p in patches:
            p.stop()

        assert state_a.values["ticker"] == "NVDA"
        assert state_b.values["ticker"] == "TSLA"
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 5: checkpoint history — kaç adım kaydedildi?
# ---------------------------------------------------------------------------

def test_checkpoint_history_has_multiple_steps():
    """
    LangGraph her node sonrası bir checkpoint yazar.
    5 node'lu graph → en az 5 checkpoint adımı beklenir.
    Debug için: get_state_history() ile her adımın state'ini inceleyebilirsin.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        patches = _mock_patches()
        for p in patches:
            p.start()

        with make_checkpointer(db_path) as cp:
            graph = build_finance_graph(checkpointer=cp)
            config = {"configurable": {"thread_id": "test-history"}}
            graph.invoke({"ticker": "TST", "messages": []}, config)

            history = list(graph.get_state_history(config))

        for p in patches:
            p.stop()

        # Her node sonrası + başlangıç = en az 5 checkpoint
        assert len(history) >= 5, f"Sadece {len(history)} checkpoint bulundu, en az 5 bekleniyor"
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 6: checkpointer olmadan graph eskisi gibi çalışır (geriye dönük uyum)
# ---------------------------------------------------------------------------

def test_graph_without_checkpointer_still_works():
    patches = _mock_patches()
    for p in patches:
        p.start()

    graph = build_finance_graph()  # checkpointer=None (varsayılan)
    result = graph.invoke({"ticker": "TST", "messages": []})

    for p in patches:
        p.stop()

    assert result["ticker"] == "TST"
    assert result["price_data"]["price"] == 100.0
