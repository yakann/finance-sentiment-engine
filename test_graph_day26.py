"""
Day 26 — Human-in-the-Loop (HITL) Testleri

Test senaryoları:
1. interrupt_after=["draft"] ile graf derlenir
2. Graf draft node'unda durur (interrupt çalışır, draft alanı yazılır)
3. Approve flow: invoke(None, config) ile graf END'e ulaşır
4. Reject + feedback: update_state sonrası revise node çalışır
5. Revise sonrası draft güncellenir (feedback metni içerir)
6. Birden fazla feedback turu (iki ardışık revise) çalışır
7. interrupt_after olmadan graf normal çalışır (geriye dönük uyum)
"""

import os
import tempfile
import pytest
from unittest.mock import patch

from graph.checkpointer import make_checkpointer
from graph.finance_graph import build_finance_graph

# ---------------------------------------------------------------------------
# Yardımcı sabitler ve mock'lar
# ---------------------------------------------------------------------------

_MOCK_NEWS_BULLISH = [
    {"ticker": "TST", "sentiment": "bullish", "urgency": "high",
     "key_event": "earnings", "summary": "Strong Q4 earnings beat"},
    {"ticker": "TST", "sentiment": "bullish", "urgency": "medium",
     "key_event": "product_event", "summary": "New product announced"},
    {"ticker": "TST", "sentiment": "neutral", "urgency": "low",
     "key_event": "market_dynamics", "summary": "Market sideways"},
]

_MOCK_NEWS_NEUTRAL = [
    {"ticker": "TST", "sentiment": "neutral", "urgency": "low",
     "key_event": "market_dynamics", "summary": "Sideways movement"},
]

_MOCK_PRICE = {
    "ticker": "TST", "price": 150.0, "pct_change": 2.5,
    "volume": 2_000_000, "market_cap": 750_000_000,
}

_MOCK_10K = {"answer": "Key risks include supply chain and competition.", "sources": []}


def _mock_patches(news=None):
    """Network çağrılarını mock'la. Varsayılan: bullish haber."""
    news = news or _MOCK_NEWS_BULLISH
    return [
        patch("graph.finance_graph._analyze_news_sentiment", return_value=news),
        patch("graph.finance_graph._get_stock_data", return_value=_MOCK_PRICE),
        patch("graph.finance_graph._query_10k", return_value=_MOCK_10K),
    ]


def _start_patches(patches):
    for p in patches:
        p.start()


def _stop_patches(patches):
    for p in patches:
        p.stop()


# ---------------------------------------------------------------------------
# Test 1: interrupt_after=["draft"] ile graf derlenir
# ---------------------------------------------------------------------------

def test_graph_compiles_with_interrupt_after_draft():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        with make_checkpointer(db_path) as cp:
            graph = build_finance_graph(checkpointer=cp, interrupt_after=["draft"])
            assert graph is not None
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 2: Graf draft node'unda durur (interrupt çalışır)
# ---------------------------------------------------------------------------

def test_graph_stops_at_draft_interrupt():
    """
    interrupt_after=["draft"] ile çalışınca graf draft node'unu çalıştırır
    ve orada durur. Sonuçta 'draft' alanı yazılmış olmalı.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        patches = _mock_patches()
        _start_patches(patches)

        with make_checkpointer(db_path) as cp:
            graph = build_finance_graph(checkpointer=cp, interrupt_after=["draft"])
            config = {"configurable": {"thread_id": "test-interrupt"}}
            result = graph.invoke({"ticker": "TST", "messages": []}, config)

            # draft node çalıştı ve draft alanı yazıldı
            assert result.get("draft"), "draft alanı boş olmamalı"
            assert "TST" in result["draft"], "draft içinde ticker bulunmalı"

            # Graf hâlâ paused durumda → checkpoint var
            state = graph.get_state(config)
            assert state is not None
            assert state.values.get("draft")

        _stop_patches(patches)
    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 3: Approve flow — invoke(None, config) ile END'e ulaşır
# ---------------------------------------------------------------------------

def test_approve_resumes_to_end():
    """
    Interrupt sonrası invoke(None, config) çağrısı (feedback yok = approve)
    grafı END'e götürür, hata vermez.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        patches = _mock_patches()
        _start_patches(patches)

        with make_checkpointer(db_path) as cp:
            graph = build_finance_graph(checkpointer=cp, interrupt_after=["draft"])
            config = {"configurable": {"thread_id": "test-approve"}}

            # İlk run → interrupt
            graph.invoke({"ticker": "TST", "messages": []}, config)

            # Approve (feedback yok) → resume → END
            final = graph.invoke(None, config)

        _stop_patches(patches)

        # Graf tamamlandı, draft hâlâ mevcut
        assert final.get("draft")

    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 4: Reject + feedback — revise node çalışır
# ---------------------------------------------------------------------------

def test_reject_with_feedback_triggers_revise():
    """
    update_state ile feedback set edilince resume sonrası revise node çalışır
    ve draft güncellenir.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        patches = _mock_patches()
        _start_patches(patches)

        with make_checkpointer(db_path) as cp:
            graph = build_finance_graph(checkpointer=cp, interrupt_after=["draft"])
            config = {"configurable": {"thread_id": "test-reject"}}

            # İlk run → interrupt
            result = graph.invoke({"ticker": "TST", "messages": []}, config)
            original_draft = result["draft"]

            # Reject: feedback ekle
            graph.update_state(config, {"feedback": "Add more risk detail"})

            # Resume → revise çalışır → draft interrupt
            revised_result = graph.invoke(None, config)

        _stop_patches(patches)

        revised_draft = revised_result["draft"]
        assert revised_draft != original_draft, "revise sonrası draft değişmeli"
        assert "Add more risk detail" in revised_draft, "feedback draft'a yansımalı"

    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 5: Revise sonrası feedback temizlenir
# ---------------------------------------------------------------------------

def test_feedback_cleared_after_revise():
    """
    revise node çalıştıktan sonra feedback alanı "" olarak sıfırlanmalı.
    Bu, sonsuz revise döngüsünü önler.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        patches = _mock_patches()
        _start_patches(patches)

        with make_checkpointer(db_path) as cp:
            graph = build_finance_graph(checkpointer=cp, interrupt_after=["draft"])
            config = {"configurable": {"thread_id": "test-feedback-clear"}}

            graph.invoke({"ticker": "TST", "messages": []}, config)
            graph.update_state(config, {"feedback": "needs revision"})
            graph.invoke(None, config)

            # revise çalıştı, şimdi state'i kontrol et
            state = graph.get_state(config)

        _stop_patches(patches)

        feedback_val = state.values.get("feedback", "")
        assert feedback_val == "", f"feedback sıfırlanmalı, bulundu: '{feedback_val}'"

    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 6: Birden fazla feedback turu çalışır
# ---------------------------------------------------------------------------

def test_multiple_feedback_rounds():
    """
    İki ardışık reject → revise döngüsü hatasız çalışır ve her seferinde
    draft güncellenir.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        patches = _mock_patches()
        _start_patches(patches)

        with make_checkpointer(db_path) as cp:
            graph = build_finance_graph(checkpointer=cp, interrupt_after=["draft"])
            config = {"configurable": {"thread_id": "test-multi-feedback"}}

            # İlk run → interrupt
            r1 = graph.invoke({"ticker": "TST", "messages": []}, config)

            # 1. Feedback
            graph.update_state(config, {"feedback": "First feedback"})
            r2 = graph.invoke(None, config)

            # 2. Feedback
            graph.update_state(config, {"feedback": "Second feedback"})
            r3 = graph.invoke(None, config)

            # Final approve → END
            graph.invoke(None, config)

        _stop_patches(patches)

        assert "First feedback" in r2["draft"]
        assert "Second feedback" in r3["draft"]
        # Her tur farklı içerik
        assert r1["draft"] != r2["draft"]
        assert r2["draft"] != r3["draft"]

    finally:
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test 7: interrupt olmadan normal çalışır (geriye dönük uyum — Day 25)
# ---------------------------------------------------------------------------

def test_graph_without_interrupt_runs_to_end():
    """
    interrupt_after=None (varsayılan) → graf tek invoke'da END'e ulaşır.
    Day 25 davranışı bozulmadı.
    """
    patches = _mock_patches(news=_MOCK_NEWS_NEUTRAL)
    _start_patches(patches)

    graph = build_finance_graph()  # checkpointer yok, interrupt yok
    result = graph.invoke({"ticker": "TST", "messages": []})

    _stop_patches(patches)

    assert result["ticker"] == "TST"
    assert result["price_data"]["price"] == 150.0
    assert result.get("draft"), "draft alanı yazılmış olmalı"
