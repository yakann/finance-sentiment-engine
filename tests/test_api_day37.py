"""
Day 37 Tests — SSE streaming endpoint (POST /stream)

Test edilen özellikler:
    1. _sse(): geçerli SSE çerçevesi (event + JSON data + boş satır)
    2. _chunk_tokens(): parçalar birleşince orijinal metin geri geliyor
    3. _event_stream(): node lifecycle → tool_start/tool_end + token + final
    4. POST /stream: TestClient ile uçtan uca SSE akışı (mocked graph)
    5. Pipeline hatası → error event'i yayınlanıyor

Gerçek LLM/ağ çağrısı yok: build_finance_graph patch'lenerek deterministik
bir `astream_events` event dizisi üreten sahte graf enjekte edilir.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import _chunk_tokens, _event_stream, _sse, app


def _collect(ticker: str) -> list[str]:
    """_event_stream async generator'ını çalıştırıp tüm SSE çerçevelerini toplar.

    pytest-asyncio bağımlılığı eklememek için basit bir asyncio.run sarmalayıcı.
    """
    async def _run() -> list[str]:
        return [frame async for frame in _event_stream(ticker)]

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Sahte graf: astream_events'in ürettiği event akışını taklit eder
# ---------------------------------------------------------------------------

_FAKE_DRAFT = "[DRAFT] TSLA Investment Brief\nSentiment: BULLISH"


def _fake_events():
    """build_finance_graph().astream_events(...) çıktısını taklit eden olay listesi."""
    return [
        {"event": "on_chain_start", "name": "collect_news", "data": {}},
        {"event": "on_chain_end", "name": "collect_news", "data": {"output": {"news": []}}},
        {"event": "on_chain_start", "name": "RunnableSequence", "data": {}},  # süzülmeli
        {"event": "on_chain_start", "name": "analyze_sentiment", "data": {}},
        {"event": "on_chain_end", "name": "analyze_sentiment", "data": {"output": {}}},
        {"event": "on_chain_start", "name": "fetch_price", "data": {}},
        {"event": "on_chain_end", "name": "fetch_price", "data": {"output": {}}},
        {"event": "on_chain_start", "name": "draft", "data": {}},
        {"event": "on_chain_end", "name": "draft", "data": {"output": {"draft": _FAKE_DRAFT}}},
    ]


class _FakeGraph:
    def __init__(self, events=None, raise_exc=None):
        self._events = events if events is not None else _fake_events()
        self._raise = raise_exc

    async def astream_events(self, _input, version="v2"):  # noqa: D401
        for ev in self._events:
            yield ev
        if self._raise:
            raise self._raise


def _patch_graph(graph):
    """build_finance_graph'i sahte graf döndürecek şekilde patch'ler."""
    return patch("graph.finance_graph.build_finance_graph", return_value=graph)


# ---------------------------------------------------------------------------
# Test 1: _sse formatı
# ---------------------------------------------------------------------------

def test_sse_frame_format():
    frame = _sse("token", {"text": "hi"})
    assert frame.startswith("event: token\n")
    assert 'data: {"text": "hi"}' in frame
    assert frame.endswith("\n\n")


# ---------------------------------------------------------------------------
# Test 2: _chunk_tokens roundtrip
# ---------------------------------------------------------------------------

def test_chunk_tokens_roundtrip():
    text = "[DRAFT] TSLA\nSentiment: BULLISH — 3 bullish"
    pieces = _chunk_tokens(text)
    assert len(pieces) > 1
    assert "".join(pieces) == text  # boşluk/newline kaybı yok


def test_chunk_tokens_empty():
    assert _chunk_tokens("") == []


# ---------------------------------------------------------------------------
# Test 3: _event_stream olay sırası (token gecikmesini kapatarak)
# ---------------------------------------------------------------------------

def test_event_stream_emits_lifecycle_and_tokens():
    with _patch_graph(_FakeGraph()), patch("api.main._TOKEN_DELAY_S", 0):
        frames = _collect("TSLA")

    joined = "".join(frames)

    # node lifecycle event'leri
    assert "event: tool_start\n" in joined
    assert "event: tool_end\n" in joined
    assert '"node": "collect_news"' in joined
    assert '"node": "draft"' in joined

    # süzülen iç Runnable yayınlanmamalı
    assert "RunnableSequence" not in joined

    # typewriter token'ları + final
    assert "event: token\n" in joined
    assert "event: final\n" in joined

    # token'lar birleşince final draft geri gelmeli
    token_texts = [
        __import__("json").loads(f.split("data: ", 1)[1])["text"]
        for f in frames
        if f.startswith("event: token\n")
    ]
    assert "".join(token_texts) == _FAKE_DRAFT


# ---------------------------------------------------------------------------
# Test 4: hata yolu → error event'i
# ---------------------------------------------------------------------------

def test_event_stream_emits_error_on_failure():
    boom = _FakeGraph(events=[], raise_exc=RuntimeError("kaboom"))
    with _patch_graph(boom), patch("api.main._TOKEN_DELAY_S", 0):
        frames = _collect("TSLA")

    joined = "".join(frames)
    assert "event: error\n" in joined
    assert "kaboom" in joined


# ---------------------------------------------------------------------------
# Test 5: POST /stream uçtan uca (TestClient)
# ---------------------------------------------------------------------------

def test_stream_endpoint_e2e():
    with _patch_graph(_FakeGraph()), patch("api.main._TOKEN_DELAY_S", 0):
        client = TestClient(app)
        resp = client.post("/stream", json={"ticker": "tsla"})

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        body = resp.text
        assert "event: tool_start\n" in body
        assert "event: final\n" in body
        # ticker büyük harfe çevrildi
        assert '"ticker": "TSLA"' in body
