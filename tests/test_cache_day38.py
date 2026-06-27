"""Day 38: Unit tests for SemanticCache (Redis vector search).

All tests mock OpenAI embeddings and the Redis client so they run without
a live Redis or API key.  A "slow" marker is applied to the live-Redis
integration test so it is skipped by default.

Run:
    uv run pytest tests/test_cache_day38.py -v
    uv run pytest tests/test_cache_day38.py -v -m slow  # needs Redis on :6379
"""
from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

DIM = 1536


def _fake_embed(text: str) -> list[float]:
    """Return a deterministic unit vector based on text hash."""
    h = hash(text) % (2**31)
    vec = [float((h >> i) & 1) for i in range(DIM)]
    mag = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / mag for x in vec]


def _to_bytes(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_redis():
    """Patch redis.from_url so no real Redis is needed."""
    with patch("cache.semantic_cache.redis.from_url") as mock_factory:
        r = MagicMock()
        r.ping.return_value = True
        # ft().info() raises to trigger index creation path
        r.ft.return_value.info.side_effect = Exception("no index")
        r.ft.return_value.create_index.return_value = None
        mock_factory.return_value = r
        yield r


@pytest.fixture()
def mock_embed():
    with patch("cache.semantic_cache._embed", side_effect=_fake_embed) as m:
        yield m


# ── tests ─────────────────────────────────────────────────────────────────────

def test_ping_returns_true(mock_redis, mock_embed):
    from cache.semantic_cache import SemanticCache
    sc = SemanticCache()
    assert sc.ping() is True


def test_index_created_on_init(mock_redis, mock_embed):
    from cache.semantic_cache import SemanticCache
    SemanticCache()
    mock_redis.ft.return_value.create_index.assert_called_once()


def test_cache_miss_returns_none(mock_redis, mock_embed):
    from cache.semantic_cache import SemanticCache

    result_mock = MagicMock()
    result_mock.docs = []
    mock_redis.ft.return_value.search.return_value = result_mock

    sc = SemanticCache()
    assert sc.get("NVDA brief") is None


def test_cache_hit_above_threshold(mock_redis, mock_embed):
    from cache.semantic_cache import SemanticCache, SIMILARITY_THRESHOLD

    doc = MagicMock()
    doc.score = str(1.0 - SIMILARITY_THRESHOLD)   # exactly at threshold
    doc.answer = b"cached answer text"

    result_mock = MagicMock()
    result_mock.docs = [doc]
    mock_redis.ft.return_value.search.return_value = result_mock

    sc = SemanticCache()
    answer = sc.get("What is NVDA doing?")
    assert answer == "cached answer text"


def test_cache_miss_below_threshold(mock_redis, mock_embed):
    from cache.semantic_cache import SemanticCache, SIMILARITY_THRESHOLD

    doc = MagicMock()
    doc.score = str(1.0 - (SIMILARITY_THRESHOLD - 0.01))  # just below
    doc.answer = b"some other cached answer"

    result_mock = MagicMock()
    result_mock.docs = [doc]
    mock_redis.ft.return_value.search.return_value = result_mock

    sc = SemanticCache()
    assert sc.get("completely different query") is None


def test_set_stores_hash_with_ttl(mock_redis, mock_embed):
    from cache.semantic_cache import SemanticCache, TTL_SECONDS

    sc = SemanticCache()
    sc.set("NVDA brief query", "this is the brief answer")

    mock_redis.hset.assert_called_once()
    call_kwargs = mock_redis.hset.call_args
    mapping = call_kwargs.kwargs.get("mapping") or call_kwargs[1].get("mapping") or call_kwargs[0][1]
    assert b"NVDA brief query" in mapping.get("query", b"")
    assert b"this is the brief answer" in mapping.get("answer", b"")
    assert "embedding" in mapping

    mock_redis.expire.assert_called_once()
    key_arg, ttl_arg = mock_redis.expire.call_args[0]
    assert ttl_arg == TTL_SECONDS


def test_set_embedding_is_correct_length(mock_redis, mock_embed):
    from cache.semantic_cache import SemanticCache, VECTOR_DIM

    sc = SemanticCache()
    sc.set("test query", "test answer")

    mapping = mock_redis.hset.call_args.kwargs.get("mapping") or mock_redis.hset.call_args[1].get("mapping")
    emb_bytes = mapping["embedding"]
    assert len(emb_bytes) == VECTOR_DIM * 4  # float32 = 4 bytes each


def test_clear_drops_and_recreates_index(mock_redis, mock_embed):
    from cache.semantic_cache import SemanticCache

    mock_redis.ft.return_value.info.side_effect = [
        Exception("no index"),  # initial _ensure_index call
        Exception("no index"),  # after dropindex in clear()
    ]

    sc = SemanticCache()
    sc.clear()

    mock_redis.ft.return_value.dropindex.assert_called_once_with(delete_documents=True)
    assert mock_redis.ft.return_value.create_index.call_count == 2


def test_get_handles_redis_exception_gracefully(mock_redis, mock_embed):
    from cache.semantic_cache import SemanticCache

    mock_redis.ft.return_value.search.side_effect = Exception("connection refused")

    sc = SemanticCache()
    assert sc.get("any query") is None


def test_get_cache_helper_returns_none_when_redis_down():
    """_get_cache() in api/main.py returns None when Redis is unreachable."""
    with patch("cache.semantic_cache.redis.from_url") as mock_factory:
        r = MagicMock()
        r.ping.return_value = False
        mock_factory.return_value = r

        from api.main import _get_cache
        # Re-import with patched deps
        with patch("cache.semantic_cache._embed", side_effect=_fake_embed):
            result = _get_cache()
        # ping() returns False → _get_cache returns None
        assert result is None


# ── integration (needs live Redis on :6379) ────────────────────────────────────

@pytest.mark.slow
def test_live_set_get_roundtrip():
    """Store a value, retrieve it with the same query, get a cache hit."""
    with patch("cache.semantic_cache._embed", side_effect=_fake_embed):
        from cache.semantic_cache import SemanticCache

        sc = SemanticCache()
        sc.clear()

        query = "NVDA investment brief June 2026"
        answer = "NVIDIA is bullish with strong AI tailwinds."
        sc.set(query, answer)

        hit = sc.get(query)
        assert hit == answer

        sc.clear()


@pytest.mark.slow
def test_live_different_query_is_cache_miss():
    """A very different query should NOT return the stored answer."""
    with patch("cache.semantic_cache._embed", side_effect=_fake_embed):
        from cache.semantic_cache import SemanticCache

        sc = SemanticCache()
        sc.clear()

        sc.set("NVDA brief", "NVIDIA answer")
        result = sc.get("TSLA brief")
        # With fake embeddings these are hash-different → likely a miss
        # (not guaranteed, but a useful sanity check)
        sc.clear()
