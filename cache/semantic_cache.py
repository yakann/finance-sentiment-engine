"""Day 38: Redis semantic cache using vector similarity search.

Query embeddings (1536-dim, text-embedding-3-small) are stored as Redis vector
fields. On lookup, the incoming query is embedded and a KNN search returns the
nearest cached entry. If cosine similarity ≥ SIMILARITY_THRESHOLD the cached
answer is returned without calling the agent.

TTL is 1 hour (haberler eskir).

Usage:
    cache = SemanticCache()            # connects to redis://localhost:6379
    hit  = cache.get("NVDA brief")    # None or cached str
    cache.set("NVDA brief", answer)
    cache.clear()                      # flush index + keys (tests)
"""
from __future__ import annotations

import json
import os
import struct
import time
import uuid
from typing import Optional

import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
INDEX_NAME = "semantic_cache_idx"
KEY_PREFIX = "scache:"
VECTOR_DIM = 1536           # text-embedding-3-small
SIMILARITY_THRESHOLD = 0.95
TTL_SECONDS = 3600          # 1 hour


def _embed(text: str) -> list[float]:
    """Return a 1536-dim embedding via OpenAI text-embedding-3-small."""
    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding


def _to_bytes(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class SemanticCache:
    """Redis vector-search backed semantic cache."""

    def __init__(self, redis_url: str = REDIS_URL) -> None:
        self._r = redis.from_url(redis_url, decode_responses=False)
        self._ensure_index()

    # ── Index lifecycle ────────────────────────────────────────────────────────

    def _ensure_index(self) -> None:
        try:
            self._r.ft(INDEX_NAME).info()
        except Exception:
            self._r.ft(INDEX_NAME).create_index(
                fields=[
                    TextField("query"),
                    TextField("answer"),
                    VectorField(
                        "embedding",
                        "FLAT",
                        {
                            "TYPE": "FLOAT32",
                            "DIM": VECTOR_DIM,
                            "DISTANCE_METRIC": "COSINE",
                        },
                    ),
                ],
                definition=IndexDefinition(prefix=[KEY_PREFIX], index_type=IndexType.HASH),
            )

    def clear(self) -> None:
        """Drop the index and delete all cache keys (useful in tests)."""
        try:
            self._r.ft(INDEX_NAME).dropindex(delete_documents=True)
        except Exception:
            pass
        self._ensure_index()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, query: str) -> Optional[str]:
        """Return cached answer if a semantically similar query exists, else None."""
        vec = _embed(query)
        vec_bytes = _to_bytes(vec)

        q = (
            Query("*=>[KNN 1 @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("query", "answer", "score")
            .dialect(2)
        )
        try:
            results = self._r.ft(INDEX_NAME).search(q, query_params={"vec": vec_bytes})
        except Exception:
            return None

        if not results.docs:
            return None

        doc = results.docs[0]
        # COSINE distance in Redis: 0 = identical, 1 = opposite
        # similarity = 1 - distance
        distance = float(doc.score)
        similarity = 1.0 - distance
        if similarity >= SIMILARITY_THRESHOLD:
            return doc.answer.decode() if isinstance(doc.answer, bytes) else doc.answer
        return None

    def set(self, query: str, answer: str) -> None:
        """Store query+answer with a 1-hour TTL."""
        vec = _embed(query)
        key = f"{KEY_PREFIX}{uuid.uuid4().hex}"
        mapping = {
            "query": query.encode(),
            "answer": answer.encode(),
            "embedding": _to_bytes(vec),
            "ts": str(time.time()).encode(),
        }
        self._r.hset(key, mapping=mapping)
        self._r.expire(key, TTL_SECONDS)

    def ping(self) -> bool:
        """Return True if Redis is reachable."""
        try:
            return self._r.ping()
        except Exception:
            return False
