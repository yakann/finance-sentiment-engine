from __future__ import annotations


def get_cache():
    """Return a SemanticCache instance if Redis is reachable, else None."""
    try:
        from cache.semantic_cache import SemanticCache
        sc = SemanticCache()
        if sc.ping():
            return sc
    except Exception:
        pass
    return None
