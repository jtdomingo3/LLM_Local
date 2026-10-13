"""
Unit tests for ResponseCache (in-memory LRU cache with TTL).
"""

import time
import pytest
from app.services.response_cache import ResponseCache


def test_cache_miss():
    """Verify get returns None on empty cache and increments misses."""
    cache = ResponseCache(max_size=10, ttl_seconds=60, enabled=True)
    result = cache.get(model="phi4-mini", temperature=0.7, payload="Hello")
    assert result is None
    stats = cache.get_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 0


def test_cache_hit():
    """Verify get returns stored response and tokens on hit."""
    cache = ResponseCache(max_size=10, ttl_seconds=60, enabled=True)
    cache.set(
        model="phi4-mini",
        temperature=0.7,
        payload="What is 2+2?",
        response="4",
        token_usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    )

    result = cache.get(model="phi4-mini", temperature=0.7, payload="What is 2+2?")
    assert result is not None
    text, usage = result
    assert text == "4"
    assert usage["total_tokens"] == 6

    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["size"] == 1


def test_cache_ttl_expiration():
    """Verify expired cache entries return None and are evicted."""
    cache = ResponseCache(max_size=10, ttl_seconds=0.1, enabled=True)
    cache.set(
        model="phi4-mini",
        temperature=0.7,
        payload="Expire me",
        response="Will expire",
    )

    time.sleep(0.15)
    result = cache.get(model="phi4-mini", temperature=0.7, payload="Expire me")
    assert result is None
    stats = cache.get_stats()
    assert stats["evictions"] == 1
    assert stats["size"] == 0


def test_cache_lru_eviction():
    """Verify least recently used entry is evicted when max_size is reached."""
    cache = ResponseCache(max_size=2, ttl_seconds=60, enabled=True)
    cache.set("model", 0.5, "prompt1", "resp1")
    cache.set("model", 0.5, "prompt2", "resp2")

    # Access prompt1 so prompt2 becomes the LRU item
    cache.get("model", 0.5, "prompt1")

    # Insert prompt3, should evict prompt2
    cache.set("model", 0.5, "prompt3", "resp3")

    assert cache.get("model", 0.5, "prompt1") is not None
    assert cache.get("model", 0.5, "prompt2") is None
    assert cache.get("model", 0.5, "prompt3") is not None
    assert cache.get_stats()["evictions"] == 1


def test_cache_disabled():
    """Verify disabled cache does not store or return entries."""
    cache = ResponseCache(max_size=10, ttl_seconds=60, enabled=False)
    cache.set("model", 0.5, "prompt", "resp")
    assert cache.get("model", 0.5, "prompt") is None
    assert cache.get_stats()["size"] == 0


def test_cache_clear():
    """Verify clearing cache empties all entries."""
    cache = ResponseCache(max_size=10, ttl_seconds=60, enabled=True)
    cache.set("model", 0.5, "p1", "r1")
    cache.set("model", 0.5, "p2", "r2")
    assert cache.get_stats()["size"] == 2

    cache.clear()
    assert cache.get_stats()["size"] == 0
