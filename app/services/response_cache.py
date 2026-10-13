"""
Response Cache — Thread-safe in-memory LRU cache with TTL for LLM completions.

Eliminates redundant GPU inference and VRAM pressure for identical queries
(e.g., repeating prompts, automated status checks, repeated code queries).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class CacheEntry:
    """Represents a single cached LLM response."""

    __slots__ = ("response", "created_at", "last_accessed", "token_usage")

    def __init__(
        self,
        response: str,
        token_usage: dict[str, int] | None = None,
        created_at: float | None = None,
    ) -> None:
        self.response = response
        self.token_usage = token_usage or {}
        now = time.time()
        self.created_at = created_at or now
        self.last_accessed = now


class ResponseCache:
    """
    Thread-safe, TTL-aware LRU response cache for LLM queries.
    Keyed by SHA-256 hash of (model, temperature, prompt/messages).
    """

    def __init__(
        self,
        max_size: int = 256,
        ttl_seconds: int = 3600,
        enabled: bool = True,
    ) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @staticmethod
    def generate_key(model: str, temperature: float, payload: Any) -> str:
        """Create a deterministic SHA-256 hash key for a request."""
        normalized_temp = round(float(temperature), 2)
        if isinstance(payload, str):
            payload_repr = payload.strip()
        else:
            try:
                payload_repr = json.dumps(payload, sort_keys=True)
            except Exception:
                payload_repr = str(payload)

        raw_str = f"{model}:{normalized_temp}:{payload_repr}"
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()

    def get(
        self,
        model: str,
        temperature: float,
        payload: Any,
    ) -> tuple[str, dict[str, int]] | None:
        """
        Retrieve a cached response if valid and not expired.

        Returns:
            Tuple of (response_text, token_usage_dict) or None
        """
        if not self.enabled:
            return None

        key = self.generate_key(model, temperature, payload)
        now = time.time()

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            # Check expiration
            if now - entry.created_at > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                self._evictions += 1
                logger.debug("Cache entry expired for key %s", key[:8])
                return None

            # Mark as recently used (move to end in OrderedDict)
            entry.last_accessed = now
            self._cache.move_to_end(key)
            self._hits += 1
            logger.info("Response cache HIT for model=%s (key=%s...)", model, key[:8])
            return entry.response, entry.token_usage

    def set(
        self,
        model: str,
        temperature: float,
        payload: Any,
        response: str,
        token_usage: dict[str, int] | None = None,
    ) -> None:
        """Store a response in cache, evicting oldest item if max_size exceeded."""
        if not self.enabled:
            return

        key = self.generate_key(model, temperature, payload)

        with self._lock:
            # If key already exists, update and move to end
            if key in self._cache:
                self._cache[key] = CacheEntry(response, token_usage)
                self._cache.move_to_end(key)
                return

            # Evict LRU item if capacity is reached
            while len(self._cache) >= self.max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug("Evicted LRU cache entry %s...", oldest_key[:8])

            self._cache[key] = CacheEntry(response, token_usage)
            logger.debug("Stored response in cache (key=%s...)", key[:8])

    def clear(self) -> None:
        """Purge all entries from cache."""
        with self._lock:
            self._cache.clear()
            logger.info("Response cache cleared.")

    def get_stats(self) -> dict[str, Any]:
        """Return cache health and performance statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total > 0 else 0.0
            return {
                "enabled": self.enabled,
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate_percent": round(hit_rate * 100, 1),
                "ttl_seconds": self.ttl_seconds,
            }


# Singleton cache instance
_response_cache: ResponseCache | None = None


def get_response_cache(settings: Settings | None = None) -> ResponseCache:
    """Get the singleton ResponseCache instance."""
    global _response_cache
    if _response_cache is None:
        cfg = settings or get_settings()
        _response_cache = ResponseCache(
            max_size=cfg.response_cache_max_size,
            ttl_seconds=cfg.response_cache_ttl_seconds,
            enabled=cfg.response_cache_enabled,
        )
    return _response_cache
