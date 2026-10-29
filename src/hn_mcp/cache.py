"""Minimal async TTL cache.

Two TTL classes matter for HN:

  * items  - effectively immutable once a story settles, so cache for long.
  * feeds  - the front page reorders constantly, so cache for ~60s.

A long item TTL is what makes fan-out affordable: the second agent call that
touches an overlapping feed costs almost nothing upstream.
"""

import asyncio
import time
from typing import Any


class TTLCache:
    def __init__(self, maxsize: int = 4096) -> None:
        self._data: dict[str, tuple[float, Any]] = {}
        self._maxsize = maxsize
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            expires_at, value = entry
            if expires_at < time.monotonic():
                del self._data[key]
                self.misses += 1
                return None
            self.hits += 1
            return value

    async def set(self, key: str, value: Any, ttl: float) -> None:
        async with self._lock:
            if len(self._data) >= self._maxsize:
                # Cheap eviction: drop whatever expires soonest.
                oldest = min(self._data, key=lambda k: self._data[k][0])
                del self._data[oldest]
            self._data[key] = (time.monotonic() + ttl, value)

    def stats(self) -> dict[str, int]:
        return {"size": len(self._data), "hits": self.hits, "misses": self.misses}
