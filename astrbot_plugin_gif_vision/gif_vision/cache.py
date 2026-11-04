"""Small in-process TTL/LRU cache used in front of the SQLite library."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    size: int = 0


class L1Cache(Generic[T]):
    def __init__(self, ttl_sec: float = 120.0, max_entries: int = 128) -> None:
        self.ttl_sec = max(1.0, float(ttl_sec))
        self.max_entries = max(1, int(max_entries))
        self._values: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._stats = CacheStats()

    def get(self, key: str) -> T | None:
        item = self._values.get(key)
        if item is None:
            self._stats.misses += 1
            return None
        expires_at, value = item
        if expires_at < time.time():
            self._values.pop(key, None)
            self._stats.misses += 1
            return None
        self._values.move_to_end(key)
        self._stats.hits += 1
        return value

    def put(self, key: str, value: T) -> None:
        self._values[key] = (time.time() + self.ttl_sec, value)
        self._values.move_to_end(key)
        self._stats.stores += 1
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)
            self._stats.evictions += 1

    def clear(self) -> None:
        self._values.clear()

    def stats(self) -> CacheStats:
        result = CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            stores=self._stats.stores,
            evictions=self._stats.evictions,
            size=len(self._values),
        )
        return result


__all__ = ["CacheStats", "L1Cache"]
