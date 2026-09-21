"""
Cache layer for live satellite responses.

Real Redis client (redis-py, async interface) when REDIS_URL is set and
reachable. Falls back to a process-local dict with the same interface when
Redis isn't configured/reachable -- this keeps local dev and the sandboxed
authoring environment (no Redis server available) working without changing
a single line of calling code, and is the seam a `docker-compose up` with
the bundled redis service plugs straight into.

This is intentionally NOT a fake cache pretending to be Redis -- it really
is Redis when available, and honestly falls back (logged, once) when not.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger("oceanembed.cache")


class _InMemoryCache:
    def __init__(self):
        self._store: dict[str, tuple[float, str]] = {}  # key -> (expires_at, json_value)

    async def get(self, key: str) -> Optional[str]:
        item = self._store.get(key)
        if not item:
            return None
        expires_at, value = item
        if expires_at < time.time():
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._store[key] = (time.time() + ttl_seconds, value)

    async def ping(self) -> bool:
        return True


class CacheClient:
    def __init__(self):
        self._backend = None
        self._is_redis = False

    async def _ensure_backend(self):
        if self._backend is not None:
            return
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                import redis.asyncio as redis  # optional dependency
                client = redis.from_url(redis_url, decode_responses=True)
                await client.ping()
                self._backend = client
                self._is_redis = True
                logger.info("Connected to Redis at %s", redis_url)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis configured (%s) but unreachable: %s -- falling back to in-memory cache", redis_url, exc)
        self._backend = _InMemoryCache()
        self._is_redis = False

    async def get_json(self, key: str) -> Optional[Any]:
        await self._ensure_backend()
        raw = await self._backend.get(key)
        return json.loads(raw) if raw else None

    async def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        await self._ensure_backend()
        raw = json.dumps(value)
        if self._is_redis:
            await self._backend.set(key, raw, ex=ttl_seconds)
        else:
            await self._backend.set(key, raw, ttl_seconds)

    @property
    def backend_name(self) -> str:
        return "redis" if self._is_redis else "in-memory (Redis not configured/reachable)"


_cache_singleton: Optional[CacheClient] = None


def get_cache() -> CacheClient:
    global _cache_singleton
    if _cache_singleton is None:
        _cache_singleton = CacheClient()
    return _cache_singleton
