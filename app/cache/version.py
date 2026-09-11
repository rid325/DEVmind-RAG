"""Corpus generations for the single-worker API; data-changing jobs suspend caching."""
from contextlib import contextmanager
from threading import RLock
from time import time_ns

from redis.exceptions import RedisError

_lock = RLock()
_dirty = True  # Also invalidate on API restart: corpus/model changes may have happened offline.
_active_updates = 0
VERSION_KEY = "devmind:corpus:version"


def get_current_version() -> str | None:
    from app.cache.redis_cache import get_redis, cache_error
    global _dirty
    with _lock:
        if _active_updates:
            return None
        client = get_redis()
        if client is None:
            return None
        try:
            # A time-based initial value avoids reusing old namespaces if the
            # counter is deleted while response keys survive. It fits Redis int64.
            client.set(VERSION_KEY, time_ns(), nx=True)
            if _dirty:
                value = client.incr(VERSION_KEY)
                _dirty = False
            else:
                value = int(client.get(VERSION_KEY))
            return str(value)
        except (RedisError, ValueError, TypeError) as exc:
            _dirty = True
            cache_error(exc)
            return None


def increment_version() -> str | None:
    global _dirty
    with _lock:
        _dirty = True
        return get_current_version()


@contextmanager
def corpus_update():
    global _active_updates, _dirty
    with _lock:
        _active_updates += 1
        _dirty = True
    try:
        yield
    finally:
        with _lock:
            _active_updates -= 1
            # Failed jobs can have committed earlier batches. Invalidate those
            # too, and retry the version bump on reconnect if Redis is down.
            increment_version()
