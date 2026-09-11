"""Exact JSON strings and bounded, linearly searched semantic hashes. No pickle."""
from hashlib import md5, sha256
import json
import logging
import os
import re
from time import monotonic

from dotenv import load_dotenv
import numpy as np
import redis
from redis.backoff import NoBackoff
from redis.retry import Retry
from redis.exceptions import RedisError

load_dotenv()
logger = logging.getLogger(__name__)
TTL_SECONDS = 86400
SEMANTIC_THRESHOLD = 0.95
MAX_SEMANTIC_ENTRIES = 1000
EMBEDDING_DIMENSIONS = 1536
_retry_after = 0.0
try:
    _client = redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6380/0"),
        decode_responses=False, max_connections=32,
        socket_connect_timeout=0.15, socket_timeout=0.15,
        retry=Retry(NoBackoff(), 0),
    )
except ValueError:
    _client = None
    logger.warning("Invalid Redis configuration; response caching is disabled")


def cache_error(exc):
    global _retry_after
    logger.warning("Cache bypassed: %s", type(exc).__name__)
    _retry_after = monotonic() + 1


def get_redis():
    if _client is None or monotonic() < _retry_after:
        return None
    try:
        _client.ping()
        return _client
    except RedisError as exc:
        cache_error(exc)
        return None


def normalize_query(query: str) -> str:
    return query.lower().strip()


def cache_scope(config: dict) -> str | None:
    from app.cache.version import get_current_version
    version = get_current_version()
    if version is None:
        return None
    # Schema tag and all behavior flags keep ablations and response versions apart.
    config_hash = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    return f"devmind:cache:v1:{version}:{config_hash}"


def scope_is_current(scope: str | None) -> bool:
    from app.cache.version import get_current_version
    return scope is not None and scope.split(":")[3] == get_current_version()


def exact_key(query: str, scope: str) -> str:
    digest = md5(normalize_query(query).encode(), usedforsecurity=False).hexdigest()
    return f"{scope}:exact:{digest}"


def get_exact(query: str, *, scope: str) -> dict | None:
    client = get_redis()
    if client is None:
        return None
    try:
        raw = client.get(exact_key(query, scope))
        value = json.loads(raw) if raw else None
        return value if isinstance(value, dict) else None
    except (RedisError, ValueError, TypeError, OverflowError) as exc:
        cache_error(exc)
        return None


def set_exact(query: str, response: dict, ttl_seconds: int = TTL_SECONDS, *, scope: str):
    client = get_redis()
    if client is None or not scope_is_current(scope):
        return
    try:
        client.set(exact_key(query, scope), json.dumps(response, allow_nan=False), ex=ttl_seconds)
    except (RedisError, ValueError, TypeError, OverflowError) as exc:
        cache_error(exc)


def serialize_embedding(embedding: list) -> bytes:
    vector = np.asarray(embedding, dtype="<f4")
    if vector.shape != (EMBEDDING_DIMENSIONS,) or not np.isfinite(vector).all() or not np.linalg.norm(vector):
        raise ValueError("Invalid cache embedding")
    return vector.tobytes()


def deserialize_embedding(raw: bytes) -> np.ndarray:
    vector = np.frombuffer(raw, dtype="<f4")
    if vector.shape != (EMBEDDING_DIMENSIONS,) or not np.isfinite(vector).all() or not np.linalg.norm(vector):
        raise ValueError("Invalid cached embedding")
    return vector


def semantic_compatible(query: str, cached_query: str) -> bool:
    # Similar vectors can conceal a changed version number, code literal or
    # negation. These conservative guards do not prove semantic equivalence.
    def details(text):
        text = normalize_query(text)
        numbers = re.findall(r"\d+(?:\.\d+)*", text)
        literals = re.findall(r"`[^`]+`|\"[^\"]+\"|'[^']+'", text)
        negative = bool(re.search(r"\b(?:not|no|never|without|cannot|can't|don't|doesn't|isn't)\b", text))
        return numbers, literals, negative
    return details(query) == details(cached_query)


def get_semantic(query_embedding: list, threshold: float = SEMANTIC_THRESHOLD, *, query: str, scope: str) -> dict | None:
    client = get_redis()
    if client is None:
        return None
    try:
        vector = deserialize_embedding(serialize_embedding(query_embedding))
        vector = vector / np.linalg.norm(vector)
        best, best_score = None, threshold
        # Fetch only this corpus/config namespace, never all generations.
        # If the cap is reached, abstain rather than choose an incomplete best match.
        keys = []
        for key in client.scan_iter(match=f"{scope}:semantic:*", count=100):
            keys.append(key)
            if len(keys) > MAX_SEMANTIC_ENTRIES:
                return None
        for offset in range(0, len(keys), 100):
            pipe = client.pipeline(transaction=False)
            for key in keys[offset:offset + 100]:
                pipe.hgetall(key)
            for entry in pipe.execute():
                try:
                    cached_query = entry[b"query"].decode()
                    if not semantic_compatible(query, cached_query):
                        continue
                    stored = deserialize_embedding(entry[b"embedding"])
                    score = float(np.dot(vector, stored / np.linalg.norm(stored)))
                    if score >= best_score:
                        response = json.loads(entry[b"response"])
                        if not isinstance(response, dict):
                            continue
                        response["cache_hit_query"] = cached_query
                        response["cache_similarity"] = min(1.0, score)
                        best, best_score = response, score
                except (KeyError, ValueError, TypeError, UnicodeError):
                    continue  # Expired or malformed entries are cache misses.
        return best
    except (RedisError, ValueError, TypeError, OverflowError) as exc:
        cache_error(exc)
        return None


def set_semantic(query: str, query_embedding: list, response: dict, ttl_seconds: int = TTL_SECONDS, *, scope: str):
    client = get_redis()
    if client is None or not scope_is_current(scope):
        return
    try:
        # Deterministic key avoids duplicate semantic rows for simultaneous misses.
        key = exact_key(query, scope).replace(":exact:", ":semantic:")
        pipe = client.pipeline(transaction=True)
        pipe.hset(key, mapping={"query": query, "embedding": serialize_embedding(query_embedding),
                                "response": json.dumps(response, allow_nan=False)})
        pipe.expire(key, ttl_seconds)
        pipe.execute()
    except (RedisError, ValueError, TypeError, OverflowError) as exc:
        cache_error(exc)


def cache_stats() -> dict:
    from app.cache.version import get_current_version
    version = get_current_version()
    client = get_redis()
    empty = {"available": False, "corpus_version": version,
             "exact_entries": None, "semantic_entries": None, "redis_db_keys": None}
    if client is None or version is None:
        return empty
    try:
        prefix = f"devmind:cache:v1:{version}:*"
        return {"available": True, "corpus_version": version,
                "exact_entries": len(set(client.scan_iter(match=f"{prefix}:exact:*", count=100))),
                "semantic_entries": len(set(client.scan_iter(match=f"{prefix}:semantic:*", count=100))),
                "redis_db_keys": client.dbsize()}
    except RedisError as exc:
        cache_error(exc)
        return empty
