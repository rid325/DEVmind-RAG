"""Query endpoint integration. Experiments call the generator directly and bypass it."""
import json
import logging
from time import perf_counter

from pydantic import ValidationError

from app.cache import redis_cache as cache
from app.embeddings import get_client
from app.generation.generator import QueryResponse
from app.models import QueryLog

logger = logging.getLogger(__name__)


def lookup_cached_answer(request, db, start):
    if not request.use_cache:
        return None, None, None
    config = {name: getattr(request, name) for name in ("use_hyde", "use_reranking", "use_expansion")}
    scope = cache.cache_scope(config)
    if scope is None:
        return None, None, None

    def restore(payload, tier):
        if payload is None or not cache.scope_is_current(scope):
            return None
        try:
            json.dumps(payload, allow_nan=False)
            result = QueryResponse.model_validate(payload)
        except (ValidationError, ValueError, TypeError):
            return None
        if result.log_id is None:
            return None
        log = db.get(QueryLog, result.log_id)
        if log is None:
            return None
        result.cache_hit_query = result.cache_hit_query or result.query
        result.query = request.query
        result.cache = tier
        # The cache may have been written while the judge was still pending.
        result.faithfulness_score = log.faithfulness_score
        result.latency_ms = round((perf_counter() - start) * 1000, 2)
        return result

    exact = restore(cache.get_exact(request.query, scope=scope), "exact")
    if exact is not None:
        return exact, scope, None
    if cache.get_redis() is None:
        return None, None, None
    try:
        response = get_client().with_options(timeout=3.0, max_retries=0).embeddings.create(
            input=[cache.normalize_query(request.query)], model="text-embedding-3-small",
        )
        if len(response.data) != 1 or response.data[0].index != 0:
            raise ValueError("Invalid cache embedding response")
        embedding = response.data[0].embedding
        cache.serialize_embedding(embedding)
    except Exception as exc:
        # This extra embedding is optional; generation has its own retrieval path.
        logger.warning("Semantic cache lookup skipped: %s", type(exc).__name__)
        return None, scope, None
    semantic = restore(cache.get_semantic(embedding, query=request.query, scope=scope), "semantic")
    return semantic, scope, embedding


def store_cached_answer(request, result, scope, embedding):
    if scope is None:
        return
    payload = result.model_dump(mode="json")
    cache.set_exact(request.query, payload, scope=scope)
    if embedding is not None:
        cache.set_semantic(request.query, embedding, payload, scope=scope)
