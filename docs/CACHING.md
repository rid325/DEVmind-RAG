# Redis response caching

Day 14 adds an exact cache and an optional semantic lookup ahead of `/query`. Redis is an optimization: a connection failure returns to the normal RAG pipeline. Experiments continue to call the generator directly, so their measurements remain uncached.

## Run locally

```bash
docker compose up -d
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8001
```

Add this to `.env`:

```dotenv
REDIS_URL=redis://localhost:6380/0
```

This project's Docker Redis uses host port **6380** because an existing local Redis service occupies 6379. Redis inside the container still listens on 6379. To choose another host port, set `REDIS_PORT` in `.env` and update `REDIS_URL` to match. Redis is bound to localhost and has a 128 MB memory limit. The existing PostgreSQL volume is unchanged.

The cache uses a shared redis-py client and connection pool, binary responses, 150ms connection/socket timeouts, and no connection retries. After a Redis error it waits one second before probing again. A failed optional semantic embedding call also falls back; that call has a three-second SDK timeout and no retries. See [redis-py connection pooling](https://redis.io/docs/latest/develop/clients/redis-py/connect/).

## Request flow

1. Validate the request and capture the current corpus version and pipeline config.
2. Check the normalized exact key. A valid hit returns without any model calls.
3. Otherwise, embed the normalized original question with `text-embedding-3-small` and search semantic entries in the same namespace.
4. Reuse the best eligible entry with cosine similarity at least **0.95**, or generate a fresh answer with the existing pipeline.
5. On success, commit the normal query log and write the response to both caches. If the optional embedding failed, only the exact cache is written.

An exact key is:

```text
devmind:cache:v1:<corpus_version>:<config_hash>:exact:<query_md5>
```

The query hash uses `query.lower().strip()`. Punctuation and interior whitespace remain significant. The config hash includes `use_hyde`, `use_reranking`, and `use_expansion`, so one configuration cannot reuse another's answer. `v1` is the cache schema tag. MD5 is used for key naming, not as a security control.

The semantic key uses the same normalized query digest with `:semantic:` instead of `:exact:`. A deterministic key avoids duplicate entries for simultaneous identical misses. Each Redis HASH contains the original query, its embedding, and the JSON response. Redis STRING values hold exact responses. Both have a fixed 24-hour TTL; reads do not extend it. The semantic HASH and its expiry are written in one transaction.

Embeddings are 1,536 little-endian float32 values: 6,144 bytes via `numpy.tobytes()` and `numpy.frombuffer()`. Shape, finite values, and a nonzero norm are checked before use. Responses use JSON and Pydantic validation; pickle is not used. See [NumPy buffer deserialization](https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html).

Semantic lookup uses [SCAN iteration](https://redis.io/docs/latest/develop/clients/redis-py/scaniter/), restricted to the current corpus **and config**. Hashes are fetched in batches of 100. At more than 1,000 discovered entries in a namespace, semantic lookup returns a miss; exact caching still works. This is a small-corpus implementation, not a vector index. Scanning and transferring cached responses can become expensive before that limit.

## Response fields and provenance

| Field | Meaning |
| --- | --- |
| `cache` | `miss`, `exact`, `semantic`, or `bypass` |
| `cache_hit_query` | Original query whose answer was reused; null on fresh generation |
| `cache_similarity` | Cosine similarity for a semantic hit; otherwise null |
| `query` | Current request's question |
| `log_id` | On hits, the original generation log, not a new hit log |
| `latency_ms` | Current foreground request time, not the cached answer's original latency |

Cached answers retain the original sources, retrieval scores, and HyDE passage. These describe the original generation, not a new retrieval for the current query. Hit responses reload the original log's latest faithfulness score, so a cache populated while evaluation was pending does not permanently report null.

Hits do not create query logs or schedule another faithfulness judge. The existing faithfulness polling URL still refers to that original answer. Per-hit analytics and judge status are not stored in Redis. Miss timing includes the cache lookup and normal work through log preparation; final log commit, cache writes, HTTP delivery, and background judging remain outside that response timing. The live report records full HTTP wall time separately.

To bypass both caches for a request:

```bash
curl -X POST http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"how does multi-head attention work","use_cache":false}'
```

This returns `cache: "bypass"`, runs the full pipeline, and writes no cache entries. Existing entries are not replaced. `/search/*` and experiments do not consult or populate the response caches.

## Invalidation and outages

[version.py](../app/cache/version.py) manages `devmind:corpus:version` using Redis [INCR](https://redis.io/docs/latest/commands/incr/). The first counter value comes from a timestamp rather than starting again at 1 if the key disappears. Treat it as an opaque generation string.

All three ingestion job wrappers and the embedding job use `corpus_update()`. While a job is active, cache lookup and writes are suspended. When it finishes, the version advances. Failed jobs invalidate too, because earlier batches may have committed. In-flight requests keep their captured namespace and check it before reusing or writing an answer; an old generation cannot populate the new namespace.

If Redis is down during an update, a local dirty flag keeps the invalidation pending. On reconnect, the version advances before cache reads resume. The first cache access after an API restart also advances the version, covering offline corpus or model changes. Old keys become unreachable and expire naturally. Manual SQL changes in a running process must be followed by `increment_version()` or an API restart.

This uses the project's **single API worker** assumption. The active-job and dirty flags are process-local, not distributed coordination. Multiple API workers or writers outside these wrappers require shared invalidation state. Redis in Compose is disposable: persistence is disabled, so restarting its container empties the cache. PostgreSQL remains the durable source of documents and logs.

## Test the tiers

```bash
# A fresh generation if this query/config/version is not cached yet.
curl -X POST http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"how does multi-head attention work"}'

# Exact hit, including case and surrounding whitespace variations.
curl -X POST http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"  HOW DOES MULTI-HEAD ATTENTION WORK  "}'

# Punctuation changes the exact key; semantic lookup may reuse the first answer.
curl -X POST http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"how does multi-head attention work?"}'

curl http://localhost:8001/stats
docker compose exec redis redis-cli --scan --pattern 'devmind:cache:*'
```

`/stats.cache` reports availability, current corpus version, and current-generation exact/semantic counts across configurations. `redis_db_keys` comes from `DBSIZE`: it includes old generations and the version counter, so it is not the count of usable cached answers. Counts are approximate under concurrent updates. When unavailable or suspended, entry counts are null.

To test resilience, stop only this project's Redis, send a query, then restart it:

```bash
docker compose stop redis
# POST /query still works; /stats.cache.available is false.
docker compose up -d redis
# After reconnection, the next fresh query warms the cache again.
```

A no-op `/embed` job is a convenient invalidation check when all documents are already embedded. Note the version before and after; active entry counts reset while old keys retain their TTLs.

## Limits and study notes

A cosine threshold is not an equivalence test. The semantic path conservatively rejects changes in numeric tokens, quoted/code literals, or the presence of negation, but these checks cannot catch every meaning change. Case-insensitive exact normalization can also collapse case-sensitive identifiers. Use `use_cache:false` for questions where exact spelling, identifiers, or subtle qualifications matter. These are limitations of answer reuse, not guarantees supplied by a high similarity score.

Semantic hits are not promoted into new exact entries. This avoids extending an original answer's TTL through repeated reuse and avoids chains of increasingly distant paraphrases. Repeated variants can therefore still need an embedding lookup. A miss adds an original-query embedding call on top of the existing pipeline, which may separately embed a HyDE passage. Semantic caching saves work on hits but adds cost on misses; no traffic-wide cost reduction has been measured.

LRU evicts recently unused entries; LFU estimates access frequency with decay. Compose uses `volatile-lfu`, limiting eviction to keys with TTLs so the non-expiring version counter is protected. This is a practical local policy, not proof that LFU always wins. In this implementation, reading every semantic HASH during a scan also touches its frequency metadata, so LFU does not reliably distinguish useful matches from merely scanned candidates. See [Redis eviction policies](https://redis.io/docs/latest/develop/reference/eviction/).

A cache stampede happens when many requests miss the same expired entry and all regenerate it. Deterministic keys prevent duplicate stored rows, but do not prevent duplicate model calls. No request coalescing, distributed lock, or early refresh is implemented here. Probabilistic early expiration lets some requests refresh before TTL expiry, with increasing probability near expiry; it reduces synchronized refreshes but does not guarantee one recomputation. See the [original probabilistic expiration paper](https://www.vldb.org/pvldb/vol8/p886-vattani.pdf).


## Recorded verification

All 80 automated tests pass. A live run issued 22 query requests and produced seven new generation logs; all seven faithfulness jobs completed. Ten exact repeats had a median HTTP time of 11.19ms (10.23–25.38ms). The cold miss took 14.97s including model loading; an explicit warm full-pipeline bypass took 6.31s.

Adding `?` to the attention question produced a semantic hit at similarity 0.9842 in 730.23ms. The brief's longer paraphrase scored 0.9076 and correctly missed the configured threshold; the threshold was not lowered to force reuse. This verifies the semantic path but does not establish paraphrase accuracy or a 200ms latency guarantee.

Live checks also verified both TTLs using temporary one-second entries, latest-score refresh after judging, configuration isolation, explicit bypass, invalidation after a no-op embedding job, and Redis stop/restart recovery. Old generation keys remained after invalidation and became unreachable. The existing local Redis on port 6379 was left running. The 984 source chunks and 100 Day 13 experiment results were unchanged.

The [JSON report](cache/verification.json) contains per-request times, cache tiers, original log IDs, TTL observations, invalidation states, and environment metadata.
