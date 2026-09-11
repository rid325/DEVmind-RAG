import json
import unittest
from time import perf_counter
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import numpy as np
from redis.exceptions import ConnectionError
from fastapi.testclient import TestClient

from app import main
from app.cache import redis_cache as cache, version, query_cache
from app.generation.generator import QueryRequest, QueryResponse


def vector(x=1, y=0):
    return [x, y] + [0] * 1534


def answer(query='attention'):
    return QueryResponse(query=query, answer='A supported answer.', sources=[], hyde_query='hypothesis',
                         retrieval_scores=[], latency_ms=5000, log_id=123)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        p = patch.object(cache, 'get_redis', return_value=self.client)
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(cache, 'scope_is_current', return_value=True)
        p.start()
        self.addCleanup(p.stop)
        self.scope = 'devmind:cache:v1:1:config'

    def test_exact_normalization_json_and_ttl(self):
        payload = answer().model_dump()
        cache.set_exact(' Attention ', payload, scope=self.scope)
        args, kwargs = self.client.set.call_args
        self.assertEqual(args[0], cache.exact_key('ATTENTION', self.scope))
        self.assertEqual(json.loads(args[1]), payload)
        self.assertEqual(kwargs['ex'], 86400)
        self.client.get.return_value = args[1].encode()
        self.assertEqual(cache.get_exact('attention', scope=self.scope), payload)
        self.assertNotEqual(cache.exact_key('attention?', self.scope), args[0])

    def test_config_and_version_are_isolated(self):
        with patch.object(version, 'get_current_version', return_value='1'):
            a = cache.cache_scope({'use_hyde': False})
            b = cache.cache_scope({'use_hyde': True})
        with patch.object(version, 'get_current_version', return_value='2'):
            c = cache.cache_scope({'use_hyde': False})
        self.assertEqual(len({a, b, c}), 3)

    def test_corrupt_exact_entry_is_a_miss(self):
        for raw in (b'not json', b'[]', b'\xff'):
            self.client.get.return_value = raw
            self.assertIsNone(cache.get_exact('attention', scope=self.scope))

    def test_embedding_roundtrip_and_validation(self):
        restored = cache.deserialize_embedding(cache.serialize_embedding(vector()))
        np.testing.assert_array_equal(restored, vector())
        self.assertEqual(len(cache.serialize_embedding(vector())), 6144)
        for invalid in ([1, 2], [0]*1536, [float('nan')]*1536, [[1]*1536]):
            with self.assertRaises(ValueError):
                cache.serialize_embedding(invalid)
        with self.assertRaises(ValueError):
            cache.deserialize_embedding(b'invalid')

    def test_semantic_best_match_current_namespace_and_guards(self):
        self.client.scan_iter.return_value = [b'a', b'b', b'c', b'bad']
        def entry(query, embedding):
            return {b'query':query.encode(), b'embedding':cache.serialize_embedding(embedding),
                    b'response':json.dumps(answer(query).model_dump()).encode()}
        self.client.pipeline.return_value.execute.return_value = [
            entry('attention explanation', vector(.97, .2)),
            entry('explain attention', vector()),
            entry('do not explain attention', vector()), {},
        ]
        result = cache.get_semantic(vector(), query='attention', scope=self.scope)
        self.assertEqual(result['cache_hit_query'], 'explain attention')
        self.assertAlmostEqual(result['cache_similarity'], 1)
        self.client.scan_iter.assert_called_once_with(match=f'{self.scope}:semantic:*', count=100)
        self.assertFalse(cache.semantic_compatible('use Python 3.12', 'use Python 3.14'))
        self.assertFalse(cache.semantic_compatible('set `foo`', 'set `bar`'))

    def test_below_threshold_and_scan_limit_miss(self):
        self.client.scan_iter.return_value = [b'a']
        self.client.pipeline.return_value.execute.return_value = [{
            b'query':b'other', b'embedding':cache.serialize_embedding(vector(0, 1)),
            b'response':json.dumps(answer().model_dump()).encode()}]
        self.assertIsNone(cache.get_semantic(vector(), query='attention', scope=self.scope))
        self.client.scan_iter.return_value = list(range(1001))
        self.assertIsNone(cache.get_semantic(vector(), query='attention', scope=self.scope))

    def test_semantic_write_is_transactional_with_ttl(self):
        cache.set_semantic('attention', vector(), answer().model_dump(), scope=self.scope)
        self.client.pipeline.assert_called_once_with(transaction=True)
        self.client.pipeline.return_value.expire.assert_called_once_with(
            cache.exact_key('attention', self.scope).replace(':exact:', ':semantic:'), 86400)
        self.client.pipeline.return_value.execute.assert_called_once()

    def test_outage_and_stale_scope_do_not_write(self):
        with patch.object(cache, 'scope_is_current', return_value=False):
            cache.set_exact('q', {}, scope=self.scope)
            cache.set_semantic('q', vector(), {}, scope=self.scope)
        self.client.set.assert_not_called()
        self.client.pipeline.assert_not_called()
        self.client.get.side_effect = ConnectionError('unavailable')
        self.assertIsNone(cache.get_exact('q', scope=self.scope))
        self.client.set.side_effect = ConnectionError('unavailable')
        cache.set_exact('q', {}, scope=self.scope)

    def test_stats_counts_only_current_generation(self):
        with patch.object(version, 'get_current_version', return_value='7'):
            self.client.scan_iter.side_effect = [[b'a', b'a', b'b'], [b'c']]
            self.client.dbsize.return_value = 9
            stats = cache.cache_stats()
        self.assertEqual(stats['exact_entries'], 2)
        self.assertEqual(stats['semantic_entries'], 1)
        self.assertEqual(stats['redis_db_keys'], 9)
        self.assertTrue(all(':7:' in call.kwargs['match'] for call in self.client.scan_iter.call_args_list))


class VersionTests(unittest.TestCase):
    def setUp(self):
        self.counter = 10
        self.client = MagicMock()
        self.client.incr.side_effect = self.increment
        self.client.get.side_effect = lambda _: str(self.counter).encode()
        p = patch.object(cache, 'get_redis', return_value=self.client)
        self.redis = p.start()
        self.addCleanup(p.stop)
        for name, value in [('_dirty', True), ('_active_updates', 0)]:
            p = patch.object(version, name, value)
            p.start()
            self.addCleanup(p.stop)

    def increment(self, key):
        self.counter += 1
        return self.counter

    def test_startup_success_and_failed_partial_updates(self):
        self.assertEqual(version.get_current_version(), '11')
        self.assertEqual(version.get_current_version(), '11')
        with version.corpus_update():
            self.assertIsNone(version.get_current_version())
        self.assertEqual(version.get_current_version(), '12')
        with self.assertRaises(RuntimeError):
            with version.corpus_update():
                raise RuntimeError('partial commit')
        self.assertEqual(version.get_current_version(), '13')

    def test_invalidation_retries_after_redis_outage(self):
        self.assertEqual(version.get_current_version(), '11')
        self.redis.return_value = None
        with version.corpus_update():
            pass
        self.assertTrue(version._dirty)
        self.redis.return_value = self.client
        self.assertEqual(version.get_current_version(), '12')
        self.assertFalse(version._dirty)

    def test_overlapping_updates_keep_cache_suspended(self):
        with version.corpus_update():
            with version.corpus_update():
                pass
            self.assertIsNone(version.get_current_version())
        self.assertIsNotNone(version.get_current_version())


class QueryCacheTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.db.get.return_value = NS(faithfulness_score=.75)
        self.request = QueryRequest(query='attention')
        for name, result in [('cache_scope', 'scope'), ('scope_is_current', True), ('get_redis', MagicMock()),
                             ('get_exact', None), ('get_semantic', None)]:
            p = patch.object(cache, name, return_value=result)
            setattr(self, name, p.start())
            self.addCleanup(p.stop)
        p = patch.object(query_cache, 'get_client')
        self.api = p.start()
        self.addCleanup(p.stop)
        self.api.return_value.with_options.return_value.embeddings.create.return_value = NS(data=[NS(index=0, embedding=vector())])

    def test_exact_hit_skips_embedding_and_refreshes_judge(self):
        self.get_exact.return_value = answer('Attention').model_dump()
        result, _, embedding = query_cache.lookup_cached_answer(self.request, self.db, perf_counter())
        self.assertEqual(result.cache, 'exact')
        self.assertEqual(result.query, 'attention')
        self.assertEqual(result.cache_hit_query, 'Attention')
        self.assertEqual(result.faithfulness_score, .75)
        self.assertLess(result.latency_ms, 5000)
        self.assertIsNone(embedding)
        self.api.assert_not_called()

    def test_semantic_hit_retains_original_log_and_current_query(self):
        self.get_semantic.return_value = dict(answer('explain attention').model_dump(), cache_hit_query='explain attention', cache_similarity=.98)
        result, _, _ = query_cache.lookup_cached_answer(self.request, self.db, perf_counter())
        self.assertEqual(result.cache, 'semantic')
        self.assertEqual(result.query, self.request.query)
        self.assertEqual(result.cache_hit_query, 'explain attention')
        self.assertEqual(result.log_id, 123)

    def test_opt_out_and_unavailable_redis_skip_embedding(self):
        result = query_cache.lookup_cached_answer(QueryRequest(query='q', use_cache=False), self.db, perf_counter())
        self.assertEqual(result, (None, None, None))
        self.cache_scope.return_value = None
        self.assertEqual(query_cache.lookup_cached_answer(self.request, self.db, perf_counter()), (None, None, None))
        self.api.assert_not_called()

    def test_bad_payload_missing_log_and_embedding_failure_fall_back(self):
        self.get_exact.return_value = {'bad':'schema'}
        self.api.side_effect = TimeoutError()
        self.assertEqual(query_cache.lookup_cached_answer(self.request, self.db, perf_counter()), (None, 'scope', None))
        self.get_exact.return_value = answer().model_dump()
        self.db.get.return_value = None
        self.assertIsNone(query_cache.lookup_cached_answer(self.request, self.db, perf_counter())[0])

    def test_endpoint_hit_skips_generation_logging_and_judging(self):
        main.app.dependency_overrides[main.get_db] = lambda:self.db
        self.addCleanup(main.app.dependency_overrides.clear)
        self.get_exact.return_value = answer().model_dump()
        with patch.object(main, 'generate_answer') as generate, \
             patch.object(main, 'save_query_log') as save, \
             patch.object(main, 'run_faithfulness_job') as judge:
            client = TestClient(main.app)
            self.addCleanup(client.close)
            response = client.post('/query', json={'query':'attention'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['cache'], 'exact')
        generate.assert_not_called()
        save.assert_not_called()
        judge.assert_not_called()

    def test_nonfinite_cached_payload_is_a_miss(self):
        self.get_exact.return_value = dict(answer().model_dump(), latency_ms=float('nan'))
        self.api.side_effect = TimeoutError()
        self.assertIsNone(query_cache.lookup_cached_answer(self.request, self.db, perf_counter())[0])

    def test_generation_failure_does_not_populate_cache(self):
        main.app.dependency_overrides[main.get_db] = lambda:self.db
        self.addCleanup(main.app.dependency_overrides.clear)
        with patch.object(main, 'generate_answer', side_effect=main.GenerationError('bad answer')), \
             patch.object(main, 'record_generation_failure'), \
             patch.object(main, 'store_cached_answer') as store:
            client = TestClient(main.app)
            self.addCleanup(client.close)
            response = client.post('/query', json={'query':'attention'})
        self.assertEqual(response.status_code, 502)
        store.assert_not_called()


class ConnectionTests(unittest.TestCase):
    def test_failure_cooldown_and_reconnection(self):
        client = MagicMock()
        with patch.object(cache, '_client', client), patch.object(cache, '_retry_after', 0), \
             patch.object(cache, 'monotonic', return_value=5) as clock:
            client.ping.side_effect = ConnectionError('down')
            self.assertIsNone(cache.get_redis())
            self.assertIsNone(cache.get_redis())
            self.assertEqual(client.ping.call_count, 1)
            clock.return_value = 7
            client.ping.side_effect = None
            self.assertIs(cache.get_redis(), client)
