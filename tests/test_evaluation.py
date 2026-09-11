import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import main
from app.evaluation.citation_verifier import verify_citations
from app.evaluation import faithfulness as f
from app.generation.generator import QueryResponse, RetrievalScore, Source
from app.models import QueryLog


class CitationTests(unittest.TestCase):
    def test_real_ids_not_list_positions(self):
        result = verify_citations('One [Source 2]. Two [Source 7]. Bad [Source 1].',
                                  [{'source_id': 2}, NS(source_id=7)])
        self.assertFalse(result['all_citations_valid'])
        self.assertEqual(result['invalid_citations'], [1])
        self.assertEqual(result['citation_count'], 3)

    def test_citations_after_period_belong_to_previous_sentence(self):
        result = verify_citations('First claim. [Source 1] [Source 2] Second claim.\nThird [Source 1].',
                                  [NS(source_id=1), NS(source_id=2)])
        self.assertTrue(result['all_citations_valid'])
        self.assertEqual(result['uncited_sentences'], 1)
        self.assertEqual(result['citation_count'], 3)

    def test_missing_and_malformed_are_distinct(self):
        result = verify_citations('An uncited claim.', [])
        self.assertTrue(result['all_citations_valid'])
        self.assertEqual(result['citation_count'], 0)
        self.assertEqual(result['uncited_sentences'], 1)
        result = verify_citations('Claim [Source 0] [Source -1] [source 1] [Source x].', [NS(source_id=1)])
        self.assertFalse(result['all_citations_valid'])
        self.assertEqual(result['invalid_citations'], [-1, 0])
        self.assertEqual(len(result['malformed_citations']), 2)

    def test_code_does_not_inflate_sentence_count(self):
        result = verify_citations('Use this [Source 1].\n```python\nx.y()\n```', [NS(source_id=1)])
        self.assertEqual(result['uncited_sentences'], 0)


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    def mock_api(self, content, finish='stop', refusal=None):
        patcher = patch.object(f, 'AsyncOpenAI')
        client = patcher.start().return_value.__aenter__.return_value
        self.addCleanup(patcher.stop)
        client.chat.completions.create = AsyncMock(return_value=NS(choices=[NS(
            finish_reason=finish, message=NS(content=content, refusal=refusal))]))
        return client.chat.completions.create

    async def test_score_is_computed_from_verdicts(self):
        claims = [{'claim': str(i), 'supported': i < 2, 'reason': 'evidence'} for i in range(3)]
        create = self.mock_api(json.dumps({'claims': claims}))
        score = await f.score_faithfulness('Answer', ['Actual context'])
        self.assertAlmostEqual(score, 2/3)
        self.assertEqual(create.call_args.kwargs['response_format'], {'type': 'json_object'})
        self.assertIn('Actual context', create.call_args.kwargs['messages'][1]['content'])

    async def test_abstention_has_no_numeric_score(self):
        self.mock_api('{"claims": []}')
        self.assertIsNone(await f.score_faithfulness('Not enough information.', []))

    async def test_string_booleans_are_rejected(self):
        self.mock_api('{"claims":[{"claim":"x","supported":"false","reason":"missing"}]}')
        with self.assertRaises(ValueError):
            await f.evaluate_claims('Answer', ['Context'])

    async def test_malformed_or_truncated_json_is_an_error(self):
        self.mock_api('not json')
        with self.assertRaises(ValueError):
            await f.evaluate_claims('Answer', [])

    async def test_refusal_is_an_error(self):
        self.mock_api(None, refusal='No')
        with self.assertRaises(ValueError):
            await f.evaluate_claims('Answer', [])


@compiles(JSONB, 'sqlite')
def sqlite_jsonb(element, compiler, **kw):
    return 'JSON'


class LogIntegrationTests(unittest.TestCase):
    def setUp(self):
        cache = patch("app.cache.redis_cache.get_redis", return_value=None)
        cache.start()
        self.addCleanup(cache.stop)
        self.engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
        QueryLog.__table__.create(self.engine)
        self.sessions = sessionmaker(bind=self.engine)
        self.addCleanup(self.engine.dispose)

        def get_db():
            with self.sessions() as db:
                yield db
        main.app.dependency_overrides[main.get_db] = get_db
        self.addCleanup(main.app.dependency_overrides.clear)
        session_patch = patch.object(f, 'SessionLocal', self.sessions)
        session_patch.start()
        self.addCleanup(session_patch.stop)
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)
        self.output = QueryResponse(
            answer='Vectors are indexed [Source 1].', query='vectors', hyde_query='Hypothesis',
            sources=[Source(source_id=1, document_id=42, domain='github', title='Index',
                            source_url='https://example.com', content='Vectors are indexed.')],
            retrieval_scores=[RetrievalScore(source_id=1, rrf_score=.02, reranker_score=2.1)],
            latency_ms=1, expanded_query='vectors indexing', chunks_retrieved=5,
            stage_latency_ms={'understanding': 1, 'embedding': 2, 'search': 3, 'reranking': 4},
        )

    def post(self, judge_result):
        with patch.object(main, 'generate_answer', return_value=self.output), \
             patch.object(f, 'evaluate_claims', new=AsyncMock(return_value=judge_result)):
            return self.client.post('/query', json={'query': 'vectors', 'use_hyde': False})

    def test_log_persistence_background_score_and_poll(self):
        result = f.FaithfulnessResult(claims=[f.Claim(claim='Vectors are indexed', supported=True, reason='Context states it')])
        response = self.post(result)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsNone(data['faithfulness_score'])
        self.assertTrue(data['citation_valid'])
        log_id = data['log_id']
        poll = self.client.get(f'/query/{log_id}/faithfulness').json()
        self.assertEqual(poll['status'], 'completed')
        self.assertEqual(poll['faithfulness_score'], 1)
        with self.sessions() as db:
            log = db.get(QueryLog, log_id)
            self.assertEqual(log.expanded_query, 'vectors indexing')
            self.assertEqual(log.chunks_retrieved, 5)
            self.assertEqual(log.sources[0]['document_id'], 42)
            self.assertFalse(log.config['use_hyde'])
            self.assertIn('embedding', log.stage_latency_ms)
            self.assertIn('citation_verification', log.stage_latency_ms)
        logs = self.client.get('/logs?limit=1').json()['logs']
        self.assertEqual(logs[0]['id'], log_id)

    def test_pending_and_failed_are_not_low_scores(self):
        with patch.object(main, 'generate_answer', return_value=self.output), \
             patch.object(main, 'run_faithfulness_job'):
            data = self.client.post('/query', json={'query':'vectors'}).json()
        poll_url = f"/query/{data['log_id']}/faithfulness"
        self.assertEqual(self.client.get(poll_url).json()['status'], 'pending')
        with patch.object(f, 'evaluate_claims', new=AsyncMock(side_effect=TimeoutError())), self.assertLogs(f.logger, level='WARNING'):
            f.run_faithfulness_job(data['log_id'])
        poll = self.client.get(poll_url).json()
        self.assertEqual(poll['status'], 'failed')
        self.assertIsNone(poll['faithfulness_score'])

    def test_no_claims_not_applicable(self):
        data = self.post(f.FaithfulnessResult(claims=[])).json()
        poll = self.client.get(f"/query/{data['log_id']}/faithfulness").json()
        self.assertEqual(poll['status'], 'not_applicable')
        self.assertIsNone(poll['faithfulness_score'])

    def test_invalid_citations_are_logged(self):
        self.output.answer = 'A claim [Source 99].'
        data = self.post(f.FaithfulnessResult(claims=[])).json()
        self.assertFalse(data['citation_valid'])
        with self.sessions() as db:
            self.assertEqual(db.get(QueryLog, data['log_id']).citation_details['invalid_citations'], [99])

    def test_unknown_log_and_invalid_limits(self):
        self.assertEqual(self.client.get('/query/999/faithfulness').status_code, 404)
        for limit in (0, -1, 101):
            self.assertEqual(self.client.get('/logs', params={'limit':limit}).status_code, 422)

    def test_generation_errors_are_logged_without_scheduling_judge(self):
        with patch.object(main, 'generate_answer', side_effect=main.GenerationError('private detail')), \
             patch.object(main, 'run_faithfulness_job') as job:
            response = self.client.post('/query', json={'query':'vectors'})
        self.assertEqual(response.status_code, 502)
        job.assert_not_called()
        with self.sessions() as db:
            log = db.query(QueryLog).one()
            self.assertEqual(log.faithfulness_status, 'generation_failed')
            self.assertEqual(log.evaluation_error, 'GenerationError')
            self.assertIsNone(log.answer)
