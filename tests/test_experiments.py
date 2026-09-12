import json
import math
import unittest
from hashlib import sha256
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import main
from app.evaluation import experiment_runner as runner
from app.evaluation.benchmarks import BENCHMARKS
from app.evaluation.metrics import (annotated_recall, keyword_coverage, compare_results,
                                    ranking_metrics, evaluate_gates)
from app.evaluation.heldout_benchmark import HELDOUT_BENCHMARK, QUESTION_SHA256
from app.evaluation.heldout_questions import HELDOUT_QUESTIONS
from app.evaluation.comparator import compare_experiments
from app.generation.generator import QueryResponse
from app.models import Experiment, ExperimentResult, QueryLog
from app.retrieval import query_understanding as qu


@compiles(JSONB, 'sqlite')
def sqlite_jsonb(element, compiler, **kw):
    return 'JSON'


def row(key, value, **kwargs):
    fields = dict(benchmark_id=key, query=key, status='complete', retrieval_recall=value,
                  faithfulness_score=value, answer_relevance=value, latency_ms=value)
    fields.update(kwargs)
    return NS(**fields)


class MetricTests(unittest.TestCase):
    def test_precision_recall_mrr_and_graded_ndcg(self):
        scores = ranking_metrics([30, 10, 40, 20, 99], {10: 2, 20: 1, 30: 0, 40: 0})
        self.assertEqual(scores["retrieval_precision"], .4)
        self.assertEqual(scores["retrieval_recall"], 1)
        self.assertEqual(scores["mrr"], .5)
        expected_dcg = 3 / math.log2(3) + 1 / math.log2(5)
        expected_idcg = 3 + 1 / math.log2(3)
        self.assertAlmostEqual(scores["ndcg"], expected_dcg / expected_idcg)

    def test_precision_uses_fixed_k_and_zero_relevant_rank(self):
        scores = ranking_metrics([1, 2], {1: 0, 2: 0, 3: 1})
        self.assertEqual(scores["retrieval_precision"], 0)
        self.assertEqual(scores["retrieval_recall"], 0)
        self.assertEqual(scores["mrr"], 0)
        self.assertEqual(scores["ndcg"], 0)

    def test_gates_fail_only_significant_quality_regressions_and_latency_ceiling(self):
        metric = lambda difference, p, p95=1: {
            "paired_count": 20, "difference_b_minus_a": difference,
            "p_value": p, "p95_b": p95,
        }
        comparison = {"metrics": {
            "retrieval_recall": metric(-.2, .001),
            "retrieval_precision": metric(-.1, .2),
            "mrr": metric(.1, .001),
            "ndcg": metric(0, 1),
            "faithfulness_score": metric(0, 1),
            "latency_ms": metric(100, .001, 9000),
        }}
        gates = evaluate_gates(comparison, latency_p95_ceiling_ms=8000)
        self.assertFalse(gates["passed"])
        self.assertEqual(gates["quality"]["retrieval_recall"]["status"], "failed")
        self.assertEqual(gates["quality"]["retrieval_precision"]["status"], "passed")
        self.assertEqual(gates["quality"]["mrr"]["status"], "passed")
        self.assertEqual(gates["latency"]["status"], "failed")

    def test_missing_gate_data_cannot_pass(self):
        comparison = {"metrics": {name: {"paired_count": 0, "p_value": None,
                                          "difference_b_minus_a": None, "p95_b": None}
                                  for name in ("retrieval_recall", "retrieval_precision", "mrr",
                                               "ndcg", "faithfulness_score", "latency_ms")}}
        self.assertFalse(evaluate_gates(comparison)["passed"])

    def test_keyword_boundaries_alternatives_and_empty_labels(self):
        self.assertEqual(keyword_coverage('An evaluation uses external-knowledge.', ['eval', 'external knowledge', 'uses/use']), 2/3)
        self.assertIsNone(keyword_coverage('answer', []))
        self.assertEqual(annotated_recall([1, 1, 2], [1, 3]), .5)
        self.assertIsNone(annotated_recall([1], []))

    def test_pairing_order_missing_values_and_latency_direction(self):
        a = [row('one', 1), row('two', 2), row('three', None)]
        b = [row('two', 4), row('three', 10), row('one', 2)]
        results = compare_results(a, b)
        score = results['retrieval_recall']
        self.assertEqual(score['paired_count'], 2)
        self.assertEqual(score['excluded_pairs'], 1)
        self.assertEqual(score['mean_a'], 1.5)
        self.assertEqual(score['mean_b'], 3)
        self.assertAlmostEqual(score['std_a'], math.sqrt(.5))
        self.assertAlmostEqual(score['p_value'], .20483276469913345)
        self.assertEqual(score['improvement_percent'], 100)
        self.assertEqual(results['latency_ms']['improvement_percent'], -100)
        self.assertFalse(results['latency_ms']['b_is_better'])

    def test_degenerate_tests_and_json(self):
        for a, b, expected in [([1, 1], [1, 1], 1), ([0, 0], [1, 1], 0), ([1], [2], None), ([], [], None)]:
            result = compare_results([row(str(i), v) for i, v in enumerate(a)], [row(str(i), v) for i, v in enumerate(b)])
            self.assertEqual(result['retrieval_recall']['p_value'], expected)
            json.dumps(result, allow_nan=False)

    def test_failed_nonfinite_and_different_queries_are_excluded(self):
        a = [row('a', 1, status='failed'), row('b', float('nan')), row('c', 1)]
        b = [row('a', 0), row('b', 0), row('c', 2, query='changed')]
        self.assertEqual(compare_results(a, b)['retrieval_recall']['paired_count'], 0)

    def test_benchmark_shape(self):
        self.assertEqual(len(BENCHMARKS), 50)
        self.assertEqual(len({b['id'] for b in BENCHMARKS}), 50)
        self.assertEqual({b['difficulty'] for b in BENCHMARKS}, {'easy', 'medium', 'hard'})
        self.assertTrue({'arxiv', 'github', 'stackoverflow'} <= {b['domain'] for b in BENCHMARKS})
        self.assertTrue(all(3 <= len(b['expected_keywords']) <= 5 and b['relevant_sources'] for b in BENCHMARKS))

    def test_heldout_benchmark_has_complete_graded_pools(self):
        self.assertEqual(len(HELDOUT_BENCHMARK), 20)
        self.assertEqual(len({item["id"] for item in HELDOUT_BENCHMARK}), 20)
        for item in HELDOUT_BENCHMARK:
            pooled = set(item["pool"]["baseline"]) | set(item["pool"]["full_pipeline"])
            labels = {document_id: grade for document_id, grade, _hash in item["judgments"]}
            self.assertEqual(pooled, set(labels))
            self.assertTrue(all(grade in {0, 1, 2} for grade in labels.values()))
            self.assertTrue(any(grade > 0 for grade in labels.values()))
        question_hash = sha256("\n".join(
            item["id"] + "\t" + item["query"] for item in HELDOUT_QUESTIONS
        ).encode()).hexdigest()
        self.assertEqual(question_hash, QUESTION_SHA256)
        self.assertEqual(
            [(item["id"], item["query"]) for item in HELDOUT_BENCHMARK],
            [(item["id"], item["query"]) for item in HELDOUT_QUESTIONS],
        )

    def test_benchmark_rejects_changed_source(self):
        from unittest.mock import MagicMock
        db = MagicMock()
        source = BENCHMARKS[0]['relevant_sources'][0]
        db.get.return_value = NS(content='different', embedding=[1], domain=source['domain'])
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            runner.validate_benchmark(db, BENCHMARKS)


class FlagTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_flag_combinations(self):
        for hyde in (False, True):
            for expansion in (False, True):
                with patch.object(qu, 'generate_hyde', new_callable=AsyncMock, return_value='hypothesis') as h, \
                     patch.object(qu, 'expand_query', new_callable=AsyncMock, return_value='expanded') as e:
                    result = await qu.process_query('question', use_hyde=hyde, use_expansion=expansion)
                    self.assertEqual(h.await_count, int(hyde))
                    self.assertEqual(e.await_count, int(expansion))
                    self.assertEqual(result.hyde_passage, 'hypothesis' if hyde else 'question')
                    self.assertEqual(result.expanded_query, 'expanded' if expansion else 'question')


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
        event.listen(self.engine, 'connect', lambda conn, _: conn.execute('PRAGMA foreign_keys=ON'))
        for model in (QueryLog, Experiment, ExperimentResult):
            model.__table__.create(self.engine)
        self.sessions = sessionmaker(bind=self.engine)
        self.addCleanup(self.engine.dispose)
        self.config = dict(use_hyde=False, use_reranking=False, use_expansion=False)
        self.benchmarks = [dict(id=f'q{i}', query=f'question{i}', expected_keywords=['supported'], relevant_sources=[{'document_id': 1}]) for i in range(3)]
        for name, value in [('SessionLocal', self.sessions), ('BENCHMARKS', self.benchmarks)]:
            p = patch.object(runner, name, value)
            p.start()
            self.addCleanup(p.stop)
        for name in ('validate_benchmark', 'build_index', 'get_encoding', 'get_model'):
            p = patch.object(runner, name)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(runner, 'corpus_fingerprint', return_value='corpus')
        p.start()
        self.addCleanup(p.stop)
        def get_db():
            with self.sessions() as db:
                yield db
        main.app.dependency_overrides[main.get_db] = get_db
        self.addCleanup(main.app.dependency_overrides.clear)
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)

    def create(self):
        with self.sessions() as db:
            return runner.create_experiment(db, runner.ExperimentRequest(name='test', config=self.config)).id

    def test_runner_commits_each_result_continues_after_failure_and_is_idempotent(self):
        experiment_id = self.create()
        calls = []
        def generate(db, query, **config):
            with self.sessions() as other:
                self.assertEqual(other.query(ExperimentResult).count(), len(calls))
            calls.append(query)
            self.assertEqual(config, self.config)
            if len(calls) == 2:
                raise RuntimeError('failure')
            return QueryResponse(answer='supported', query=query, sources=[], hyde_query=query,
                                 latency_ms=1, retrieval_scores=[], retrieved_document_ids=[1])
        def judge(log_id):
            with self.sessions() as db:
                log = db.get(QueryLog, log_id)
                log.faithfulness_status = 'completed'
                log.faithfulness_score = .75
                db.commit()
        with patch.object(runner, 'generate_answer', side_effect=generate) as generation, \
             patch.object(runner, 'run_faithfulness_job', side_effect=judge):
            runner.run_experiment(experiment_id)
            runner.run_experiment(experiment_id)
            self.assertEqual(generation.call_count, 3)
        with self.sessions() as db:
            self.assertEqual(db.get(Experiment, experiment_id).status, 'complete_with_errors')
            rows = db.query(ExperimentResult).order_by(ExperimentResult.benchmark_id).all()
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0].retrieval_recall, 1)
            self.assertEqual(rows[0].faithfulness_score, .75)
            self.assertIsNone(rows[1].retrieval_recall)
            self.assertEqual(rows[1].status, 'failed')
            self.assertEqual(db.get(QueryLog, rows[0].query_log_id).config, self.config)

    def test_corpus_change_marks_failed_without_generating(self):
        experiment_id = self.create()
        with patch.object(runner, 'corpus_fingerprint', return_value='changed'), \
             patch.object(runner, 'generate_answer') as generate:
            runner.run_experiment(experiment_id)
        generate.assert_not_called()
        with self.sessions() as db:
            self.assertEqual(db.get(Experiment, experiment_id).status, 'failed')

    def test_endpoints_enqueue_validate_and_route_compare(self):
        with patch.object(main, 'run_experiment') as job:
            response = self.client.post('/experiments', json=dict(name='baseline', config=self.config))
        self.assertEqual(response.status_code, 202)
        experiment_id = response.json()['experiment_id']
        job.assert_called_once_with(experiment_id)
        detail = self.client.get(f'/experiments/{experiment_id}').json()
        self.assertEqual(detail['progress'], dict(completed=0, total=3, failed=0))
        self.assertEqual(self.client.get('/experiments/compare?exp_a=1&exp_b=999').status_code, 404)
        self.assertEqual(self.client.get('/experiments/compare?exp_a=1&exp_b=1').status_code, 409)
        self.assertEqual(self.client.get('/experiments/999').status_code, 404)
        for config in ({}, dict(self.config, typo=True), dict(self.config, use_hyde='false')):
            self.assertEqual(self.client.post('/experiments', json=dict(name='test', config=config)).status_code, 422)
        self.assertEqual(self.client.post('/experiments', json=dict(
            name='test', config=self.config, benchmark='unknown')).status_code, 422)

    def test_comparator_refuses_incompatible_runs(self):
        a, b = self.create(), self.create()
        with self.sessions() as db:
            with self.assertRaisesRegex(ValueError, 'finish'):
                compare_experiments(db, a, b)
            for id in (a, b):
                db.get(Experiment, id).status = 'complete'
            db.commit()
            self.assertEqual(compare_experiments(db, a, b)['metrics']['retrieval_recall']['paired_count'], 0)
            db.get(Experiment, b).corpus_fingerprint = 'changed'
            db.commit()
            with self.assertRaisesRegex(ValueError, 'corpus'):
                compare_experiments(db, a, b)


    def test_judge_failure_and_abstention_do_not_become_zero(self):
        experiment_id = self.create()
        output = QueryResponse(answer='supported', query='question', sources=[], hyde_query='question',
                               latency_ms=1, retrieval_scores=[], retrieved_document_ids=[1])
        statuses = iter(['failed', 'not_applicable', 'completed'])
        def judge(log_id):
            with self.sessions() as db:
                log = db.get(QueryLog, log_id)
                log.faithfulness_status = next(statuses)
                log.faithfulness_score = .5 if log.faithfulness_status == 'completed' else None
                log.evaluation_error = 'TimeoutError' if log.faithfulness_status == 'failed' else None
                db.commit()
        with patch.object(runner, 'generate_answer', return_value=output), \
             patch.object(runner, 'run_faithfulness_job', side_effect=judge):
            runner.run_experiment(experiment_id)
        with self.sessions() as db:
            rows = db.query(ExperimentResult).order_by(ExperimentResult.benchmark_id).all()
            self.assertEqual([r.faithfulness_score for r in rows], [None, None, .5])
            self.assertEqual([r.status for r in rows], ['evaluation_failed', 'complete', 'complete'])
            self.assertTrue(all(r.retrieval_recall == 1 for r in rows))
            self.assertEqual(rows[0].error, 'TimeoutError')
            self.assertEqual(rows[1].metric_details['faithfulness_status'], 'not_applicable')

    def test_persistence_failure_keeps_prior_results(self):
        experiment_id = self.create()
        output = QueryResponse(answer='supported', query='question', sources=[], hyde_query='question',
                               latency_ms=1, retrieval_scores=[], retrieved_document_ids=[1])
        save = runner.save_query_log
        calls = 0
        def save_or_fail(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('storage failed')
            return save(*args)
        with patch.object(runner, 'generate_answer', return_value=output), \
             patch.object(runner, 'run_faithfulness_job'), \
             patch.object(runner, 'save_query_log', side_effect=save_or_fail):
            runner.run_experiment(experiment_id)
        with self.sessions() as db:
            self.assertEqual(db.query(ExperimentResult).count(), 1)
            self.assertEqual(db.get(Experiment, experiment_id).status, 'failed')

    def test_completed_comparison_serializes_missing_metrics(self):
        a, b = self.create(), self.create()
        with self.sessions() as db:
            for id in (a, b):
                db.get(Experiment, id).status = 'complete'
            db.commit()
        response = self.client.get(f'/experiments/compare?exp_a={a}&exp_b={b}')
        self.assertEqual(response.status_code, 200)
        metric = response.json()['metrics']['faithfulness_score']
        self.assertIsNone(metric['mean_a'])
        self.assertIsNone(metric['p_value'])
