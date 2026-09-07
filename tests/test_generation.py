import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient
from openai import APITimeoutError

from app import main
from app.generation import generator as g
from app.models import Document
from app.retrieval.query_understanding import EnhancedQuery


def doc(id, parent=None, domain='arxiv', content='A supported fact.'):
    return Document(id=id, parent_doc_id=parent, domain=domain, content=content,
                    source_url=f'https://example.com/{id}', metadata_={'title': f'Title {id}'})


class ContextTests(unittest.TestCase):
    def test_order_deduplication_and_source_numbers(self):
        results = [(doc(1, 'same'), .5, 1), (doc(2, 'same'), .1, 9),
                   (doc(3, 'same', 'github'), .2, 5), (doc(4), .3, 3), (doc(5), .4, 2)]
        context, sources = g.assemble_context(results)
        self.assertEqual([s.document_id for s in sources], [2, 3, 4, 5])
        self.assertIn('[Source 1: arxiv - Title 2]', context)
        self.assertEqual([s.source_id for s in sources], [1, 2, 3, 4])

    def test_token_trimming_and_special_text(self):
        text = 'word ' * 1000 + '<|endoftext|>'
        context, sources = g.assemble_context([(doc(1, content=text), .1)])
        expected = g.get_encoding().decode(g.get_encoding().encode(text, disallowed_special=())[:600])
        self.assertEqual(sources[0].content, expected)
        self.assertIn(expected, context)
        short = 'word ' * 700
        self.assertEqual(g.assemble_context([(doc(2, content=short), .2)])[1][0].content, short)

    def test_empty_and_max_sources(self):
        self.assertEqual(g.assemble_context([]), ('', []))
        self.assertEqual(len(g.assemble_context([(doc(i), i) for i in range(8)])[1]), 5)

    def test_titles_for_existing_source_formats(self):
        so = doc(1, domain='stackoverflow', content='Question: CUDA memory\n\nBody')
        so.metadata_ = {}
        self.assertEqual(g.source_title(so), 'CUDA memory')
        github = doc(2, domain='github')
        github.metadata_ = {'repo_name': 'pgvector/pgvector', 'section_title': 'Indexing'}
        self.assertEqual(g.source_title(github), 'pgvector/pgvector — Indexing')


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.enhanced = EnhancedQuery('question', 'HYPOTHETICAL TEXT', 'question expanded')
        self.process = patch.object(g, 'process_query', new=AsyncMock(return_value=self.enhanced)).start()
        self.search = patch.object(g, 'search_hybrid', return_value=[(doc(1), .2, 4.5)]).start()
        self.client = patch.object(g, 'get_client').start()
        self.addCleanup(patch.stopall)
        self.create = self.client.return_value.with_options.return_value.chat.completions.create
        self.create.return_value = NS(choices=[NS(finish_reason='stop', message=NS(
            refusal=None, content=json.dumps({'answer': 'A supported fact. [Source 1]'})))])

    def test_pipeline_uses_actual_context_and_server_metadata(self):
        response = g.generate_answer(MagicMock(), 'question', use_hyde=False, use_reranking=False)
        self.process.assert_awaited_once_with('question', use_hyde=False)
        self.assertIs(self.search.call_args.kwargs['enhanced'], self.enhanced)
        messages = self.create.call_args.kwargs['messages']
        self.assertIn('A supported fact.', messages[1]['content'])
        self.assertNotIn('HYPOTHETICAL TEXT', messages[1]['content'])
        self.assertTrue(self.create.call_args.kwargs['response_format']['json_schema']['strict'])
        self.assertEqual(response.sources[0].document_id, 1)
        self.assertEqual(response.retrieval_scores[0].reranker_score, 4.5)
        self.assertGreaterEqual(response.latency_ms, 0)

    def test_no_context_skips_generation(self):
        self.search.return_value = []
        response = g.generate_answer(MagicMock(), 'question')
        self.client.assert_not_called()
        self.assertEqual(response.answer, g.INSUFFICIENT_CONTEXT)
        self.assertEqual(response.sources, [])

    def test_reranking_disabled_has_null_reranker_score(self):
        self.search.return_value = [(doc(1), .2)]
        response = g.generate_answer(MagicMock(), 'question', use_reranking=False)
        self.assertIsNone(response.retrieval_scores[0].reranker_score)

    def test_invalid_generation_is_not_returned_as_success(self):
        for content, finish, refusal in [('not json', 'stop', None), ('{}', 'stop', None),
                                         ('{"answer":" "}', 'stop', None),
                                         ('{"answer":"partial"}', 'length', None),
                                         (None, 'stop', 'refused')]:
            self.create.return_value = NS(choices=[NS(finish_reason=finish, message=NS(content=content, refusal=refusal))])
            with self.subTest(content=content), self.assertRaises(g.GenerationError):
                g.generate_answer(MagicMock(), 'question')


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.db.add.side_effect = lambda log: setattr(log, 'id', 123)
        main.app.dependency_overrides[main.get_db] = lambda: self.db
        job = patch.object(main, 'run_faithfulness_job')
        job.start()
        self.addCleanup(job.stop)
        self.addCleanup(main.app.dependency_overrides.clear)
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)

    def test_valid_request_and_flags(self):
        output = g.QueryResponse(answer='Insufficient context.', sources=[], query='question',
                                 hyde_query='question', retrieval_scores=[], latency_ms=1)
        with patch.object(main, 'generate_answer', return_value=output) as generate:
            response = self.client.post('/query', json={'query': ' question ', 'use_hyde': False, 'use_reranking': False})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(generate.call_args.args[1], 'question')
        self.assertEqual(generate.call_args.kwargs, {'use_hyde': False, 'use_reranking': False})

    def test_invalid_requests_skip_pipeline(self):
        with patch.object(main, 'generate_answer') as generate:
            for body in ({}, {'query': ' '}, {'query': 'x'*2001}, {'query': 123}, {'query':'test','extra':True}):
                self.assertEqual(self.client.post('/query', json=body).status_code, 422)
        generate.assert_not_called()

    def test_upstream_errors(self):
        errors = [(g.GenerationError('internal detail'), 502),
                  (APITimeoutError(request=httpx.Request('POST', 'https://example.com')), 504)]
        for error, status in errors:
            with patch.object(main, 'generate_answer', side_effect=error):
                response = self.client.post('/query', json={'query': 'question'})
            self.assertEqual(response.status_code, status)
            self.assertNotIn('internal detail', response.text)
