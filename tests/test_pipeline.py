import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app import main
from app.ingestion import arxiv_ingester, embedder, github_ingester, stackoverflow_ingester
from app.ingestion.chunker import CHUNK_SIZE, chunk_readme, split_document, split_text
from app.models import Document
from app.retrieval import bm25_index, dense, hybrid, reranker, query_understanding as qu


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document(id=i, content=f'Content {i}', domain='github',
                     source_url=f'https://example.com/{i}', metadata_={'title': str(i)})
            for i in range(1, 31)
        ]
        self.db = MagicMock()
        self.db.query.return_value.filter.return_value.all.return_value = self.docs
        main.app.dependency_overrides[main.get_db] = lambda: self.db
        self.addCleanup(main.app.dependency_overrides.clear)
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)

    def test_result_count_and_sources(self):
        with patch.object(hybrid, 'process_query', new=AsyncMock(return_value=qu.EnhancedQuery('test', 'passage', 'test terms'))), \
             patch.object(hybrid, 'get_embeddings', return_value=[[0.1] * 1536]), \
             patch.object(hybrid, 'search_dense', side_effect=lambda db, v, k: [(i, .1) for i in range(1, k + 1)]), \
             patch.object(hybrid, 'search_bm25', return_value=[]), \
             patch.object(reranker, 'get_model') as model:
            model.return_value.predict.side_effect = lambda pairs: list(range(len(pairs)))
            for rerank in (False, True):
                for k in (1, 8):
                    with self.subTest(k=k, rerank=rerank):
                        response = self.client.get('/search/hybrid', params={'query': 'test', 'k': k, 'rerank': rerank})
                        self.assertEqual(response.status_code, 200)
                        results = response.json()['results']
                        self.assertEqual(len(results), k)
                        self.assertTrue(results[0]['source_url'])
                        self.assertIn('content', results[0])
                        self.assertIn('metadata', results[0])

    def test_invalid_queries_do_not_call_embedding_api(self):
        with patch.object(hybrid, 'get_embeddings') as embed:
            for endpoint in ('/search/bm25', '/search/hybrid'):
                for params in ({'query': ' '}, {'query': ''}, {'query': 'test', 'k': 0},
                               {'query': 'test', 'k': -1}, {'query': 'test', 'k': 101}):
                    self.assertEqual(self.client.get(endpoint, params=params).status_code, 422)
            embed.assert_not_called()

    def test_dense_excludes_pending_embeddings(self):
        from sqlalchemy.orm import Session
        from sqlalchemy.dialects import postgresql
        with Session() as session:
            query = session.query(Document.id, Document.embedding.cosine_distance([.1] * 1536).label('distance'))
            db = MagicMock()
            db.query.return_value = query
            with patch('sqlalchemy.orm.Query.all', autospec=True, return_value=[]) as execute:
                dense.search_dense(db, [.1] * 1536)
                sql = str(execute.call_args.args[0].statement.compile(dialect=postgresql.dialect()))
                self.assertIn('documents.embedding IS NOT NULL', sql)

    def test_bm25_empty_and_punctuation_only_corpus(self):
        for docs in ([], [SimpleNamespace(id=1, content='!!!')]):
            db = MagicMock()
            db.query.return_value.order_by.return_value.all.return_value = docs
            bm25_index.build_index(db)
            self.assertEqual(bm25_index.search('test'), [])

    def test_bm25_rebuild_picks_up_new_content(self):
        db = MagicMock()
        docs = [SimpleNamespace(id=1, content='python lists'), SimpleNamespace(id=2, content='neural networks')]
        db.query.return_value.order_by.return_value.all.return_value = docs
        bm25_index.build_index(db)
        self.assertEqual(bm25_index.search('cosine'), [])
        docs.append(SimpleNamespace(id=3, content='cosine vectors'))
        bm25_index.build_index(db)
        self.assertEqual(bm25_index.search('cosine')[0][0], 3)

    def test_rrf_rewards_shared_results(self):
        self.assertEqual(hybrid.rrf_merge([(1, .1), (2, .2)], [(2, 5), (3, 4)])[0][0], 2)


class IngestionTests(unittest.TestCase):
    def test_long_text_preserves_tail_and_bounds(self):
        text = 'word ' * 2000 + 'TAIL'
        chunks = split_text(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= CHUNK_SIZE for chunk in chunks))
        self.assertTrue(chunks[-1].endswith('TAIL'))
        self.assertIn(chunks[0][-100:].strip(), chunks[1][:220])

    def test_readme_headings_inside_code_are_not_sections(self):
        chunks = chunk_readme('# Intro\n' + 'a' * 60 + '\n```python\n# code\nprint(1)\n```\n## Install\n' + 'b' * 60, 'org/repo')
        self.assertEqual([c['section_title'] for c in chunks], ['Intro', 'Install'])

    def test_split_documents_keep_provenance(self):
        doc = Document(content='x' * 5000, domain='stackoverflow', parent_doc_id='123', source_url='https://example.com', metadata_={'title': 'Test'})
        chunks = split_document(doc)
        self.assertEqual([c.chunk_index for c in chunks], list(range(len(chunks))))
        self.assertTrue(all(c.parent_doc_id == '123' and c.metadata_ == doc.metadata_ for c in chunks))
        self.assertTrue(all(len(c.content) <= CHUNK_SIZE for c in chunks))

    def test_ingestion_refreshes_index_even_after_failure(self):
        for module, job, ingest in (
            (arxiv_ingester, 'run_arxiv_ingestion', 'ingest_arxiv_papers'),
            (github_ingester, 'run_github_ingestion', 'ingest_github_repos'),
            (stackoverflow_ingester, 'run_stackoverflow_ingestion', 'ingest_stackoverflow_threads'),
        ):
            with self.subTest(source=module.__name__), \
                 patch.object(module, 'SessionLocal') as session, \
                 patch.object(module, ingest, side_effect=RuntimeError('fetch failed')), \
                 patch.object(bm25_index, 'build_index') as rebuild:
                with self.assertRaises(RuntimeError):
                    getattr(module, job)()
                session.return_value.rollback.assert_called_once()
                rebuild.assert_called_once_with(session.return_value)
                session.return_value.close.assert_called_once()

    def test_embedding_mismatch_rolls_back(self):
        db = MagicMock()
        doc = Document(content='text')
        db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [doc]
        with patch.object(embedder, 'get_embeddings', return_value=[]), self.assertLogs(level='ERROR'):
            with self.assertRaises(ValueError):
                embedder.embed_pending_documents(db)
        db.rollback.assert_called_once()
        db.commit.assert_not_called()
        self.assertIsNone(doc.embedding)

    def test_embedding_batch_commits(self):
        db = MagicMock()
        doc = Document(content='text')
        db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.side_effect = [[doc], []]
        with patch.object(embedder, 'get_embeddings', return_value=[[.1] * 1536]):
            embedder.embed_pending_documents(db)
        self.assertEqual(len(doc.embedding), 1536)
        db.commit.assert_called_once()


class QueryUnderstandingTests(unittest.IsolatedAsyncioTestCase):
    async def test_enhancements_run_concurrently(self):
        import asyncio
        started = set()
        both_started = asyncio.Event()

        async def complete(query, prompt, max_tokens):
            started.add(prompt)
            if len(started) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1)
            return 'A technical passage about attention.' if prompt == qu.HYDE_PROMPT else 'attention, alignment, alignment'

        with patch.object(qu, '_complete', side_effect=complete):
            result = await qu.process_query('attention')
        self.assertEqual(result.hyde_passage, 'A technical passage about attention.')
        self.assertEqual(result.expanded_query, 'attention alignment')
        self.assertEqual(result.original_query, 'attention')

    async def test_hyde_disabled_still_expands(self):
        with patch.object(qu, 'generate_hyde', new_callable=AsyncMock) as hyde, \
             patch.object(qu, '_complete', new_callable=AsyncMock, return_value='alignment'):
            result = await qu.process_query('attention', use_hyde=False)
        hyde.assert_not_awaited()
        self.assertEqual(result.hyde_passage, 'attention')
        self.assertEqual(result.expanded_query, 'attention alignment')

    async def test_failure_falls_back_independently(self):
        async def complete(query, prompt, max_tokens):
            if prompt == qu.HYDE_PROMPT:
                raise TimeoutError()
            return 'alignment'

        with patch.object(qu, '_complete', side_effect=complete), self.assertLogs(qu.logger, level='WARNING'):
            result = await qu.process_query('attention')
        self.assertEqual(result.hyde_passage, 'attention')
        self.assertEqual(result.expanded_query, 'attention alignment')

    async def test_bad_outputs_fall_back(self):
        for bad_output in ('', '- bullet\n- list', '```json', 'Text\n```bash\ncommand\n```'):
            with patch.object(qu, '_complete', new_callable=AsyncMock, return_value=bad_output), self.assertLogs(qu.logger, level='WARNING'):
                result = await qu.process_query('attention')
            self.assertEqual(result, qu.EnhancedQuery('attention', 'attention', 'attention'))

    async def test_missing_key_falls_back(self):
        with patch.object(qu, 'AsyncOpenAI', side_effect=ValueError('missing key')), self.assertLogs(qu.logger, level='WARNING'):
            result = await qu.process_query('attention')
        self.assertEqual(result, qu.EnhancedQuery('attention', 'attention', 'attention'))

    async def test_blank_query_skips_calls(self):
        with patch.object(qu, '_complete', new_callable=AsyncMock) as complete:
            result = await qu.process_query(' ')
        complete.assert_not_awaited()
        self.assertEqual(result.original_query, '')


class EnhancedRetrievalTests(unittest.TestCase):
    def test_each_stage_receives_the_correct_query(self):
        doc = Document(id=1, domain='arxiv', content='Retrieved passage')
        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = [doc]
        enhanced = qu.EnhancedQuery('attention', 'Hypothetical passage', 'attention alignment')
        with patch.object(hybrid, 'process_query', new_callable=AsyncMock, return_value=enhanced) as process, \
             patch.object(hybrid, 'get_embeddings', return_value=[[.1] * 1536]) as embed, \
             patch.object(hybrid, 'search_dense', return_value=[(1, .1)]), \
             patch.object(hybrid, 'search_bm25', return_value=[]) as sparse, \
             patch.object(reranker, 'rerank', return_value=[]) as rerank:
            hybrid.search_hybrid(db, 'attention', use_hyde=True)
        process.assert_awaited_once_with('attention', use_hyde=True)
        embed.assert_called_once_with(['Hypothetical passage'])
        sparse.assert_called_once_with('attention alignment', k=20)
        self.assertEqual(rerank.call_args.args[0], 'attention')

    def test_endpoint_passes_hyde_flag(self):
        main.app.dependency_overrides[main.get_db] = lambda: MagicMock()
        self.addCleanup(main.app.dependency_overrides.clear)
        with patch.object(main, 'search_hybrid', return_value=[]) as search:
            client = TestClient(main.app)
            self.addCleanup(client.close)
            for flag in (True, False):
                response = client.get('/search/hybrid', params={'query': 'attention', 'hyde': flag})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(search.call_args.kwargs['use_hyde'], flag)


if __name__ == '__main__':
    unittest.main()
