# DEVmind RAG

A FastAPI backend for searching arXiv abstracts, GitHub READMEs, and Stack Overflow answers. It retrieves relevant chunks and generates answers with inline source citations.

For a full explanation of the implementation and design choices, read the [project guide](docs/PROJECT_GUIDE.md).

## How it works

- **arXiv:** searches 10 configured topics and stores abstracts with paper metadata. It doesn't download PDFs.
- **GitHub:** fetches READMEs from 20 configured repositories and splits them by heading.
- **Stack Overflow:** fetches the top 25 questions for each configured tag and keeps threads with an accepted answer.

Long content is split into chunks of up to 1,800 characters with overlap. Documents are stored in PostgreSQL with their source URL and metadata. Existing source IDs are skipped; ingestion does not update previously stored content.

Embedding is a separate step. `/embed` processes pending chunks in batches of 50 using OpenAI's `text-embedding-3-small` model and stores 1,536-dimensional vectors in pgvector.

Before hybrid retrieval, `gpt-4o-mini` generates a hypothetical technical passage (HyDE) for dense embedding and related terms for BM25. The two calls run concurrently. Reranking still uses the original question. If either enhancement fails, that part falls back to the original query.

Search combines cosine similarity and BM25 using reciprocal rank fusion. Optional reranking uses `cross-encoder/ms-marco-MiniLM-L-6-v2`. The model downloads on the first reranked search, which can take longer.

## Run locally

Install dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```dotenv
DATABASE_URL=postgresql://devmind:devmind@localhost:5433/devmind
OPENAI_API_KEY=your-key
GITHUB_TOKEN=your-token
# STACKEXCHANGE_API_KEY is optional
```

Start Docker, then run:

```bash
docker compose up -d
uvicorn app.main:app --reload --port 8001
```

Open http://localhost:8001/docs to try the API. Trigger ingestion, wait for it to finish in the logs, then call `/embed` before using hybrid search. BM25 works without embeddings.

## Ask a question

```bash
curl -X POST http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "how does multi-head attention work in transformers"}'
```

`POST /query` accepts `query`, `use_hyde` (default `true`), and `use_reranking` (default `true`). It returns:

- `answer`: text with inline `[Source N]` citations.
- `sources`: numbered sources with document ID, domain, title, URL, and the exact chunk text supplied to the model.
- `query` and `hyde_query`: the original question and the text used for dense retrieval.
- `retrieval_scores`: RRF and reranker scores keyed by source number. Reranker scores are `null` when disabled.
- `latency_ms`: foreground pipeline time, including query enhancement, generation, and logging work (before the final log commit and HTTP delivery).
- `log_id`: saved query ID for polling evaluation results.
- `citation_valid`: whether all recognised source references resolve and no malformed source markers were found.
- `faithfulness_score`: initially `null`; computed after the answer is sent.

The top five results are ordered by score and deduplicated by source domain and parent document, so the final context may contain fewer than five sources. Chunks longer than 800 tokens are trimmed to 600. All context sources are returned, including those the answer doesn't cite.

Generation uses `gpt-4o-mini` with a strict JSON schema for the answer. Source metadata comes from the database, not the model. The prompt requires answers to stay within the context and acknowledge missing information. With no retrieved context, the endpoint skips generation and returns an insufficient-information answer.

The endpoint is non-streaming. Invalid requests return 422; model timeouts return 504; other model API errors or malformed/refused answers return 502. Logs include query-understanding, embedding, retrieval, reranking, context assembly, generation, and total timing. First-use model loading can make the first request slower.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /query` | Answer a question using retrieved sources |
| `GET /query/{log_id}/faithfulness` | Poll the judge status, score, and claim verdicts |
| `GET /logs?limit=20` | Read recent query logs and metrics |
| `GET /health` | Basic API health check |
| `POST /ingest/arxiv` | Ingest arXiv abstracts |
| `POST /ingest/github` | Ingest GitHub READMEs |
| `POST /ingest/stackoverflow` | Ingest questions and accepted answers |
| `POST /embed` | Embed pending chunks |
| `GET /search/bm25` | Keyword search |
| `GET /search/hybrid` | Combined search; optional `rerank=false` and `hyde=false` |
| `GET /stats` | Chunk counts by source and embedding status |

`hyde=false` disables the hypothetical passage; query expansion remains enabled. `/search/bm25` continues to use the original query. Enhancement calls have a 10-second SDK timeout and no automatic retries.

Both search endpoints accept `query` and `k` (1–100, default 5). Hybrid search returns content, source URLs, metadata, and retrieval scores.

```bash
curl 'http://localhost:8001/search/hybrid?query=vector%20similarity%20search&k=5&rerank=false'
```

To compare HyDE on and off, run the same query with `hyde=true` and `hyde=false`:

```bash
curl -G http://localhost:8001/search/hybrid --data-urlencode 'query=mathematical intuition behind attention' -d k=5 -d hyde=true
curl -G http://localhost:8001/search/hybrid --data-urlencode 'query=mathematical intuition behind attention' -d k=5 -d hyde=false
```

Also try `pgvector installation postgres` and `why do transformers struggle with long sequences`. Compare the returned document IDs and content. A change in results alone doesn't prove better relevance; inspect whether the retrieved chunks answer the question.

The query-understanding logger (`app.retrieval.query_understanding`) logs the active mode at INFO and generated text at DEBUG. The hypothetical passage is used only for retrieval, never returned as a sourced answer.

## Current limits

Use one API worker: BM25 lives in memory and rebuilds at startup and after ingestion. Background jobs run inside the API process, so restarting the server interrupts them. Avoid launching overlapping ingestion or embedding jobs.

The chunking changes apply to newly ingested sources. Existing rows are left untouched. Oversized older rows may need rechunking before embedding; the embedding client no longer silently truncates content.

## Generation smoke checks

Tested `/query` against the local corpus of 984 embedded chunks:

| Question | Observed result |
| --- | --- |
| How does multi-head attention work? | Cited arXiv and acknowledged that the retrieved abstracts lacked a detailed explanation. |
| How do I fix CUDA out of memory in PyTorch? | Cited a Stack Overflow thread with memory troubleshooting steps. |
| How do I create a vector index in pgvector? | Returned IVFFlat SQL from the pgvector README. |
| What is retrieval augmented generation? | Summarized RAG using retrieved arXiv abstracts. |
| What are the limitations of transformer attention for long sequences? | Cited arXiv discussion of relative-attention memory costs and computation. |

All five responses returned HTTP 200, and their citation numbers matched returned sources. This was a manual smoke check, not a faithfulness benchmark. In that run the first request took 20.5 seconds and subsequent requests took 4–6 seconds; the under-five-second target is not consistently met yet. Stage logs help distinguish model loading, API latency, and local retrieval costs. A slow total alone does not imply HyDE and expansion ran sequentially.

## Citation checks, faithfulness, and logs

After generation, a pure-Python verifier checks `[Source N]` references against the actual source IDs. It records citation occurrences, invalid IDs, malformed markers, and an approximate count of uncited sentences. Citations placed immediately after a sentence's period are attached to that sentence; fenced code is excluded from sentence counting. Abbreviations, Markdown, and complex punctuation can still confuse this heuristic.

`citation_valid=true` means reference validity only. An answer with no citations can have this flag set to true because it contains no invalid references; check `citation_count` and `uncited_sentences` in the log as well. It does not mean that each sentence is cited or supported. Semantic claim-to-cited-source embedding checks are not implemented.

A background job asks `gpt-4o-mini` to break the answer into atomic factual claims and judge each against the exact context sent to generation. The judge returns JSON verdicts and brief reasons. Python computes `supported claims / total claims`; the model does not supply the final numeric score. Claim extraction and checking share one call, a simpler implementation inspired by the [RAGAS faithfulness approach](https://arxiv.org/html/2309.15217v1#S3). This is not the RAGAS package or a calibrated benchmark.

Example workflow:

```bash
curl -X POST http://localhost:8001/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"how do I create a vector index in pgvector"}'

# Replace 123 with the log_id returned above.
curl http://localhost:8001/query/123/faithfulness
curl 'http://localhost:8001/logs?limit=20'
```

| Polling status | Meaning |
| --- | --- |
| `pending` | The response is saved; evaluation has not finished |
| `completed` | A numeric score and claim verdicts are available |
| `not_applicable` | The judge found no factual claims, such as a pure abstention; score stays null |
| `failed` | Evaluation failed; score stays null and `evaluation_error` identifies the error class |
| `generation_failed` | A caught generation/API error prevented an answer; no judge was scheduled |

Scores are estimates of support in context, not truth probabilities. A wrong citation can coexist with a high faithfulness score if another context source supports the claim. The same model is used for answering and judging, so shared biases and missed claims remain possible. Calibrate against human review before using scores as reliability claims; see [Anthropic's evaluation guidance](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).

`query_logs` stores the question, HyDE passage, expansion, retrieved count before deduplication, source context, retrieval scores, answer, citation diagnostics, configuration flags, stage timings, judge verdicts, and timestamp. JSONB config has a GIN index; timestamps and numeric faithfulness scores have B-tree indexes. `/logs` accepts limits from 1 to 100 and sorts newest first. Recognised generation failures are logged too; request-validation errors do not enter the pipeline.

The background job opens its own database sessions and releases the read connection before waiting for the judge. Judge calls use a 20-second SDK timeout and no retries. `faithfulness_latency_ms` is separate from answer latency. Background tasks are process-local, not durable: a restart may leave a log pending, and no automatic recovery/retry worker is implemented. Keep logs local: `/logs` currently has no authentication and contains question and answer text.

Inspect the latest metrics directly:

```sql
SELECT query, citation_valid, faithfulness_score, faithfulness_status,
       total_latency_ms, stage_latency_ms, config
FROM query_logs
ORDER BY created_at DESC
LIMIT 5;
```

## Day 11 verification

The automated suite now has 46 passing tests, including log persistence across separate sessions, polling states, invalid citations, judge parsing, strict booleans, and generation failure logs. PostgreSQL created the new table and indexes successfully; the 984 source chunks were unchanged.

Six live `/query` requests returned immediately with a null score and a pending polling state, then received persisted evaluations. In the initial run, the five technical questions scored 0.83–1.00. The France question declined to answer and received `not_applicable`. Warm foreground responses took roughly 3.5–8 seconds; the first request took about 22 seconds with model initialization. These measurements are smoke checks, not a benchmark.

Reviewing the stored claims caught a judge mistake: it inverted a CUDA answer's negation and penalized a claim the answer did not make. The prompt was tightened to preserve negation and uncertainty. A controlled live example then scored one supported negative claim and one unsupported performance claim as 0.5. Rejudging the same attention answer also changed its verdict, illustrating that model-based scores need human calibration. The original logs are retained; they were not overwritten to improve the reported numbers.

## Generation limits and reading

Day 11 adds reference validation and background faithfulness scoring. It does not verify the meaning of each citation independently, guarantee complete claim extraction, or establish source correctness. The broader benchmark and retrieval metrics remain for Day 12.

Structured outputs constrain JSON shape, not factual accuracy. See [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs). Streaming can be added later using [OpenAI streaming](https://developers.openai.com/api/docs/guides/streaming-responses) and [FastAPI StreamingResponse](https://fastapi.tiangolo.com/advanced/stream-data/).

Retrieved documents can contain malicious instructions. Corpus poisoning introduces such content into the index; once retrieved, it can influence the answer. The system prompt treats context as untrusted data, but this is only one layer. Production defenses also need trusted ingestion, access controls, output checks, and restricted tool permissions. This generator has no execution tools. See the [OWASP RAG security guide](https://cheatsheetseries.owasp.org/cheatsheets/RAG_Security_Cheat_Sheet.html).

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use fixtures and mocks, so they do not need API keys, a running database, or model downloads.
