# Comparing pipeline configurations

Day 13 adds a fixed 50-question benchmark, a background runner, database metrics, and paired statistical comparisons. The benchmark was created here because a Day 12 harness was not present.

## Run it yourself

Start Docker and the API. Use one worker and omit `--reload` for a long experiment so saving a file does not interrupt it:

```bash
docker compose up -d
source .venv/bin/activate
uvicorn app.main:app --port 8001
```

The questions in [benchmarks.py](../app/evaluation/benchmarks.py) refer to the current development corpus. Each supporting chunk includes its ID, source identity, and a SHA-256 content hash. The POST endpoint checks those labels and requires those chunks to have embeddings. A different corpus returns 422; ingesting arbitrary documents does not make the benchmark valid. If you replace the corpus, review and update the supporting-source labels and increment `BENCHMARK_VERSION` before making new comparisons.

```bash
curl -X POST http://localhost:8001/experiments \
  -H 'Content-Type: application/json' \
  -d '{"name":"baseline","description":"Hybrid retrieval only","config":{"use_hyde":false,"use_reranking":false,"use_expansion":false}}'

curl -X POST http://localhost:8001/experiments \
  -H 'Content-Type: application/json' \
  -d '{"name":"full_pipeline","description":"HyDE, reranking and expansion","config":{"use_hyde":true,"use_reranking":true,"use_expansion":true}}'
```

Both calls return HTTP 202 with `experiment_id`. Replace the example IDs below with those returned by your requests:

```bash
curl http://localhost:8001/experiments/1
curl http://localhost:8001/experiments/2
curl 'http://localhost:8001/experiments/compare?exp_a=1&exp_b=2'
```

The detail endpoint returns status, completed/total/failed counts, and each query's metrics and `query_log_id`. Comparison returns 409 until both runs finish, and also rejects incompatible benchmarks or corpus fingerprints. Unknown IDs return 404. `/experiments/compare` is registered before the integer-ID route.

All three config fields are required JSON booleans. The field name is `use_expansion` throughout the API; unknown keys are rejected. Ordinary `/query` requests default all three flags to true. For `/search/hybrid`, the query parameters are `hyde`, `rerank`, and `expansion`.

| Configuration | `use_hyde` | `use_reranking` | `use_expansion` |
| --- | --- | --- | --- |
| A: baseline | false | false | false |
| B: reranking only | false | true | false |
| C: HyDE only | true | false | false |
| D: full pipeline | true | true | true |

A versus B measures the effect of enabling the existing reranking path. That path also retrieves a larger candidate pool before selecting five results. A versus C measures HyDE with the other enhancements disabled. B versus D changes **both** HyDE and expansion; it cannot isolate HyDE. To isolate HyDE on top of reranking, keep expansion fixed in both runs.

## What happens during a run

1. The POST handler validates source labels and stores an experiment with its config, benchmark version, full question/label snapshot, and corpus fingerprint.
2. A background task waits for the process's experiment lock, then claims the queued row. Experiments run sequentially to reduce local resource contention.
3. It verifies the corpus, rebuilds BM25, and loads the tokenizer and optional reranker before query timing starts.
4. For each benchmark question it calls the same `generate_answer` function used by `/query`, with all three flags. No benchmark keywords or labels are given to the generator.
5. Shared logging code validates citations and commits the answer, context, flags, and timing to `query_logs`. The experiment then waits for that answer's faithfulness evaluation.
6. It computes retrieval recall and keyword coverage, writes an `experiment_results` row linked to the query log, and commits before starting the next question. `tqdm` shows progress in the server terminal.
7. After checking that the corpus is unchanged, it marks the experiment `complete` or `complete_with_errors`.

A generation failure produces a failure query log and result with null metrics; the loop continues. A judge failure leaves faithfulness null while preserving retrieval, keyword coverage, and answer latency. An answer with no factual claims also has null faithfulness, with `not_applicable` in its metric details. These cases are not silently scored as zero.

An unexpected persistence or setup error marks the experiment `failed`; earlier committed results remain readable. Calling the runner again on a non-queued row does nothing. This prevents duplicate execution, but is not restart recovery.

The tasks and lock live inside one API process. Restarting it can leave an experiment `running` or `queued`, and no durable queue or automatic resume exists. Start a new experiment after a restart and retain the interrupted run for inspection. Do not run ingestion, embedding, or unrelated query workloads during timing comparisons. The corpus fingerprint covers document content, embeddings, and source metadata and is checked at the beginning and end; it is not database snapshot isolation throughout the run.

## What the original metrics mean

| Stored field | Calculation | Interpretation |
| --- | --- | --- |
| `retrieval_recall` | Labelled supporting chunk IDs found in the top five / number of labelled supporting chunks | Annotated chunk recall@5; higher is better |
| `faithfulness_score` | Supported extracted claims / all extracted factual claims | Same-model judge estimate of support in the supplied context; higher is better |
| `answer_relevance` | Expected keyword groups matched / expected keyword groups | Keyword coverage proxy; higher is better |
| `latency_ms` | Foreground pipeline elapsed time through log preparation | Lower is better; excludes final log commit, HTTP delivery, background judging, and one-time model loading |

Retrieval is measured **before context deduplication and token trimming**. The result records those top-five document IDs, so this score can be audited independently of the smaller context eventually sent to generation. Labels are known supporting chunks, not exhaustive judgements of every chunk. A different, useful passage may earn no credit. Do not describe this as context precision or exhaustive recall over all relevant corpus content.

Keyword groups use `/` for alternatives. Matching ignores case and punctuation and respects word boundaries. It does not recognize arbitrary paraphrases, understand negation, or establish answer correctness. For example, “not faithful” can still match `faithful`. The database field keeps the brief's `answer_relevance` name, but reports should call it keyword coverage.

The benchmark has 16 arXiv questions, 16 Stack Overflow questions, 15 GitHub questions, and 3 cross-domain questions, spanning easy, medium, and hard tasks. It was built from the current corpus before either run, not sampled independently from real user traffic. Labels and keywords need human review before using this as a release gate or making broader claims.

Faithfulness uses the same `gpt-4o-mini` family as generation. Its claim extraction and support decisions can be wrong or vary between runs. A high score can accompany an incomplete answer; an abstention is not automatically a perfect answer.

## How comparisons are calculated

Results are paired by benchmark question ID and exact query text. For each metric separately, the comparator removes pairs with a missing or nonfinite value or a failed generation. Both reported means and sample standard deviations (`ddof=1`) use that same paired subset. `paired_count` and `excluded_pairs` show how much data was used.

`difference_b_minus_a` is always mean B minus mean A. `improvement_percent` is `(B − A) / abs(A) × 100` for quality scores and `(A − B) / abs(A) × 100` for latency. A zero baseline gives null percentage improvement. A negative number means B is worse under that metric.

The comparator uses a two-sided [SciPy paired t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html). `significant` means unadjusted p < 0.05, independently of whether B is better. `b_is_better` gives direction. Multiple reported metrics create multiple testing opportunities, so `p_value_holm` and `significant_holm` report a Holm adjustment across all reported metrics with finite p-values. Regression gates perform their own adjustment over the five quality-gate metrics.

Fewer than two complete pairs have no p-value. Identical scores have p=1; constant nonzero paired differences have no finite t-test and receive a note. No NaN or infinity is returned as JSON. Standard deviation is spread across questions, not a confidence interval for the mean.

A p-value is not the probability that the improvement is real. These are exploratory comparisons: bounded scores, related questions, incomplete relevance labels, model variability, and sequential run timing limit generalization. Repeat and alternate run order, review answer-level changes, and use independently labelled questions before making a strong performance claim. Equal scores across configurations are possible and do not by themselves establish a wiring bug.

## Database and code map

`experiments` owns configuration, benchmark snapshot, corpus fingerprint, timestamps, and status. `experiment_results` stores per-question metrics and foreign keys to the experiment and query log. `(experiment_id, benchmark_id)` is unique. Each result has a unique query-log link. Experiment IDs and timestamps are indexed. Answers and source text remain in `query_logs`, rather than being copied into experiment rows.

| File | Responsibility |
| --- | --- |
| [benchmarks.py](../app/evaluation/benchmarks.py) | Fixed questions, keywords, supporting source labels |
| [experiment_runner.py](../app/evaluation/experiment_runner.py) | Validation, creation, background loop and per-query persistence |
| [metrics.py](../app/evaluation/metrics.py) | Recall, keyword coverage and paired statistics |
| [comparator.py](../app/evaluation/comparator.py) | Compatible-run checks and comparison response |
| [query_logging.py](../app/evaluation/query_logging.py) | Shared success/failure logging for requests and experiments |
| [test_experiments.py](../tests/test_experiments.py) | Flags, metrics, persistence, failures and endpoint tests |

Inspect stored results:

```sql
SELECT e.id, e.name, e.status, COUNT(er.id) AS queries_run,
       COUNT(er.faithfulness_score) AS scored_answers,
       AVG(er.faithfulness_score) AS avg_faithfulness,
       AVG(er.retrieval_recall) AS avg_recall,
       AVG(er.latency_ms) AS avg_latency_ms
FROM experiments e
LEFT JOIN experiment_results er ON e.id = er.experiment_id
GROUP BY e.id
ORDER BY e.id;

SELECT er.benchmark_id, er.status, er.retrieval_recall,
       er.faithfulness_score, er.answer_relevance, q.answer, q.sources
FROM experiment_results er
JOIN query_logs q ON q.id = er.query_log_id
WHERE er.experiment_id = 1
ORDER BY er.benchmark_id;
```

SQL averages above are descriptive, unpaired averages and can differ from the comparison endpoint when missing values occur on different questions.


## First recorded run

Experiments 1 and 2 both completed 50/50 questions without generation or evaluation failures. The [JSON report](experiments/baseline-vs-full.json) exports the comparison, all 100 per-question metric rows, and environment metadata. Raw answers and contexts remain in PostgreSQL through the recorded query-log IDs.

Annotated recall was 0.96 for both configurations; keyword coverage was 0.90 for both. Judge faithfulness increased from 0.9793 to 1.0000 (paired p=0.0226, Holm p=0.0677). This is inconclusive after correction and is not evidence of perfect factual accuracy. Latency increased from 2.401 to 5.080 seconds on average (111.56% slower, Holm p≈7.59×10⁻²³). See the [README results table](../README.md#first-measured-comparison) for means and sample standard deviations.

The matching retrieval means hide a gain and a loss: q01 retrieved both annotated chunks instead of one; q49 retrieved neither instead of one. Source annotations are incomplete, so read those answers before equating a label miss with an entirely useless response. No labels were changed to improve the measured results.

The baseline/full comparison does not establish which individual component caused these changes. It also runs configurations sequentially, with baseline first. These are measurements on the current development questions and API conditions, not a general speed or quality guarantee.

## Held-out pooled ranking benchmark

The second benchmark is selected with `"benchmark":"heldout_20"`. Its 20
question texts were frozen before candidate retrieval. For each question, the
judgment pool is the union of the baseline and full-pipeline top 10, producing
274 query–chunk labels. Every pooled chunk has a content hash and a grade:

- `0`: irrelevant
- `1`: useful partial evidence
- `2`: direct or essential evidence

The first grading pass used `gpt-4o-mini` as an annotation aid. Every positive
judgment was then reviewed against the passage and permissive adjacent-topic
labels were corrected; negative labels were spot-checked. This is stronger than
keyword matching, but it is still a single-reviewer project benchmark rather
than independently adjudicated human ground truth.

The pool is built from the two configurations being compared. It covers their
observed candidates without pretending that all 984 corpus chunks were judged.
For this reason, the metric is called **pooled Recall@5**. A future system that
retrieves a relevant document outside this fixed pool will receive no credit
until the pool is expanded and labels are versioned again.

Ranking metrics use the first five retrieved document IDs before context
deduplication:

| Metric | Definition |
| --- | --- |
| Precision@5 | Number of grade-1/2 documents in the first five divided by 5 |
| Pooled Recall@5 | Distinct grade-1/2 documents in the first five divided by all grade-1/2 documents in the judged pool |
| MRR | Reciprocal rank of the first grade-1/2 result, or zero when none appears |
| NDCG@5 | Graded gain `(2^grade - 1) / log2(rank + 1)`, divided by the ideal ordering of pooled grades |

The original 50-question development benchmark remains available as
`"benchmark":"development_50"`, which is the default for compatibility. It has
known supporting-source labels but not a pooled 0/1/2 assessment, so
Precision@5, MRR, and NDCG@5 are null for those old-style runs.
Comparisons of those runs report the corresponding gates as
`insufficient_data`; use `heldout_20` for a complete gate decision.

```bash
curl -X POST http://localhost:8001/experiments \
  -H 'Content-Type: application/json' \
  -d '{"name":"heldout_baseline","benchmark":"heldout_20","config":{"use_hyde":false,"use_reranking":false,"use_expansion":false}}'
```

### Regression gates

The comparison response now includes `gates`. Recall, faithfulness,
Precision@5, MRR, and NDCG@5 fail only when configuration B has a negative mean
difference and its paired two-sided t-test remains below alpha 0.05 after Holm
adjustment across the available quality gate metrics. A significant improvement
passes; a nonsignificant decline also passes but remains visible in the table.
Missing or insufficient paired data cannot pass a gate.

Latency uses a separate hard rule because a large slowdown is undesirable even
when noisy: configuration B's p95 foreground latency must be no more than
**8,000ms**. It does not use a significance test. Override the ceiling for a
particular comparison without changing stored results:

```bash
curl 'http://localhost:8001/experiments/compare?exp_a=3&exp_b=4&latency_p95_ceiling_ms=6000'
```

The default ceiling is deliberately explicit in the response. It was chosen as
a practical local-development budget above the earlier full-pipeline mean, not
as an external service-level objective. Revisit it for deployed hardware and
real traffic. Gate results are meaningful only when both experiments use the
same benchmark and corpus, which the comparator already enforces.

### First held-out comparison

Experiments 3 and 4 ran the held-out benchmark sequentially against the same
984-document corpus. Baseline completed 20/20. The full pipeline had one
generation timeout and two separate faithfulness-judge timeouts, so ranking and
latency use 19 pairs and faithfulness uses 17. Failed values remain null.

| Metric | Baseline mean | Full mean | B−A | Paired p-value |
| --- | ---: | ---: | ---: | ---: |
| Pooled Recall@5 | 0.8043 | 0.8538 | +0.0495 | 0.1405 |
| Precision@5 | 0.4000 | 0.4316 | +0.0316 | 0.2680 |
| MRR | 0.9053 | 0.9737 | +0.0684 | 0.0907 |
| NDCG@5 | 0.8177 | 0.8719 | +0.0542 | 0.0963 |
| Faithfulness | 0.9765 | 0.9882 | +0.0118 | 0.3322 |
| Foreground latency | 3.012s | 6.906s | +3.893s | 0.0020 |

All five quality gates pass because there is no significant negative delta.
All five metrics moved in a positive direction, which is a useful signal, but
the 20-question sample does not yet have enough statistical power to confirm
the effects. “Not significant” here means that this sample did not establish a
difference; it does not mean that there is no quality difference. The
overall gate fails because full-pipeline p95 latency was **17.306s**, above the
8s ceiling. Large generation, query-understanding, and embedding outliers drove
the tail; this was a sequential local run over remote APIs, not a load test.
The baseline retrieved no documents outside the judged pool. A fresh HyDE run
made the full pipeline retrieve one unpooled document for h12; it received zero
gain because it had no frozen judgment. This is a concrete limitation of pooled
evaluation when query enhancement is nondeterministic and means the reported
full-pipeline ranking scores may be a slight underestimate.

All three timeouts occurred in the full pipeline: one during generation and two
during faithfulness judging. Each failed call went to the same remote model API,
so the timeouts and latency tail may share an upstream-latency cause, although a
single run cannot prove that. Foreground latency excludes the judge calls and
the failed generation has no latency value, which means the 17.306s p95 does not
fully account for those failures.

The full report, including all per-query metric rows and failure types, is in
[heldout-baseline-vs-full.json](experiments/heldout-baseline-vs-full.json).

### Pull-request gate

The [GitHub Actions workflow](../.github/workflows/rag-evaluation.yml) runs the
20-question baseline followed by the full pipeline for every non-draft pull
request. It uses the same general PR-comment pattern as ReviewBot, but updates a
single comment identified by an HTML marker so synchronization events do not
create duplicates. The aggregate JSON and rendered Markdown are also uploaded
as a workflow artifact. A final workflow step exits nonzero when evaluation
could not complete or `gates.passed` is false.

Configure these repository Actions secrets:

| Secret | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Query understanding, embeddings, generation, and faithfulness judging |
| `CI_DATABASE_URL` | Dedicated writable PostgreSQL/pgvector database with the exact frozen 984-document corpus |

The database requirement is deliberate. Held-out labels reference exact
document IDs and content hashes; an empty PostgreSQL service or freshly scraped
corpus cannot satisfy them. The CI runner does not create or migrate tables.
Provision the schema beforehand and give its database role `SELECT` on
`documents`, read/write access to `experiments`, `experiment_results`, and
`query_logs`, and access to the sequences used by those result tables. This
prevents PR evaluation code from modifying the frozen corpus through its normal
database credentials. Corpus hashes are checked before and after each run.

The workflow uses repository-wide concurrency so two PR evaluations do not
distort timing on the shared corpus. Its explicit job name is kept stable for
branch protection, and reports record the PR head SHA rather than GitHub's
synthetic merge SHA. Configure the
`RAG evaluation / baseline-vs-full` status check as required in the `main`
branch ruleset to prevent merging around a failure. Pull requests from forks do
not receive repository secrets under GitHub's standard `pull_request` security
model and therefore fail with a configuration message instead of executing
untrusted fork code with secrets.

#### Creating `CI_DATABASE_URL`

`CI_DATABASE_URL` is the PostgreSQL connection string for a hosted database
that GitHub-hosted runners can reach. It looks like:

```text
postgresql://devmind_ci:PASSWORD@HOST:5432/DATABASE?sslmode=require
```

Create a PostgreSQL database with pgvector through Neon, Supabase, or another
managed PostgreSQL service, then copy its connection string from the provider
dashboard. A local `localhost:5433` URL will not work from a GitHub-hosted
runner. Copy the existing frozen database so document IDs, content, and vectors
remain unchanged:

```bash
pg_dump "$DATABASE_URL" --format=custom --no-owner --no-acl --file=/tmp/devmind-ci.dump
pg_restore --dbname="$CI_DATABASE_ADMIN_URL" --no-owner --no-acl /tmp/devmind-ci.dump
```

Use the provider's administrator URL only for setup. Create a separate CI login
with `SELECT` on `documents`, `SELECT/INSERT/UPDATE` on `query_logs`,
`experiments`, and `experiment_results`, and usage on their ID sequences. Put
that restricted login's connection string in the GitHub Actions secret named
`CI_DATABASE_URL`. Never commit either URL.

Verify the restricted URL locally before adding the secret:

```bash
DATABASE_URL="$CI_DATABASE_URL" python -m scripts.check_ci_database
```

The same preflight runs in Actions before any model request. It checks table
permissions and validates every held-out document ID and content hash, giving a
short error instead of failing halfway through an evaluation.

#### Gemini key limitation

Google provides an OpenAI-compatible endpoint for Gemini chat and embeddings,
but changing only `OPENAI_API_KEY` to `GEMINI_API_KEY` is not valid for this
benchmark. The frozen corpus contains OpenAI `text-embedding-3-small` vectors;
Gemini embeddings occupy a different vector space even when configured to the
same 1,536 dimensions. A complete Gemini migration must re-embed every document,
rebuild and review the top-10 judgment pool, version the benchmark, and record a
new baseline. Until then, CI needs the provider that created the frozen vectors.
