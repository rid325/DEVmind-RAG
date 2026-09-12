"""Sequential, in-process experiments with a durable result after each query."""
from datetime import datetime, timezone
from hashlib import sha256
import logging
from threading import Lock
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field, StrictBool
from typing import Literal
from sqlalchemy import text, update
from tqdm import tqdm

from app.database import SessionLocal
from app.models import Document, Experiment, ExperimentResult, QueryLog
from app.generation.generator import QueryRequest, generate_answer, get_encoding
from app.evaluation.benchmarks import BENCHMARKS, BENCHMARK_VERSION
from app.evaluation.heldout_benchmark import HELDOUT_BENCHMARK
from app.evaluation.heldout_questions import HELDOUT_VERSION
from app.evaluation.faithfulness import run_faithfulness_job
from app.evaluation.metrics import annotated_recall, keyword_coverage, ranking_metrics
from app.evaluation.query_logging import record_generation_failure, save_query_log
from app.retrieval.bm25_index import build_index
from app.retrieval.reranker import get_model

logger = logging.getLogger(__name__)
_run_lock = Lock()
FINISHED = {"complete", "complete_with_errors"}


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    use_hyde: StrictBool
    use_reranking: StrictBool
    use_expansion: StrictBool


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=2000)
    config: PipelineConfig
    benchmark: Literal["development_50", "heldout_20"] = "development_50"


def selected_benchmark(name: str):
    if name == "heldout_20":
        return HELDOUT_VERSION, HELDOUT_BENCHMARK
    return BENCHMARK_VERSION, BENCHMARKS


def corpus_fingerprint(db) -> str:
    # Include embeddings and source metadata as well as content: any of these
    # can change the retrieval or the judge's evidence between configurations.
    digest = db.execute(text("""
        SELECT md5(COALESCE(string_agg(
            id::text || ':' || md5(content) || ':' || COALESCE(md5(embedding::text), '')
            || ':' || COALESCE(domain, '') || ':' || COALESCE(source_url, '')
            || ':' || COALESCE(parent_doc_id, '') || ':' || COALESCE(metadata::text, ''),
            '|' ORDER BY id), '')) FROM documents
    """)).scalar_one()
    return digest


def validate_benchmark(db, benchmarks) -> None:
    if not benchmarks or len({item["id"] for item in benchmarks}) != len(benchmarks):
        raise ValueError("Benchmark must contain unique query IDs")
    for item in benchmarks:
        if "judgments" in item:
            source_labels = [
                {"document_id": document_id, "content_sha256": content_hash}
                for document_id, _grade, content_hash in item["judgments"]
            ]
            pooled = set(item["pool"]["baseline"]) | set(item["pool"]["full_pipeline"])
            if pooled != {source["document_id"] for source in source_labels}:
                raise ValueError(f"Judgment pool mismatch for {item['id']}")
        else:
            source_labels = item.get("relevant_sources", [])
        if not source_labels:
            raise ValueError(f"Missing source labels for {item['id']}")
        for source in source_labels:
            doc = db.get(Document, source["document_id"])
            if (doc is None or doc.embedding is None
                    or sha256(doc.content.encode()).hexdigest() != source["content_sha256"]
                    or ("domain" in source and doc.domain != source["domain"])
                    or ("parent_doc_id" in source and doc.parent_doc_id != source["parent_doc_id"])
                    or ("source_url" in source and doc.source_url != source["source_url"])):
                raise ValueError(f"Benchmark source mismatch for {item['id']}: document {source['document_id']}. Update labels for this corpus first.")


def create_experiment(db, request: ExperimentRequest) -> Experiment:
    version, benchmarks = selected_benchmark(request.benchmark)
    validate_benchmark(db, benchmarks)
    experiment = Experiment(
        name=request.name, description=request.description, config=request.config.model_dump(),
        benchmark_version=version, benchmark_snapshot=benchmarks,
        corpus_fingerprint=corpus_fingerprint(db), status="queued",
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    return experiment


def run_experiment(experiment_id: int) -> None:
    # Use one API worker. This lock keeps local experiments from competing for
    # the reranker while timings are being measured. It is not a job queue.
    with _run_lock:
        with SessionLocal() as db:
            claimed = db.execute(update(Experiment).where(
                Experiment.id == experiment_id, Experiment.status == "queued"
            ).values(status="running")).rowcount
            db.commit()
            if not claimed:
                return
            try:
                experiment = db.get(Experiment, experiment_id)
                config = PipelineConfig.model_validate(experiment.config).model_dump()
                benchmarks = experiment.benchmark_snapshot
                fingerprint = experiment.corpus_fingerprint
                if corpus_fingerprint(db) != fingerprint:
                    raise ValueError("Corpus changed after the experiment was queued")
                validate_benchmark(db, benchmarks)
                build_index(db)
                get_encoding()
                if config["use_reranking"]:
                    get_model()  # Exclude one-time model loading from query latency.
                db.commit()
                had_errors = False
                for item in tqdm(benchmarks, desc=f"Experiment {experiment_id}"):
                    request = QueryRequest(query=item["query"], **config)
                    start = perf_counter()
                    result = None
                    try:
                        result = generate_answer(db, request.query, **config)
                    except Exception as exc:
                        logger.warning("Experiment %s query %s failed: %s", experiment_id, item["id"], type(exc).__name__)
                        log_id = record_generation_failure(db, request, start, exc)
                    else:
                        log_id = save_query_log(db, request, result, start)
                        run_faithfulness_job(log_id)
                    db.expire_all()
                    log = db.get(QueryLog, log_id)
                    failed = result is None
                    status = "failed" if failed else "evaluation_failed" if log.faithfulness_status == "failed" else "complete"
                    had_errors |= status != "complete"
                    retrieved = result.retrieved_document_ids if result else []
                    if "judgments" in item:
                        judgments = {int(document_id): int(grade)
                                     for document_id, grade, _content_hash in item["judgments"]}
                        ranking = ranking_metrics(retrieved, judgments) if not failed else {
                            "retrieval_precision": None, "retrieval_recall": None,
                            "mrr": None, "ndcg": None,
                        }
                        relevant = [document_id for document_id, grade in judgments.items() if grade > 0]
                    else:
                        relevant = [source["document_id"] for source in item["relevant_sources"]]
                        ranking = {"retrieval_precision": None,
                                   "retrieval_recall": annotated_recall(retrieved, relevant) if not failed else None,
                                   "mrr": None, "ndcg": None}
                    db.add(ExperimentResult(
                        experiment_id=experiment_id, query_log_id=log_id,
                        benchmark_id=item["id"], query=item["query"], config=config,
                        retrieval_recall=ranking["retrieval_recall"],
                        faithfulness_score=log.faithfulness_score,
                        answer_relevance=keyword_coverage(result.answer, item.get("expected_keywords", [])) if not failed else None,
                        latency_ms=log.total_latency_ms if not failed else None,
                        status=status, error=log.evaluation_error,
                        metric_details={"relevant_document_ids": relevant, "retrieved_document_ids": retrieved,
                                        "retrieval_precision": ranking["retrieval_precision"],
                                        "mrr": ranking["mrr"], "ndcg": ranking["ndcg"],
                                        "faithfulness_status": log.faithfulness_status},
                    ))
                    db.commit()
                if corpus_fingerprint(db) != fingerprint:
                    raise ValueError("Corpus changed during the experiment; results cannot be compared")
                experiment = db.get(Experiment, experiment_id)
                experiment.status = "complete_with_errors" if had_errors else "complete"
                experiment.completed_at = datetime.now(timezone.utc)
                db.commit()
            except Exception as exc:
                db.rollback()
                logger.exception("Experiment %s stopped", experiment_id)
                experiment = db.get(Experiment, experiment_id)
                experiment.status = "failed"
                experiment.error = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                experiment.completed_at = datetime.now(timezone.utc)
                db.commit()
