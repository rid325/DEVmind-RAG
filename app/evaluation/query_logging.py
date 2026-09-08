import time

from sqlalchemy.orm import Session
from app.models import QueryLog
from app.generation.generator import QueryRequest, QueryResponse
from app.evaluation.citation_verifier import verify_citations


def record_generation_failure(db: Session, request: QueryRequest, start: float, error: Exception):
    db.rollback()
    log = QueryLog(
        query=request.query,
        total_latency_ms=round((time.perf_counter() - start) * 1000, 2),
        faithfulness_status="generation_failed",
        evaluation_error=type(error).__name__,
        config={"use_hyde": request.use_hyde, "use_reranking": request.use_reranking, "use_expansion": request.use_expansion},
    )
    db.add(log)
    db.flush()
    log_id = log.id
    db.commit()
    return log_id


def save_query_log(db: Session, request: QueryRequest, result: QueryResponse, start: float) -> int:
    citation_start = time.perf_counter()
    citations = verify_citations(result.answer, result.sources)
    result.citation_valid = citations["all_citations_valid"]
    stages = dict(result.stage_latency_ms)
    stages["citation_verification"] = round((time.perf_counter() - citation_start) * 1000, 2)
    log_start = time.perf_counter()
    log = QueryLog(
        query=request.query,
        hyde_query=result.hyde_query,
        expanded_query=result.expanded_query,
        chunks_retrieved=result.chunks_retrieved,
        reranker_scores=[score.model_dump() for score in result.retrieval_scores],
        answer=result.answer,
        sources=[source.model_dump() for source in result.sources],
        citation_valid=result.citation_valid,
        citation_details=citations,
        faithfulness_status="pending",
        total_latency_ms=result.latency_ms,
        config={"use_hyde": request.use_hyde, "use_reranking": request.use_reranking, "use_expansion": request.use_expansion},
    )
    db.add(log)
    db.flush()
    result.log_id = log.id
    stages["logging"] = round((time.perf_counter() - log_start) * 1000, 2)
    result.latency_ms = round((time.perf_counter() - start) * 1000, 2)
    log.total_latency_ms = result.latency_ms
    log.stage_latency_ms = stages
    db.commit()
    return result.log_id
