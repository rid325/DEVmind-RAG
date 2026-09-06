import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, Depends, BackgroundTasks, Query, HTTPException
from openai import APIError, APITimeoutError
from app.generation.generator import GenerationError, QueryRequest, QueryResponse, generate_answer
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.database import engine, Base, get_db
from app.models import Document
from app.ingestion.arxiv_ingester import run_arxiv_ingestion
from app.ingestion.stackoverflow_ingester import run_stackoverflow_ingestion
from app.ingestion.github_ingester import run_github_ingestion
from app.ingestion.embedder import run_embedding_job
from app.retrieval.bm25_index import build_index, search
from app.retrieval.hybrid import search_hybrid
from app.database import SessionLocal

@asynccontextmanager
async def lifespan(app: FastAPI):
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)


    db = SessionLocal()
    try:
        build_index(db)
    finally:
        db.close()

    yield


app = FastAPI(title="DevMind RAG", lifespan=lifespan)


def validate_query(query: Annotated[str, Query(min_length=1, max_length=2000)]) -> str:
    query = query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Query must not be blank")
    return query


SearchQuery = Annotated[str, Depends(validate_query)]
ResultCount = Annotated[int, Query(ge=1, le=100)]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest/arxiv")
def ingest_arxiv(
    background_tasks: BackgroundTasks,
):
    background_tasks.add_task(run_arxiv_ingestion)
    return {"status": "ingestion started", "message": "check logs for progress"}

@app.post("/ingest/stackoverflow")
def ingest_stackoverflow(
    background_tasks: BackgroundTasks,
):
    background_tasks.add_task(run_stackoverflow_ingestion)
    return {"status": "ingestion started", "message": "check logs for progress"}

@app.post("/ingest/github")
def ingest_github(
    background_tasks: BackgroundTasks,
):
    background_tasks.add_task(run_github_ingestion)
    return {"status": "ingestion started", "message": "check logs for progress"}

@app.post("/embed")
def embed(
    background_tasks: BackgroundTasks,
):
    background_tasks.add_task(run_embedding_job)
    return {"status": "embedding job started", "message": "check logs for progress"}

@app.get("/search/bm25")
def search_bm25(query: SearchQuery, k: ResultCount = 5, db: Session = Depends(get_db)):
    results = search(query, k)

    response = []
    for doc_id, score in results:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            response.append({
                "id": doc.id,
                "domain": doc.domain,
                "score": score,
                "source_url": doc.source_url,
                "metadata": doc.metadata_,
                "content": doc.content
            })

    return {"results": response}

@app.get("/search/hybrid")
def hybrid_search_endpoint(query: SearchQuery, k: ResultCount = 5, rerank: bool = True, hyde: bool = True, db: Session = Depends(get_db)):
    start_time = time.time()

    results = search_hybrid(db, query, k=k, rerank=rerank, use_hyde=hyde)

    end_time = time.time()
    logging.info("Search completed in %.3f seconds (rerank=%s, hyde=%s)", end_time - start_time, rerank, hyde)

    response = []
    for item in results:
        doc = item[0]
        hybrid_score = item[1]

        result_dict = {
            "id": doc.id,
            "domain": doc.domain,
            "source_url": doc.source_url,
            "metadata": doc.metadata_,
            "content": doc.content,
            "content_preview": doc.content[:200],
            "rrf_score": hybrid_score
        }

        if len(item) == 3:
            result_dict["reranker_score"] = float(item[2])

        response.append(result_dict)

    return {"results": response}


@app.get("/stats")
def stats(db: Session = Depends(get_db)):
    total = db.query(Document).count()
    arxiv_count = db.query(Document).filter(Document.domain == "arxiv").count()
    stackoverflow_count = db.query(Document).filter(Document.domain == "stackoverflow").count()
    github_count = db.query(Document).filter(Document.domain == "github").count()
    embedded = db.query(Document).filter(Document.embedding.is_not(None)).count()

    return {
        "total_documents": total,
        "by_domain": {
            "arxiv": arxiv_count,
            "stackoverflow": stackoverflow_count,
            "github": github_count,
        },
        "embedded": embedded,
        "pending_embedding": total - embedded
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest, db: Session = Depends(get_db)):
    try:
        return generate_answer(db, request.query, use_hyde=request.use_hyde, use_reranking=request.use_reranking)
    except APITimeoutError as exc:
        raise HTTPException(status_code=504, detail="The model request timed out. Please try again.") from exc
    except (APIError, GenerationError) as exc:
        logging.warning("Query failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Could not generate an answer. Please try again.") from exc
