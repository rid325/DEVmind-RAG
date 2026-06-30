from fastapi import FastAPI, Depends, BackgroundTasks
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.database import engine, Base, get_db
from app.models import Document
from app.ingestion.arxiv_ingester import run_arxiv_ingestion
from app.ingestion.stackoverflow_ingester import run_stackoverflow_ingestion
from app.ingestion.github_ingester import run_github_ingestion
from app.ingestion.embedder import run_embedding_job
from app.retrieval.bm25_index import build_index, search
from app.database import SessionLocal

app = FastAPI(title="DevMind RAG")

@app.on_event("startup")
async def startup():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
    
    # Build BM25 index on startup
    db = SessionLocal()
    try:
        build_index(db)
    finally:
        db.close()

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
def search_bm25(query: str, k: int = 5, db: Session = Depends(get_db)):
    results = search(query, k)
    
    response = []
    for doc_id, score in results:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            response.append({
                "id": doc.id,
                "domain": doc.domain,
                "score": score,
                "content": doc.content
            })
            
    return {"results": response}

@app.get("/stats")
def stats(db: Session = Depends(get_db)):
    total = db.query(Document).count()
    arxiv_count = db.query(Document).filter(Document.domain == "arxiv").count()
    stackoverflow_count = db.query(Document).filter(Document.domain == "stackoverflow").count()
    github_count = db.query(Document).filter(Document.domain == "github").count()
    embedded = db.query(Document).filter(Document.embedding != None).count()

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
