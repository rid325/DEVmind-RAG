from fastapi import FastAPI, Depends, BackgroundTasks
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.database import engine, Base, get_db
from app.models import Document
from app.ingestion.arxiv_ingester import run_arxiv_ingestion

app = FastAPI(title="DevMind RAG")

@app.on_event("startup")
async def startup():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest/arxiv")
def ingest_arxiv(
    background_tasks: BackgroundTasks,
):
    """
    Triggers arXiv ingestion as a background task.
    Returns immediately — ingestion runs behind the scenes.
    This is the right pattern: long-running jobs don't block HTTP responses.
    """
    background_tasks.add_task(run_arxiv_ingestion)
    return {"status": "ingestion started", "message": "check logs for progress"}

@app.get("/stats")
def stats(db: Session = Depends(get_db)):
    """
    Shows how many documents are in DB per domain.
    Your sanity check endpoint — use this constantly.
    """
    total = db.query(Document).count()
    arxiv_count = db.query(Document).filter(Document.domain == "arxiv").count()
    embedded = db.query(Document).filter(Document.embedding != None).count()

    return {
        "total_documents": total,
        "by_domain": {
            "arxiv": arxiv_count,
        },
        "embedded": embedded,
        "pending_embedding": total - embedded
    }
