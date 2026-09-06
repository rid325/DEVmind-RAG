import logging

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.embeddings import get_embeddings
from app.models import Document


BATCH_SIZE = 50


def embed_pending_documents(db: Session):
    while True:
        pending_docs = (
            db.query(Document)
            .filter(Document.embedding.is_(None))
            .order_by(Document.id)
            .limit(BATCH_SIZE)
            .all()
        )
        if not pending_docs:
            return

        try:
            embeddings = get_embeddings([doc.content for doc in pending_docs])
            if len(embeddings) != len(pending_docs):
                raise ValueError("Embedding count does not match the batch size")
            for doc, embedding in zip(pending_docs, embeddings):
                doc.embedding = embedding
            db.commit()
        except Exception:
            db.rollback()
            logging.exception("Embedding batch failed")
            raise

        logging.info("Embedded %d documents", len(pending_docs))


def run_embedding_job() -> None:
    db = SessionLocal()
    try:
        embed_pending_documents(db)
    finally:
        db.close()
