import logging 
import time 
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import Document
from app.embeddings import get_embeddings


BATCH_SIZE=50
def embed_pending_documents(db: Session):
    while True:
        pending_docs=db.query(Document).filter(Document.embedding.is_(None)).limit(BATCH_SIZE).all()
        if not pending_docs:
            break
        try:
            texts=[doc.content for doc in pending_docs]
            embeddings=get_embeddings(texts)
            for doc,embedding in zip(pending_docs,embeddings):
                doc.embedding=embedding
        except Exception as e:
            logging.error(f"Error embedding documents: {e}")
            break
        
        db.commit()
        logging.info(f"Embedded batch of {len(pending_docs)} documents.")

def run_embedding_job() -> None:
    db = SessionLocal()
    try:
        embed_pending_documents(db)
    finally:
        db.close()