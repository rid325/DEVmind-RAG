import logging
import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session
from app.models import Document
from app.retrieval.tokenizer import tokenize


bm25 = None
document_ids = []

def build_index(db: Session) -> None:
    global bm25, document_ids
    
    logging.info("Building BM25 index...")
    
  
    docs = db.query(Document.id, Document.content).order_by(Document.id).all()
    
    if not docs:
        logging.warning("No documents found in DB. BM25 index will be empty.")
        bm25 = None
        document_ids = []
        return
    
    document_ids = [doc.id for doc in docs]
    
  
    tokenized_corpus = [tokenize(doc.content) for doc in docs]
    
    
    bm25 = BM25Okapi(tokenized_corpus)
    
    logging.info(f"BM25 index built successfully with {len(document_ids)} documents.")

def search(query: str, k: int = 5) -> list[tuple[int, float]]:
    global bm25, document_ids
    
    if bm25 is None or not document_ids:
        logging.warning("BM25 index is not built or is empty.")
        return []
    
    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []
        
    scores = bm25.get_scores(tokenized_query)
    
   
    top_k_indices = np.argsort(scores)[::-1][:k]
    
    results = []
    for idx in top_k_indices:
        score = scores[idx]
        if score > 0:  
            results.append((document_ids[idx], float(score)))
            
    return results
