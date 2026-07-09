from sqlalchemy.orm import Session
from app.models import Document
from app.retrieval.dense import search_dense
from app.retrieval.bm25_index import search as search_bm25
from app.embeddings import get_embeddings

def rrf_merge(dense_results: list[tuple[int, float]], sparse_results: list[tuple[int, float]], rrf_k: int = 60) -> list[tuple[int, float]]:
    rrf_scores = {}
    
    for i, (doc_id, _) in enumerate(dense_results):
        rank = i + 1
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))
        
    for i, (doc_id, _) in enumerate(sparse_results):
        rank = i + 1
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))
        
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results

def search_hybrid(db: Session, query: str, k: int = 20) -> list[tuple[Document, float]]:
    query_vectors = get_embeddings([query])
    if not query_vectors:
        return []
    query_vector = query_vectors[0]
    
    dense_results = search_dense(db, query_vector, k=k)
    sparse_results = search_bm25(query, k=k)
    
    merged_results = rrf_merge(dense_results, sparse_results)
    top_results = merged_results[:k]
    
    doc_ids = [doc_id for doc_id, _ in top_results]
    
    if not doc_ids:
        return []
        
    documents = db.query(Document).filter(Document.id.in_(doc_ids)).all()
    doc_map = {doc.id: doc for doc in documents}
    
    final_results = []
    for doc_id, score in top_results:
        if doc_id in doc_map:
            final_results.append((doc_map[doc_id], score))
            
    return final_results
