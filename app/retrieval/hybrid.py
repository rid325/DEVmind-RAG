import asyncio
import logging
from time import perf_counter

from sqlalchemy.orm import Session
from app.models import Document
from app.retrieval.dense import search_dense
from app.retrieval.bm25_index import search as search_bm25
from app.embeddings import get_embeddings
from app.retrieval import reranker
from app.retrieval.query_understanding import EnhancedQuery, process_query

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

def search_hybrid(db: Session, query: str, k: int = 5, rerank: bool = True, use_hyde: bool = True, enhanced: EnhancedQuery | None = None) -> list:
    if not query.strip() or k <= 0:
        return []

    fetch_k = max(20, k) if rerank else k
    # FastAPI runs this synchronous search in a worker thread.
    if enhanced is None:
        enhanced = asyncio.run(process_query(query, use_hyde=use_hyde))
    start = perf_counter()
    query_vectors = get_embeddings([enhanced.hyde_passage])
    if not query_vectors:
        return []
    query_vector = query_vectors[0]
    embedded = perf_counter()

    dense_results = search_dense(db, query_vector, k=fetch_k)
    sparse_results = search_bm25(enhanced.expanded_query, k=fetch_k)

    merged_results = rrf_merge(dense_results, sparse_results)
    top_results = merged_results[:fetch_k]

    doc_ids = [doc_id for doc_id, _ in top_results]

    if not doc_ids:
        return []

    documents = db.query(Document).filter(Document.id.in_(doc_ids)).all()
    doc_map = {doc.id: doc for doc in documents}

    final_results = []
    for doc_id, score in top_results:
        if doc_id in doc_map:
            final_results.append((doc_map[doc_id], score))

    retrieved = perf_counter()
    if rerank:
        final_results = reranker.rerank(query, final_results, top_k=k)
    end = perf_counter()
    logging.info("Retrieval latency ms: embedding=%.0f search=%.0f reranking=%.0f",
                 (embedded-start)*1000, (retrieved-embedded)*1000, (end-retrieved)*1000)
    return final_results[:k]
