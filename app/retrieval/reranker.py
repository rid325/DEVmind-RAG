from functools import lru_cache


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)


def rerank(query: str, hybrid_results: list, top_k: int = 5):
    if not hybrid_results or top_k <= 0:
        return []

    pairs = [[query, document.content] for document, _ in hybrid_results]
    scores = get_model().predict(pairs)
    results = [
        (document, hybrid_score, float(score))
        for (document, hybrid_score), score in zip(hybrid_results, scores)
    ]
    results.sort(key=lambda item: item[2], reverse=True)
    return results[:top_k]
