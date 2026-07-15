from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512
)


def rerank(query: str, hybrid_results: list, top_k: int = 5):
    if not hybrid_results:
        return []

    documents = []
    hybrid_scores = []
    pairs = []

    for document, score in hybrid_results:
        documents.append(document)
        hybrid_scores.append(score)

    for document in documents:
        pairs.append([query, document.content])

    reranker_scores = model.predict(pairs)

    combined = list(zip(documents, hybrid_scores, reranker_scores))

    combined.sort(key=lambda x: x[2], reverse=True)

    return combined[:top_k]
