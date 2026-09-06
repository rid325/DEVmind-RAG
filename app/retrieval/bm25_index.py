import logging

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session

from app.models import Document
from app.retrieval.tokenizer import tokenize


# Swap the model and IDs together so searches can finish during a rebuild.
_index = (None, ())


def build_index(db: Session) -> None:
    global _index

    docs = db.query(Document.id, Document.content).order_by(Document.id).all()
    corpus = []
    ids = []
    for doc in docs:
        tokens = tokenize(doc.content)
        if tokens:
            ids.append(doc.id)
            corpus.append(tokens)

    model = BM25Okapi(corpus) if corpus else None
    _index = (model, tuple(ids))
    logging.info("BM25 index contains %d documents", len(ids))


def search(query: str, k: int = 5) -> list[tuple[int, float]]:
    model, ids = _index
    tokens = tokenize(query)
    if model is None or not tokens or k <= 0:
        return []

    scores = model.get_scores(tokens)
    indices = np.argsort(scores)[::-1][:k]
    return [(ids[i], float(scores[i])) for i in indices if scores[i] > 0]
