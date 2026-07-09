from sqlalchemy.orm import Session
from app.models import Document

def search_dense(db: Session, query_embedding: list[float], k: int = 20) -> list[tuple[int, float]]:
    distance_expr = Document.embedding.cosine_distance(query_embedding).label("distance")
    results = db.query(Document.id, distance_expr)\
                .order_by(distance_expr)\
                .limit(k)\
                .all()
    
    return [(row.id, float(row.distance)) for row in results]
