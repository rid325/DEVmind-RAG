from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, func, Boolean, Float, Index, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector 
from app.database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)       
    domain = Column(String(50), nullable=False) 
    source_url = Column(String(500))             
    metadata_ = Column("metadata", JSONB)        
    embedding = Column(Vector(1536))             
    chunk_index = Column(Integer, default=0)     
    parent_doc_id = Column(String(200))          
    created_at = Column(DateTime, server_default=func.now())


class QueryLog(Base):
    __tablename__ = "query_logs"
    __table_args__ = (
        Index("ix_query_logs_config", "config", postgresql_using="gin"),
        CheckConstraint("faithfulness_score IS NULL OR (faithfulness_score >= 0 AND faithfulness_score <= 1)",
                        name="ck_query_logs_faithfulness_range"),
    )

    id = Column(Integer, primary_key=True)
    query = Column(Text, nullable=False)
    hyde_query = Column(Text)
    expanded_query = Column(Text)
    chunks_retrieved = Column(Integer, nullable=False, default=0)
    reranker_scores = Column(JSONB, nullable=False, default=list)
    answer = Column(Text)
    sources = Column(JSONB, nullable=False, default=list)
    citation_valid = Column(Boolean)
    citation_details = Column(JSONB, nullable=False, default=dict)
    faithfulness_score = Column(Float, index=True)
    faithfulness_status = Column(String(30), nullable=False, default="pending")
    faithfulness_claims = Column(JSONB, nullable=False, default=list)
    evaluation_error = Column(String(100))
    faithfulness_latency_ms = Column(Float)
    total_latency_ms = Column(Float, nullable=False)
    stage_latency_ms = Column(JSONB, nullable=False, default=dict)
    config = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
