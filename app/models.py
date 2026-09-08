from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, func, Boolean, Float, Index, CheckConstraint, ForeignKey, UniqueConstraint
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


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=False, default="")
    config = Column(JSONB, nullable=False)
    benchmark_version = Column(String(100), nullable=False)
    benchmark_snapshot = Column(JSONB, nullable=False)
    corpus_fingerprint = Column(String(64), nullable=False)
    status = Column(String(30), nullable=False, default="queued")
    error = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    completed_at = Column(DateTime(timezone=True))


class ExperimentResult(Base):
    __tablename__ = "experiment_results"
    __table_args__ = (UniqueConstraint("experiment_id", "benchmark_id"),)

    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False, index=True)
    query_log_id = Column(Integer, ForeignKey("query_logs.id"), nullable=False, unique=True)
    benchmark_id = Column(String(30), nullable=False)
    query = Column(Text, nullable=False)
    config = Column(JSONB, nullable=False)
    retrieval_recall = Column(Float)
    faithfulness_score = Column(Float)
    answer_relevance = Column(Float)
    latency_ms = Column(Float)
    status = Column(String(30), nullable=False)
    error = Column(String(100))
    metric_details = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
