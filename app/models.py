from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, func
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
