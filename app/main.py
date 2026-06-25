from fastapi import FastAPI
from sqlalchemy import text 
from app.database import engine, Base 
from app.models import Document

app = FastAPI(title="DEVMind RAG")

@app.on_event("startup")
async def startup():
    with engine.connect() as con:
        con.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        con.commit()
    Base.metadata.create_all(bind=engine)   

@app.get("/health")
def health():
    return {"status": "ok"}

