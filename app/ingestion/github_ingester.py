import os
import time
import logging
import base64
import requests
from tqdm import tqdm
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import Document
from app.ingestion.cleaner import clean_markdown
from app.ingestion.chunker import chunk_readme

GITHUB_API_BASE="https://api.github.com"
REQUEST_TIMEOUT= 15
SLEEP_BETWEEN_REPOS= 0.5
GITHUB_REPOS = [
    "facebookresearch/faiss",
    "huggingface/transformers",
    "langchain-ai/langchain",
    "qdrant/qdrant",
    "pgvector/pgvector",
    "openai/openai-python",
    "microsoft/DeepSpeed",
    "ray-project/ray",
    "pytorch/pytorch",
    "keras-team/keras",
    "milvus-io/milvus",
    "chroma-core/chroma",
    "nomic-ai/gpt4all",
    "ggerganov/llama.cpp",
    "ollama/ollama",
    "vllm-project/vllm",
    "BerriAI/litellm",
    "run-llama/llama_index",
    "pinecone-io/pinecone-python-client",
    "explosion/spaCy",
]

def build_headers() -> dict:
    token= os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN environment variable is not set")
    return {
        "Authorization": f"Bearer {token}",
        "Accept" : "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

def fetch_repo_metadata(owner: str, repo: str) -> dict | None:
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}"
    response=requests.get(url=url, headers=build_headers(), timeout=REQUEST_TIMEOUT)
    if response.status_code !=200:
        logging.warning(f"Failed to fetch metadata for {owner}/{repo} status code: {response.status_code}")
        return None

    return response.json()

def fetch_readme_content(owner: str, repo: str) -> str | None:
    url=f"{GITHUB_API_BASE}/repos/{owner}/{repo}/readme"
    response=requests.get(url=url, headers=build_headers(), timeout=REQUEST_TIMEOUT)

    if response.status_code !=200:
        logging.warning(f"Failed to fetch content for {owner}/{repo} status code:{response.status_code}")
        return None
    decoded_content=base64.b64decode(response.json()["content"]).decode("utf-8")
    return decoded_content

def repo_exists(db: Session, repo_name : str) -> bool:
    return db.query(Document).filter(
        Document.parent_doc_id == repo_name,
        Document.domain == "github"
    ).first() is not None

def build_github_metadata(repo_meta: dict, section_title: str, repo_name: str) -> dict:
    return{
        "repo_name" : repo_name,
        "section_title": section_title,
        "stars" : repo_meta["stargazers_count"],
        "description" : repo_meta.get("description"),
        "html_url" : repo_meta.get("html_url")
    }

def ingest_github_repos(db: Session) -> dict:
    total_repos_fetched=0
    total_chunks_inserted=0
    total_skipped=0
    for repo_name in tqdm(GITHUB_REPOS, desc="Processing GitHub repositories"):
        owner, repo = repo_name.split("/")
        if repo_exists(db,repo_name):
            total_skipped+=1
            continue
        repo_meta=fetch_repo_metadata(owner,repo)
        if not repo_meta:
            continue
        total_repos_fetched+=1
        raw_readme=fetch_readme_content(owner,repo)
        if not raw_readme:
            continue
        clean_readme=clean_markdown(raw_readme)
        chunks=chunk_readme(clean_readme,repo_name)
        repo_chunks_count = 0
        for chunk in chunks:
            if len(chunk["content"]) < 50 :
                continue
            doc=Document(
                content=chunk["content"],
                domain="github",
                source_url=repo_meta.get("html_url"),
                metadata_= build_github_metadata(repo_meta, chunk["section_title"], repo_name),
                embedding= None,
                chunk_index= chunk["chunk_index"],
                parent_doc_id=repo_name
            )
            db.add(doc)
            repo_chunks_count += 1
        total_chunks_inserted+=repo_chunks_count
        time.sleep(SLEEP_BETWEEN_REPOS)
        db.commit()
        logging.info(f"Ingested {repo_name} ({repo_chunks_count} chunks)")
    return {
        "total_repos_fetched" : total_repos_fetched,
        "total_chunks_inserted" : total_chunks_inserted,
        "total_skipped" : total_skipped
    }

def run_github_ingestion() -> dict:
    db = SessionLocal()
    try:
        return ingest_github_repos(db)
    finally:
        db.close()