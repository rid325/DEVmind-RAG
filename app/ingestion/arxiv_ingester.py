import arxiv
import logging
from tqdm import tqdm
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import Document
from app.ingestion.cleaner import clean_arxiv_text, clean_title

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEARCH_QUERIES = [
    "retrieval augmented generation",
    "large language models",
    "transformer architecture",
    "vector database embeddings",
    "machine learning systems",
    "neural information retrieval",
    "natural language processing",
    "diffusion models",
    "reinforcement learning from human feedback",
    "prompt engineering"
]

MAX_RESULTS_PER_QUERY = 50


def fetch_arxiv_papers(query: str, max_results: int) -> list[arxiv.Result]:
    """
    Calls arXiv API for a given query.
    Returns a list of arxiv.Result objects.
    """
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3, 
        num_retries=3
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = list(client.results(search))
    logger.info(f"Fetched {len(results)} papers for query: '{query}'")
    return results


def paper_exists(db: Session, arxiv_id: str) -> bool:
    """
    Check if we've already ingested this paper.
    arXiv IDs are unique — use them to avoid duplicates.
    """
    return db.query(Document).filter(
        Document.parent_doc_id == arxiv_id,
        Document.domain == "arxiv"
    ).first() is not None


def result_to_document(result: arxiv.Result) -> Document:
    """
    Converts one arXiv result into a Document model instance.
    We're storing the abstract as content for now.
    Full text PDF parsing comes later — abstracts are enough to start.
    """
    arxiv_id = result.entry_id.split("/")[-1]  

    metadata = {
        "title": clean_title(result.title),
        "authors": [str(a) for a in result.authors[:5]], 
        "published": result.published.isoformat() if result.published else None,
        "categories": result.categories,
        "arxiv_id": arxiv_id,
        "pdf_url": result.pdf_url,
    }

    content = clean_arxiv_text(result.summary)  

    return Document(
        content=content,
        domain="arxiv",
        source_url=result.entry_id,
        metadata_=metadata,
        embedding=None,
        chunk_index=0,
        parent_doc_id=arxiv_id
    )


def ingest_arxiv_papers(db: Session) -> dict:
    """
    Main ingestion function. Loops over all queries,
    fetches papers, deduplicates, and inserts to DB.
    Returns a summary dict so the API endpoint can report results.
    """
    total_fetched = 0
    total_inserted = 0
    total_skipped = 0

    for query in tqdm(SEARCH_QUERIES, desc="Processing queries"):
        papers = fetch_arxiv_papers(query, MAX_RESULTS_PER_QUERY)
        total_fetched += len(papers)

        for result in papers:
            arxiv_id = result.entry_id.split("/")[-1]


            if paper_exists(db, arxiv_id):
                total_skipped += 1
                continue

            doc = result_to_document(result)
            db.add(doc)
            total_inserted += 1


        db.commit()
        logger.info(f"Committed batch for query: '{query}'")

    summary = {
        "total_fetched": total_fetched,
        "total_inserted": total_inserted,
        "total_skipped": total_skipped
    }
    logger.info(f"Ingestion complete: {summary}")
    return summary


def run_arxiv_ingestion() -> dict:
    db = SessionLocal()
    try:
        return ingest_arxiv_papers(db)
    finally:
        db.close()
