import os
import logging
import time
from html import unescape

import requests
from tqdm import tqdm
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import Document
from app.ingestion.cleaner import clean_stackoverflow_text, clean_title


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STACKOVERFLOW_TAGS = [
    "python",
    "machine-learning",
    "nlp",
    "pytorch",
    "transformers",
    "vector-database",
    "langchain",
    "openai-api",
]

STACKEXCHANGE_API_URL = "https://api.stackexchange.com/2.3/questions"
STACKEXCHANGE_ANSWERS_URL = "https://api.stackexchange.com/2.3/answers/{ids}"
PAGE_SIZE = 25
REQUEST_TIMEOUT_SECONDS = 20


def build_stackoverflow_params(tag: str) -> dict:
    params = {
        "site": "stackoverflow",
        "tagged": tag,
        "sort": "votes",
        "order": "desc",
        "pagesize": PAGE_SIZE,
        "filter": "withbody",
    }

    api_key = os.getenv("STACKEXCHANGE_API_KEY")
    if api_key:
        params["key"] = api_key

    return params


def build_answer_params() -> dict:
    params = {
        "site": "stackoverflow",
        "filter": "withbody",
    }

    api_key = os.getenv("STACKEXCHANGE_API_KEY")
    if api_key:
        params["key"] = api_key

    return params


def handle_rate_limit(response_data: dict) -> None:
    quota_remaining = response_data.get("quota_remaining")
    if quota_remaining is not None:
        logger.info(f"Stack Exchange quota remaining: {quota_remaining}")

    backoff = response_data.get("backoff")
    if backoff:
        logger.info(f"Stack Exchange requested backoff for {backoff} seconds")
        time.sleep(backoff)


def fetch_stackoverflow_questions(tag: str) -> list[dict]:
    response = requests.get(
        STACKEXCHANGE_API_URL,
        params=build_stackoverflow_params(tag),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    data = response.json()
    handle_rate_limit(data)

    questions = data.get("items", [])
    logger.info(f"Fetched {len(questions)} Stack Overflow questions for tag: '{tag}'")
    return questions


def fetch_accepted_answers(answer_ids: list[int]) -> dict[int, dict]:
    if not answer_ids:
        return {}

    ids = ";".join(str(answer_id) for answer_id in answer_ids)
    response = requests.get(
        STACKEXCHANGE_ANSWERS_URL.format(ids=ids),
        params=build_answer_params(),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    data = response.json()
    handle_rate_limit(data)

    answers = {
        answer["answer_id"]: answer
        for answer in data.get("items", [])
        if answer.get("answer_id")
    }
    logger.info(f"Fetched {len(answers)} accepted Stack Overflow answers")
    return answers


def attach_accepted_answers(questions: list[dict]) -> list[dict]:
    accepted_answer_ids = [
        question["accepted_answer_id"]
        for question in questions
        if question.get("accepted_answer_id")
    ]
    answers_by_id = fetch_accepted_answers(accepted_answer_ids)

    for question in questions:
        accepted_answer_id = question.get("accepted_answer_id")
        accepted_answer = answers_by_id.get(accepted_answer_id)
        question["answers"] = [accepted_answer] if accepted_answer else []

    return questions


def get_accepted_answer(question: dict) -> dict | None:
    accepted_answer_id = question.get("accepted_answer_id")
    if not accepted_answer_id:
        return None

    for answer in question.get("answers", []):
        if answer.get("answer_id") == accepted_answer_id:
            return answer

    return None


def build_stackoverflow_content(question: dict) -> str | None:
    accepted_answer = get_accepted_answer(question)
    if not accepted_answer:
        return None

    title = clean_title(unescape(question.get("title", "")))
    question_body = clean_stackoverflow_text(question.get("body", ""))
    answer_body = clean_stackoverflow_text(accepted_answer.get("body", ""))

    if not question_body or not answer_body:
        return None

    return (
        f"Question: {title}\n\n"
        f"{question_body}\n\n"
        f"Answer:\n"
        f"{answer_body}"
    )

def build_stackoverflow_metadata(question: dict) -> dict:
    return {
        "question_id": question.get("question_id"),
        "accepted_answer_id": question.get("accepted_answer_id"),
        "score": question.get("score"),
        "view_count": question.get("view_count"),
        "tags": question.get("tags", []),
        "creation_date": question.get("creation_date"),
        "answer_count": question.get("answer_count"),
        "link": question.get("link"),
        "is_answered": question.get("is_answered"),
    }

def question_to_document(question: dict) -> Document | None:
    content = build_stackoverflow_content(question)
    question_id = question.get("question_id")

    if not content or not question_id:
        return None

    return Document(
        content=content,
        domain="stackoverflow",
        source_url=question.get("link"),
        metadata_=build_stackoverflow_metadata(question),
        embedding=None,
        chunk_index=0,
        parent_doc_id=str(question_id),
    )


def stackoverflow_thread_exists(db: Session, question_id: int | str) -> bool:
    return db.query(Document).filter(
        Document.parent_doc_id == str(question_id),
        Document.domain == "stackoverflow",
    ).first() is not None


def ingest_stackoverflow_threads(db: Session) -> dict:
    total_fetched = 0
    total_inserted = 0
    total_skipped = 0
    total_invalid = 0  
    total_missing_accepted_answer = 0  

    for tag in tqdm(STACKOVERFLOW_TAGS, desc="Processing Stack Overflow tags"):
        questions = fetch_stackoverflow_questions(tag)
        total_fetched += len(questions)
        questions = attach_accepted_answers(questions)

        for question in questions:
            question_id = question.get("question_id")

            if not question_id:
                total_invalid += 1
                continue

            if stackoverflow_thread_exists(db, question_id):
                total_skipped += 1
                continue

            doc = question_to_document(question)
            if not doc:
                total_missing_accepted_answer += 1
                continue

            db.add(doc)
            total_inserted += 1

        db.commit()
        logger.info(f"Committed Stack Overflow batch for tag: '{tag}'")

    summary = {
        "total_fetched": total_fetched,
        "total_inserted": total_inserted,
        "total_skipped_duplicate": total_skipped,
        "total_invalid": total_invalid,
        "total_missing_accepted_answer": total_missing_accepted_answer,
    }
    logger.info(f"Stack Overflow ingestion complete: {summary}")
    return summary


def run_stackoverflow_ingestion() -> dict:
    db = SessionLocal()
    try:
        return ingest_stackoverflow_threads(db)
    finally:
        db.close()

