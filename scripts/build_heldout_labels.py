"""Build relevance labels for the frozen held-out pool.

This is an annotation aid, not part of the API. It checkpoints after every
question and rejects incomplete model output. Review the resulting judgments
before promoting them to the benchmark fixture.
"""
import hashlib
import json
from pathlib import Path

from app.database import SessionLocal
from app.embeddings import get_client
from app.models import Document


POOL_PATH = Path("/tmp/heldout_pool.json")
OUTPUT_PATH = Path("/tmp/heldout_graded.json")
PROMPT = """Grade passage relevance to the query using only the supplied text.
Grade 2: directly answers the query or contains essential, specific evidence.
Grade 1: useful partial evidence, relevant background, or one part of a
multi-part answer. Grade 0: does not help answer the query. Judge passages
independently. Do not reward shared vocabulary alone. Return every document ID
exactly once."""
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "relevance_grades",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "grades": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "document_id": {"type": "integer"},
                            "grade": {"type": "integer", "enum": [0, 1, 2]},
                            "reason": {"type": "string"},
                        },
                        "required": ["document_id", "grade", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["grades"],
            "additionalProperties": False,
        },
    },
}

# Review corrections after reading the pooled passages. The model-assisted pass
# was deliberately permissive; these overrides remove passages that only share
# vocabulary or discuss an adjacent topic. Any positive grade not listed here
# was also reviewed and retained.
MANUAL_OVERRIDES = {
    "h01": {669: 2},
    "h03": {677: 1, 694: 1, 666: 0},
    "h04": {655: 1, 660: 1, 683: 0, 700: 0},
    "h13": {12: 2, 15: 1, 34: 0, 36: 0, 18: 0, 25: 0, 17: 0, 47: 1, 83: 0},
    "h14": {32: 2, 11: 0, 18: 0, 30: 0, 21: 0, 12: 1, 19: 1, 50: 0},
    "h16": {63: 0, 8: 1, 172: 1, 82: 0, 84: 0, 229: 0, 238: 0},
    "h17": {80: 0},
    "h18": {268: 1, 283: 1, 280: 0, 251: 0},
    "h19": {38: 2, 130: 0, 289: 1, 21: 1, 284: 0, 25: 0},
    "h20": {23: 0, 18: 0, 195: 0, 11: 0, 12: 1, 50: 1, 32: 0,
            13: 1, 459: 1, 201: 0, 247: 0},
}


def main() -> None:
    benchmark = json.loads(POOL_PATH.read_text())
    if OUTPUT_PATH.exists():
        saved = json.loads(OUTPUT_PATH.read_text())
        if saved["version"] == benchmark["version"]:
            saved_by_id = {item["id"]: item for item in saved["questions"]}
            for item in benchmark["questions"]:
                if item["id"] in saved_by_id and "judgments" in saved_by_id[item["id"]]:
                    item["judgments"] = saved_by_id[item["id"]]["judgments"]

    with SessionLocal() as db:
        for number, item in enumerate(benchmark["questions"], 1):
            if "judgments" in item:
                for grade in item["judgments"]:
                    override = MANUAL_OVERRIDES.get(item["id"], {}).get(grade["document_id"])
                    if override is not None:
                        grade["grade"] = override
                        grade["reason"] = "Manually reviewed pooled passage; final grade overrides the assisted draft."
                continue
            passages = []
            for document_id in item["candidate_ids"]:
                document = db.get(Document, document_id)
                passages.append({
                    "document_id": document_id,
                    "domain": document.domain,
                    "source_url": document.source_url,
                    "content": document.content,
                })
            grades = []
            for offset in range(0, len(passages), 6):
                batch = passages[offset:offset + 6]
                expected = {passage["document_id"] for passage in batch}
                for attempt in range(3):
                    response = get_client().with_options(timeout=60, max_retries=1).chat.completions.create(
                        model="gpt-4o-mini",
                        temperature=0,
                        max_completion_tokens=2500,
                        response_format=SCHEMA,
                        messages=[
                            {"role": "system", "content": PROMPT},
                            {"role": "user", "content": json.dumps({"query": item["query"], "passages": batch})},
                        ],
                    )
                    batch_grades = json.loads(response.choices[0].message.content)["grades"]
                    if (len(batch_grades) == len(expected)
                            and {grade["document_id"] for grade in batch_grades} == expected):
                        grades.extend(batch_grades)
                        break
                else:
                    raise ValueError(f"Incomplete labels for {item['id']} batch {offset // 6 + 1}")

            by_id = {passage["document_id"]: passage for passage in passages}
            for grade in grades:
                passage = by_id[grade["document_id"]]
                grade.update(
                    domain=passage["domain"],
                    source_url=passage["source_url"],
                    content_sha256=hashlib.sha256(passage["content"].encode()).hexdigest(),
                )
            for grade in grades:
                override = MANUAL_OVERRIDES.get(item["id"], {}).get(grade["document_id"])
                if override is not None:
                    grade["grade"] = override
                    grade["reason"] = "Manually reviewed pooled passage; final grade overrides the assisted draft."
            item["judgments"] = grades
            OUTPUT_PATH.write_text(json.dumps(benchmark, indent=2) + "\n")
            counts = {value: sum(grade["grade"] == value for grade in grades) for value in (0, 1, 2)}
            print(number, item["id"], counts, flush=True)
        OUTPUT_PATH.write_text(json.dumps(benchmark, indent=2) + "\n")


if __name__ == "__main__":
    main()
