"""Collect baseline/full top-10 candidates for frozen held-out questions.

Run the API on localhost:8001 first. This script writes a review artifact to
/tmp; it does not modify the checked-in benchmark.
"""
import json
from pathlib import Path

import httpx

from app.evaluation.heldout_questions import HELDOUT_QUESTIONS, HELDOUT_VERSION


OUTPUT_PATH = Path("/tmp/heldout_pool.json")
CONFIGS = {
    "baseline": {"rerank": False, "hyde": False, "expansion": False},
    "full_pipeline": {"rerank": True, "hyde": True, "expansion": True},
}


def main() -> None:
    output = {
        "version": HELDOUT_VERSION,
        "pooling": {"depth": 10, "configs": list(CONFIGS)},
        "questions": [],
    }
    with httpx.Client(base_url="http://127.0.0.1:8001", timeout=90) as client:
        for number, item in enumerate(HELDOUT_QUESTIONS, 1):
            pools = {}
            for name, config in CONFIGS.items():
                response = client.get(
                    "/search/hybrid",
                    params={"query": item["query"], "k": 10, **config},
                )
                response.raise_for_status()
                pools[name] = [result["id"] for result in response.json()["results"]]
            candidate_ids = list(dict.fromkeys(pools["baseline"] + pools["full_pipeline"]))
            output["questions"].append({**item, "pool": pools, "candidate_ids": candidate_ids})
            print(number, item["id"], len(candidate_ids), flush=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
