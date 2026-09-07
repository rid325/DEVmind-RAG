import re


CITATION = re.compile(r"\[Source (-?\d+)\]")


def verify_citations(answer: str, sources: list) -> dict:
    source_ids = {s['source_id'] if isinstance(s, dict) else s.source_id for s in sources}
    references = [int(match) for match in CITATION.findall(answer)]
    invalid = sorted(set(references) - source_ids)
    malformed = [marker for marker in re.findall(r"\[Source\b[^\]]*\]", answer, re.IGNORECASE)
                 if not CITATION.fullmatch(marker)]

    # Attach citations written after a period to the preceding sentence.
    prose = re.sub(r"```.*?```", "", answer, flags=re.DOTALL)
    prose = re.sub(r"([.!?])\s+((?:\[Source -?\d+\]\s*)+)", r" \2\1 ", prose)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", prose)
    uncited = sum(1 for sentence in sentences
                  if re.search(r"\w", CITATION.sub('', sentence)) and not CITATION.search(sentence))
    return {
        "all_citations_valid": not invalid and not malformed,
        "citation_count": len(references),
        "invalid_citations": invalid,
        "uncited_sentences": uncited,
        "malformed_citations": malformed,
    }
