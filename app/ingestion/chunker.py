import re


CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200


def split_text(content: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(content):
        end = min(start + CHUNK_SIZE, len(content))
        if end < len(content):
            boundary = content.rfind(" ", start + CHUNK_SIZE // 2, end)
            if boundary != -1:
                end = boundary
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(content):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def split_document(doc):
    from app.models import Document

    return [
        Document(
            content=content,
            domain=doc.domain,
            source_url=doc.source_url,
            metadata_=doc.metadata_,
            parent_doc_id=doc.parent_doc_id,
            chunk_index=index,
        )
        for index, content in enumerate(split_text(doc.content))
    ]


def chunk_readme(content: str, repo_name: str) -> list[dict]:
    lines = content.splitlines()
    chunks = []
    current_title = ""
    current_body = []
    in_code_block = False

    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block

        if re.match(r"^#+\s", line) and not in_code_block:
            if current_body:
                body_text = "\n".join(current_body).strip()
                if body_text:
                    chunks.append({
                        "chunk_index": len(chunks),
                        "content": f"{repo_name} — {current_title}\n\n{body_text}",
                        "section_title": current_title,
                    })
            current_title = re.sub(r"^#+\s*", "", line).strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        body_text = "\n".join(current_body).strip()
        if body_text:
            chunks.append({
                "chunk_index": len(chunks),
                "content": f"{repo_name} — {current_title}\n\n{body_text}",
                "section_title": current_title,
            })

    bounded_chunks = []
    for chunk in chunks:
        for content in split_text(chunk["content"]):
            bounded_chunks.append({
                **chunk,
                "content": content,
                "chunk_index": len(bounded_chunks),
            })
    return bounded_chunks
