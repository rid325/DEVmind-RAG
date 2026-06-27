import re


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
                if len(body_text) >= 50:
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
        if len(body_text) >= 50:
            chunks.append({
                "chunk_index": len(chunks),
                "content": f"{repo_name} — {current_title}\n\n{body_text}",
                "section_title": current_title,
            })

    return chunks
