import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def get_embeddings(texts: list[str]) -> list[list[float]]:
    # Truncate each text to ~24000 characters to safely stay under the 8192 token limit
    safe_texts = [text[:24000] for text in texts]
    response=client.embeddings.create(
        input=safe_texts,
        model="text-embedding-3-small"
    )
    return[item.embedding for item in response.data]
    