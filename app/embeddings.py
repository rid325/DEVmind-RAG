from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI


@lru_cache(maxsize=1)
def get_client():
    load_dotenv()
    return OpenAI()


def get_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if any(not text.strip() for text in texts):
        raise ValueError("Cannot embed empty text")

    response = get_client().embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    items = sorted(response.data, key=lambda item: item.index)
    if [item.index for item in items] != list(range(len(texts))):
        raise ValueError("Embedding response does not match the input batch")
    return [item.embedding for item in items]
