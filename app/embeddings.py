import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def get_embeddings(texts: list[str]) -> list[list[float]]:
    response=client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return[item.embedding for item in response.data]
    