"""Vector search over the embedded chunks, shared by /search and /ask."""

from pydantic import BaseModel

from app.db import connect
from app.embeddings import Embedder, to_pgvector

SEARCH = """
    SELECT chunk_id, author, work, part, 1 - (embedding <=> %(q)s::vector) AS similarity, text
    FROM chunks
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %(q)s::vector
    LIMIT %(k)s
"""


class SearchHit(BaseModel):
    chunk_id: str
    author: str
    work: str
    part: str
    similarity: float
    text: str


def search_chunks(question: str, k: int, embedder: Embedder) -> list[SearchHit]:
    """The k chunks closest in meaning to the question, best first."""
    query_vector = to_pgvector(embedder.embed_query(question))
    with connect() as conn:
        rows = conn.execute(SEARCH, {"q": query_vector, "k": k}).fetchall()
    return [
        SearchHit(chunk_id=r[0], author=r[1], work=r[2], part=r[3], similarity=r[4], text=r[5])
        for r in rows
    ]
