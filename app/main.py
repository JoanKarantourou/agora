from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Query, Request
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Loaded once at startup; every request reuses it.
    app.state.embedder = Embedder.load()
    yield


app = FastAPI(title="Agora", lifespan=lifespan)


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


class SearchHit(BaseModel):
    chunk_id: str
    author: str
    work: str
    part: str
    similarity: float
    text: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/search")
def search(
    q: str = Query(min_length=1),
    k: int = Query(5, ge=1, le=50),
    embedder: Embedder = Depends(get_embedder),
) -> list[SearchHit]:
    query_vector = to_pgvector(embedder.embed_query(q))
    with connect() as conn:
        rows = conn.execute(SEARCH, {"q": query_vector, "k": k}).fetchall()
    return [
        SearchHit(chunk_id=r[0], author=r[1], work=r[2], part=r[3], similarity=r[4], text=r[5])
        for r in rows
    ]
