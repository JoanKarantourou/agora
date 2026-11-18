import math
import os
from collections.abc import Iterator

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.db import connect
from app.embeddings import QUERY_PREFIX, Embedder, to_pgvector
from app.main import app, get_embedder

SOURCE_ID = "search_sample"
DIMS = 768


def unit(*weights: float) -> list[float]:
    """A unit vector with the given leading components and zeros after."""
    norm = math.sqrt(sum(w * w for w in weights))
    return [w / norm for w in weights] + [0.0] * (DIMS - len(weights))


class FakeModel:
    """Stands in for SentenceTransformer: always returns the first axis, records its input."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, sentences: list[str], *, normalize_embeddings: bool) -> list[list[float]]:
        assert normalize_embeddings
        self.calls.append(sentences)
        return [unit(1.0)]


@pytest.fixture
def conn() -> Iterator[psycopg.Connection]:
    try:
        conn = connect()
    except (RuntimeError, psycopg.OperationalError) as e:
        # CI has a Postgres service, so a missing database there is a real failure.
        if os.environ.get("CI"):
            raise
        pytest.skip(f"database not available: {e}")
    with conn:
        yield conn
        # Only this test's rows are touched, so a local dev database keeps its real chunks.
        with conn.transaction():
            conn.execute("DELETE FROM chunks WHERE source_id = %s", (SOURCE_ID,))


def test_search_returns_nearest_chunks_first(conn: psycopg.Connection) -> None:
    rows = [  # chunk_id suffix, embedding; the query vector is the first axis
        ("exact", unit(1.0)),
        ("halfway", unit(1.0, 1.0)),
        ("orthogonal", unit(0.0, 1.0)),
    ]
    with conn.transaction():
        for i, (name, vector) in enumerate(rows, start=1):
            conn.execute(
                """
                INSERT INTO chunks (chunk_id, source_id, author, work, part,
                                    para_start, para_end, word_count, text, embedding)
                VALUES (%s, %s, 'Nobody', 'Search', 'only', %s, %s, 2, %s, %s::vector)
                """,
                (f"{SOURCE_ID}-{name}", SOURCE_ID, i, i, f"{name} text", to_pgvector(vector)),
            )

    model = FakeModel()
    app.dependency_overrides[get_embedder] = lambda: Embedder(model)
    try:
        # Not used as a context manager, so the startup hook (the real model) never runs.
        response = TestClient(app).get("/search", params={"q": "a question", "k": 2})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert model.calls == [[QUERY_PREFIX + "a question"]]
    hits = response.json()
    assert [h["chunk_id"] for h in hits] == [f"{SOURCE_ID}-exact", f"{SOURCE_ID}-halfway"]
    assert hits[0] == {
        "chunk_id": f"{SOURCE_ID}-exact",
        "author": "Nobody",
        "work": "Search",
        "part": "only",
        "similarity": pytest.approx(1.0, abs=1e-6),
        "text": "exact text",
    }
    assert hits[1]["similarity"] == pytest.approx(math.sqrt(0.5), abs=1e-6)
