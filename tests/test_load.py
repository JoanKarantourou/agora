import os
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from app.config import get_database_url
from scripts.load import load_source, psycopg_url, read_chunks

FIXTURES = Path(__file__).parent / "fixtures" / "load"
SOURCE_ID = "load_sample"


@pytest.fixture
def conn() -> Iterator[psycopg.Connection]:
    try:
        conn = psycopg.connect(psycopg_url(get_database_url()), connect_timeout=5)
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


def count_rows(conn: psycopg.Connection) -> int:
    return conn.execute("SELECT count(*) FROM chunks WHERE source_id = %s", (SOURCE_ID,)).fetchone()[0]


def test_reloading_a_source_replaces_its_rows(conn: psycopg.Connection) -> None:
    v1 = read_chunks(SOURCE_ID, FIXTURES / "v1")
    v2 = read_chunks(SOURCE_ID, FIXTURES / "v2")

    assert load_source(conn, SOURCE_ID, v1) == (0, 3)
    assert count_rows(conn) == 3

    assert load_source(conn, SOURCE_ID, v1) == (3, 3)
    assert count_rows(conn) == 3

    # A shorter version of the same source leaves no stale rows behind.
    assert load_source(conn, SOURCE_ID, v2) == (3, 2)
    assert count_rows(conn) == 2
    texts = [row[0] for row in conn.execute(
        "SELECT text FROM chunks WHERE source_id = %s ORDER BY chunk_id", (SOURCE_ID,)
    )]
    assert texts == [r["text"] for r in v2]
