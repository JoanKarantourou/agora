"""Replace each source's rows in the chunks table with data/chunks/<id>.jsonl."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg  # noqa: E402

from app.config import get_database_url  # noqa: E402
from chunk import CHUNKS_DIR  # noqa: E402
from download import load_sources  # noqa: E402

COLUMNS = [
    "chunk_id", "source_id", "author", "work", "translator", "part",
    "para_start", "para_end", "word_count", "text",
]
CONNECT_TIMEOUT = 10  # seconds; fail fast instead of hanging when the database is down
INSERT = f"INSERT INTO chunks ({', '.join(COLUMNS)}) VALUES ({', '.join(['%s'] * len(COLUMNS))})"


def psycopg_url(url: str) -> str:
    """psycopg wants a plain libpq URL, without SQLAlchemy's "+psycopg" driver suffix."""
    return re.sub(r"^postgresql\+\w+://", "postgresql://", url)


def connect() -> psycopg.Connection:
    return psycopg.connect(psycopg_url(get_database_url()), connect_timeout=CONNECT_TIMEOUT)


def read_chunks(source_id: str, chunks_dir: Path = CHUNKS_DIR) -> list[dict]:
    with (chunks_dir / f"{source_id}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_source(conn: psycopg.Connection, source_id: str, records: list[dict]) -> tuple[int, int]:
    """Delete and re-insert one source's chunks in a single transaction."""
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE source_id = %s", (source_id,))
        deleted = cur.rowcount
        cur.executemany(INSERT, [[r[c] for c in COLUMNS] for r in records])
        return deleted, len(records)


def main() -> None:
    with connect() as conn:
        for source in load_sources():
            deleted, inserted = load_source(conn, source["id"], read_chunks(source["id"]))
            print(f"load  {source['id']}: {deleted} deleted, {inserted} inserted")


if __name__ == "__main__":
    main()
