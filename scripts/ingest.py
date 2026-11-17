"""Run the whole ingestion pipeline: download -> clean -> chunk -> load.

Stops at the first step that fails and exits with status 1.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunk import chunk_file  # noqa: E402
from clean import clean_file  # noqa: E402
from download import RAW_DIR, download, load_sources  # noqa: E402
from load import connect, load_source, read_chunks  # noqa: E402


def step_download(sources: list[dict]) -> str:
    present = sum((RAW_DIR / f"{s['id']}.txt").exists() for s in sources)
    for source in sources:
        download(source)
    return f"{len(sources) - present} fetched, {present} already present"


def step_clean(sources: list[dict]) -> str:
    chars = sum(len(clean_file(s).read_text(encoding="utf-8")) for s in sources)
    return f"{chars:,} chars of clean text"


def step_chunk(sources: list[dict]) -> str:
    chunks = 0
    for source in sources:
        with chunk_file(source).open(encoding="utf-8") as f:
            chunks += sum(1 for line in f if line.strip())
    return f"{chunks} chunks"


def step_load(sources: list[dict]) -> str:
    deleted = inserted = 0
    with connect() as conn:
        for source in sources:
            d, i = load_source(conn, source["id"], read_chunks(source["id"]))
            print(f"load  {source['id']}: {d} deleted, {i} inserted")
            deleted, inserted = deleted + d, inserted + i
    return f"{deleted} deleted, {inserted} inserted"


STEPS = [("download", step_download), ("clean", step_clean), ("chunk", step_chunk), ("load", step_load)]


def main() -> int:
    sources = load_sources()
    for name, step in STEPS:
        try:
            summary = step(sources)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            return 1
        print(f"[{name}] ok: {len(sources)} source(s), {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
