"""Ask a few real questions and check the expected passage comes back in the top 5.

Uses the real model and the local database (run embed_chunks.py first); not part of CI.
Exits with status 1 if any expected chunk is missed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embeddings import Embedder  # noqa: E402
from app.main import search  # noqa: E402

K = 5
CASES = [  # question, expected chunk_id
    ("a life without questioning isn't worth living", "plato_apology-054"),  # "the unexamined life"
    ("the oracle said no one is wiser than Socrates", "plato_apology-010"),  # Chaerephon asks the oracle
    ("I know that I don't know anything", "plato_apology-012"),  # "neither know nor think that I know"
    ("death may be like a dreamless sleep", "plato_apology-059"),  # sleep "undisturbed even by dreams"
]


def main() -> None:
    embedder = Embedder.load()
    misses = 0
    for question, expected in CASES:
        hits = search(q=question, k=K, embedder=embedder)
        ids = [h.chunk_id for h in hits]
        if expected in ids:
            rank = ids.index(expected) + 1
            result = f"rank {rank}  similarity {hits[rank - 1].similarity:.4f}"
        else:
            misses += 1
            result = "missed           "
        print(f"{result}  top-1 {ids[0] if ids else '-'}  | {question!r} -> {expected}")

    print(f"\n{len(CASES) - misses}/{len(CASES)} expected chunks in the top {K}")
    sys.exit(1 if misses else 0)


if __name__ == "__main__":
    main()
