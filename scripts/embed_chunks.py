"""Embed every chunk that has no embedding yet and store the vector in chunks.embedding.

Only rows where embedding IS NULL are read, so re-running embeds nothing new. Chunks are passages,
so they are embedded as-is: bge's query instruction prefix is only for search questions.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from load import connect  # noqa: E402

MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 32
UPDATE = "UPDATE chunks SET embedding = %s::vector WHERE chunk_id = %s"


def to_pgvector(vector) -> str:
    """pgvector's text format: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"


def main() -> None:
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available; this script is meant to run on the GPU")

    start = time.perf_counter()

    with connect() as conn:
        rows = conn.execute(
            "SELECT chunk_id, text FROM chunks WHERE embedding IS NULL ORDER BY chunk_id"
        ).fetchall()
        if not rows:
            print("Chunks embedded: 0 (nothing to do)")
            return

        load_start = time.perf_counter()
        model = SentenceTransformer(MODEL_NAME, device="cuda")
        load_time = time.perf_counter() - load_start

        embed_start = time.perf_counter()
        vectors = model.encode(
            [text for _, text in rows],
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embed_time = time.perf_counter() - embed_start

        with conn.transaction(), conn.cursor() as cur:
            cur.executemany(
                UPDATE, [(to_pgvector(v), chunk_id) for (chunk_id, _), v in zip(rows, vectors)]
            )

    total_time = time.perf_counter() - start
    print(f"Chunks embedded: {len(rows)} on {torch.cuda.get_device_name(0)}")
    print(f"Model load time: {load_time:.2f}s")
    print(f"Embedding time:  {embed_time:.2f}s")
    print(f"Total time:      {total_time:.2f}s")


if __name__ == "__main__":
    main()
