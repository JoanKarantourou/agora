"""The embedding model: one instance per app, shared by every request."""

from typing import Any, Protocol

MODEL_NAME = "BAAI/bge-base-en-v1.5"
# bge wants this instruction in front of search questions, not in front of passages.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EncodesText(Protocol):
    def encode(self, sentences: list[str], *, normalize_embeddings: bool) -> Any: ...


def to_pgvector(vector) -> str:
    """pgvector's text format: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"


class Embedder:
    def __init__(self, model: EncodesText) -> None:
        self.model = model

    @classmethod
    def load(cls) -> "Embedder":
        # Imported here so code that never loads the model (tests, scripts) skips torch's import time.
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(SentenceTransformer(MODEL_NAME, device=device))

    def embed_query(self, question: str) -> list[float]:
        (vector,) = self.model.encode([QUERY_PREFIX + question], normalize_embeddings=True)
        return [float(x) for x in vector]
