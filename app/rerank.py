"""Reranking: pool the vector and keyword candidates, then let a cross-encoder re-sort them.

A cross-encoder reads the question and a passage together, so it judges relevance far better
than comparing two separately made embeddings, but it is too slow to run over every chunk.
Hence the pool: the union of the top POOL_PER_SEARCH from each search, deduplicated.
SearchHit.similarity holds the reranker's score.
"""

import time
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel

from app.embeddings import Embedder
from app.keyword_search import keyword_search
from app.search import SearchHit, search_chunks

RERANK_MODEL = "BAAI/bge-reranker-base"
POOL_PER_SEARCH = 20  # candidates taken from each of vector and keyword search
MAX_LENGTH = 512  # tokens per (question, passage) pair; the longest chunk is ~300 words


class ScoresPairs(Protocol):
    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> Sequence[float]: ...


class Reranker:
    def __init__(self, model: ScoresPairs) -> None:
        self.model = model

    @classmethod
    def load(cls) -> "Reranker":
        # Imported here so code that never loads the model skips torch's import time.
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(CrossEncoder(RERANK_MODEL, device=device, max_length=MAX_LENGTH))

    def score(self, question: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.predict([(question, p) for p in passages], show_progress_bar=False)
        return [float(s) for s in scores]


def build_pool(*ranked_lists: list[SearchHit]) -> list[SearchHit]:
    """Union of the lists, each chunk once, in first-seen order (first list first)."""
    pool: dict[str, SearchHit] = {}
    for ranked in ranked_lists:
        for hit in ranked:
            pool.setdefault(hit.chunk_id, hit)
    return list(pool.values())


def rerank(question: str, pool: list[SearchHit], reranker: Reranker, k: int) -> list[SearchHit]:
    """The k pool chunks the reranker scores highest; ties keep pool order."""
    scores = reranker.score(question, [h.text for h in pool])
    order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
    return [pool[i].model_copy(update={"similarity": scores[i]}) for i in order[:k]]


class RerankRun(BaseModel):
    hits: list[SearchHit]
    pool: list[SearchHit]
    rerank_ms: float  # the reranker call alone, not the two searches


def rerank_search_details(
    question: str, k: int, embedder: Embedder, reranker: Reranker
) -> RerankRun:
    pool = build_pool(
        search_chunks(question, POOL_PER_SEARCH, embedder),
        keyword_search(question, POOL_PER_SEARCH),
    )
    start = time.perf_counter()
    hits = rerank(question, pool, reranker, k)
    return RerankRun(hits=hits, pool=pool, rerank_ms=(time.perf_counter() - start) * 1000)


def rerank_search(question: str, k: int, embedder: Embedder, reranker: Reranker) -> list[SearchHit]:
    """The k best chunks after reranking the pooled vector and keyword candidates."""
    return rerank_search_details(question, k, embedder, reranker).hits
