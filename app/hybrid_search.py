"""Hybrid search: vector and keyword results merged with Reciprocal Rank Fusion (RRF).

Each chunk scores sum(1 / (RRF_K + rank)) over the lists it appears in, with ranks starting at 1.
Only ranks are used, so the cosine similarities and BM25 scores never need to be compared.
SearchHit.similarity holds the RRF score.
"""

from app.embeddings import Embedder
from app.keyword_search import keyword_search
from app.search import SearchHit, search_chunks

RRF_K = 60  # the usual RRF constant: damps the gap between rank 1 and the ranks below it
CANDIDATES = 20  # how many results to take from each search before merging


def rrf_merge(ranked_lists: list[list[SearchHit]], k: int) -> list[SearchHit]:
    """The k chunks with the highest RRF score across the lists, best first.

    Ties are broken by the chunk's best rank in any list, then by chunk_id, so the order
    never depends on dict or set ordering.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    hits: dict[str, SearchHit] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1 / (RRF_K + rank)
            best_rank[hit.chunk_id] = min(rank, best_rank.get(hit.chunk_id, rank))
            hits.setdefault(hit.chunk_id, hit)
    order = sorted(scores, key=lambda cid: (-scores[cid], best_rank[cid], cid))
    return [hits[cid].model_copy(update={"similarity": scores[cid]}) for cid in order[:k]]


def hybrid_search(question: str, k: int, embedder: Embedder) -> list[SearchHit]:
    """The k best chunks after fusing the top CANDIDATES from vector and keyword search."""
    vector = search_chunks(question, CANDIDATES, embedder)
    keyword = keyword_search(question, CANDIDATES)
    return rrf_merge([vector, keyword], k)
