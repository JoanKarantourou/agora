import pytest

from app.hybrid_search import RRF_K, rrf_merge
from app.search import SearchHit


def hit(chunk_id: str, similarity: float = 0.5) -> SearchHit:
    return SearchHit(chunk_id=chunk_id, author="Plato", work="Apology", part="defence",
                     similarity=similarity, text=f"{chunk_id} text")


def ids(hits: list[SearchHit]) -> list[str]:
    return [h.chunk_id for h in hits]


def test_a_chunk_in_both_lists_outranks_chunks_in_only_one() -> None:
    vector = [hit("a"), hit("both")]  # "both" is only 2nd here...
    keyword = [hit("b"), hit("both")]  # ...and 2nd here, but appears in both lists

    merged = rrf_merge([vector, keyword], k=3)

    assert ids(merged) == ["both", "a", "b"]
    assert merged[0].similarity == pytest.approx(2 / (RRF_K + 2))
    assert merged[1].similarity == pytest.approx(1 / (RRF_K + 1))


def test_ties_break_by_best_rank_then_chunk_id_whatever_the_input_order() -> None:
    # "x" and "y" both score 1/(K+1) + 1/(K+3); "p" and "q" both score 1/(K+2).
    vector = [hit("y"), hit("q"), hit("x")]
    keyword = [hit("x"), hit("p"), hit("y")]

    merged = rrf_merge([vector, keyword], k=4)
    swapped = rrf_merge([keyword, vector], k=4)

    assert ids(merged) == ids(swapped) == ["x", "y", "p", "q"]


def test_merge_keeps_hit_fields_returns_top_k_and_stores_the_rrf_score() -> None:
    merged = rrf_merge([[hit("a", 0.9), hit("b", 0.8)], [hit("c", 12.0)]], k=2)

    assert ids(merged) == ["a", "c"]
    assert merged[0].text == "a text" and merged[0].work == "Apology"
    assert merged[0].similarity == pytest.approx(1 / (RRF_K + 1))


def test_merge_handles_empty_lists() -> None:
    assert rrf_merge([[], []], k=5) == []
    assert ids(rrf_merge([[hit("a")], []], k=5)) == ["a"]
