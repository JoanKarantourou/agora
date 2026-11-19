import app.rerank as rerank_module
from app.rerank import Reranker, build_pool, rerank, rerank_search_details
from app.search import SearchHit


def hit(chunk_id: str, similarity: float = 0.5) -> SearchHit:
    return SearchHit(chunk_id=chunk_id, author="Plato", work="Apology", part="defence",
                     similarity=similarity, text=f"{chunk_id} text")


def ids(hits: list[SearchHit]) -> list[str]:
    return [h.chunk_id for h in hits]


class FakeModel:
    """Scores a passage by a fixed table keyed on its text; records what it was asked."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, **kwargs):
        self.calls.append(pairs)
        return [self.scores.get(text, 0.0) for _, text in pairs]


def test_pool_is_the_union_with_each_chunk_once_in_first_seen_order() -> None:
    vector = [hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)]
    keyword = [hit("c", 12.0), hit("d", 9.0), hit("a", 3.0)]

    pool = build_pool(vector, keyword)

    assert ids(pool) == ["a", "b", "c", "d"]
    assert pool[0].similarity == 0.9  # the first list's copy of a shared chunk is kept


def test_pool_handles_empty_and_identical_lists() -> None:
    assert build_pool([], []) == []
    assert ids(build_pool([hit("a"), hit("b")], [hit("a"), hit("b")])) == ["a", "b"]


def test_rerank_orders_by_reranker_score_keeps_ties_in_pool_order_and_stores_score() -> None:
    model = FakeModel({"a text": 0.1, "b text": 0.9, "c text": 0.5, "d text": 0.5})

    hits = rerank("q?", [hit("a"), hit("b"), hit("c"), hit("d")], Reranker(model), k=3)

    assert ids(hits) == ["b", "c", "d"]
    assert [h.similarity for h in hits] == [0.9, 0.5, 0.5]
    assert model.calls == [[("q?", "a text"), ("q?", "b text"), ("q?", "c text"), ("q?", "d text")]]


def test_rerank_search_scores_every_pooled_chunk_once(monkeypatch) -> None:
    monkeypatch.setattr(rerank_module, "search_chunks", lambda q, k, e: [hit("a"), hit("b")])
    monkeypatch.setattr(rerank_module, "keyword_search", lambda q, k: [hit("b"), hit("c")])
    model = FakeModel({"c text": 0.8, "a text": 0.6})

    run = rerank_search_details("q?", 2, embedder=None, reranker=Reranker(model))

    assert ids(run.pool) == ["a", "b", "c"]
    assert [text for _, text in model.calls[0]] == ["a text", "b text", "c text"]
    assert ids(run.hits) == ["c", "a"]
    assert run.rerank_ms >= 0


def test_reranker_skips_the_model_for_an_empty_pool() -> None:
    model = FakeModel({})

    assert rerank("q?", [], Reranker(model), k=5) == []
    assert model.calls == []
