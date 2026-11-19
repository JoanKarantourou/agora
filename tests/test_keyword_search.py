from app.keyword_search import KeywordIndex, tokenize

ROWS = [
    ("k-001", "Plato", "Apology", "defence", "I am that gadfly which God has attached to the state."),
    ("k-002", "Plato", "Apology", "penalty", "The unexamined life is not worth living."),
    ("k-003", "Plato", "Apology", "final", "Death is either a dreamless sleep or a journey."),
]


def test_tokenize_lowercases_strips_punctuation_and_stopwords() -> None:
    assert tokenize("The Unexamined life—isn't it worth living?") == ["unexamined", "life", "worth", "living"]
    assert tokenize("Socrates’ 'gadfly'!") == ["socrates", "gadfly"]


def test_search_ranks_by_keywords_and_skips_chunks_with_none() -> None:
    hits = KeywordIndex(ROWS).search("Which passage mentions the gadfly?", k=5)

    assert [h.chunk_id for h in hits] == ["k-001"]
    assert (hits[0].work, hits[0].part, hits[0].text) == ("Apology", "defence", ROWS[0][4])
    assert hits[0].similarity > 0


def test_search_returns_at_most_k_best_first() -> None:
    hits = KeywordIndex(ROWS).search("life death sleep living", k=1)

    assert len(hits) == 1
    assert hits[0].chunk_id in {"k-002", "k-003"}
