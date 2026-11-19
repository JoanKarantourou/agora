from app.rag.citations import cited_numbers


def test_cited_numbers_reads_single_and_grouped_citations_once_each_in_order() -> None:
    answer = "Death may be a gain [3]. He obeys the god [1, 3] and says so again [2][1]. Not [x]."

    assert cited_numbers(answer) == [3, 1, 2]


def test_cited_numbers_is_empty_without_citations() -> None:
    assert cited_numbers("I can't find that in the texts I have.") == []
