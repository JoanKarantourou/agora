import re
from pathlib import Path

from scripts.chunk import chunk_text, split_parts
from scripts.clean import clean

FIXTURE = Path(__file__).parent / "fixtures" / "gutenberg_parts_sample.txt"
SOURCE = {
    "id": "parts_sample",
    "author": "Nobody",
    "work": "Parts",
    "translator": "A. Fictional",
    "parts": ["defence", "penalty", "final"],
}
SENTENCE_END = re.compile(r"[.?!][”’\"')\]]*$")


def cleaned() -> str:
    return clean(FIXTURE.read_text(encoding="utf-8"), SOURCE["work"])


def chunks() -> list[dict]:
    return chunk_text(cleaned(), SOURCE)


def chunk_containing(text: str) -> dict:
    (match,) = [c for c in chunks() if text in c["text"]]
    return match


def test_clean_preserves_double_blank_lines_between_parts() -> None:
    text = cleaned()

    assert text.count("\n\n\n") == 2
    assert "\n\n\n\n" not in text


def test_short_turns_are_merged_into_one_chunk() -> None:
    exchange = chunk_containing("The laws.")

    assert (exchange["para_start"], exchange["para_end"]) == (2, 11)
    assert exchange["text"].endswith("And the senators, do they improve the young as well?\n\nYes, the senators improve them.")


def test_long_question_and_its_short_answers_share_a_chunk() -> None:
    question = chunk_containing("Then tell the judges who improves them.")

    answers = question["text"].split("\n\n")[3:]

    assert len(question["text"].split("\n\n")[2].split()) >= 40
    # The answers alone are 40+ words, so they would otherwise stand as a chunk of their own.
    assert sum(len(a.split()) for a in answers) >= 40
    assert answers[0] == "The laws."
    assert question["text"].endswith("\n\nYes, the senators improve them.")


def test_short_opening_question_starts_the_exchange_not_the_speech_before() -> None:
    speech = chunk_containing("The defence opens with a medium paragraph.")
    exchange = chunk_containing("Come here, Accuser")

    assert (speech["para_start"], speech["para_end"]) == (1, 1)
    assert exchange["para_start"] == 2
    assert exchange["text"].startswith("Come here, Accuser, and answer a question. You care about the young?\n\nYes, I do.\n\n")


def test_long_paragraph_is_split_at_sentence_ends() -> None:
    long_para = split_parts(cleaned(), SOURCE["parts"])[0][1][12]
    pieces = [c for c in chunks() if c["para_start"] == c["para_end"] == 13]

    assert len(long_para.split()) > 300
    assert len(pieces) >= 2
    assert all(150 <= p["word_count"] <= 300 for p in pieces)
    assert all(SENTENCE_END.search(p["text"]) for p in pieces)
    assert " ".join(p["text"] for p in pieces) == long_para


def test_no_chunk_crosses_a_part() -> None:
    records = chunks()
    paras_by_part = {"defence": range(1, 14), "penalty": range(14, 16), "final": range(16, 18)}

    assert [r["part"] for r in records] == sorted(
        (r["part"] for r in records), key=list(paras_by_part).index
    )
    for r in records:
        assert r["para_start"] in paras_by_part[r["part"]]
        assert r["para_end"] in paras_by_part[r["part"]]
    # The short line opening the penalty speech joins the paragraph after it, not the defence.
    penalty_first = next(r for r in records if r["part"] == "penalty")
    assert penalty_first["text"].startswith("So he proposes death.\n\n")
    # The short closing line joins the paragraph before it.
    assert records[-1]["text"].endswith("\n\nThe hour of departure has arrived.")


def test_records_carry_source_metadata() -> None:
    records = chunks()

    assert [r["chunk_id"] for r in records] == [f"parts_sample-{i:03d}" for i in range(1, len(records) + 1)]
    for r in records:
        assert (r["source_id"], r["author"], r["work"], r["translator"]) == (
            "parts_sample", "Nobody", "Parts", "A. Fictional",
        )
        assert r["word_count"] == len(r["text"].split())
