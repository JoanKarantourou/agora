from pathlib import Path

import pytest

from scripts.clean import RAW_DIR, clean_file

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = {"id": "gutenberg_sample", "work": "Sample"}
APOLOGY = {"id": "plato_apology", "work": "Apology"}


def test_clean_strips_gutenberg_boilerplate_from_sample(tmp_path: Path) -> None:
    text = clean_file(SAMPLE, raw_dir=FIXTURES, clean_dir=tmp_path).read_text(encoding="utf-8")

    assert text == (
        "First line of the dialogue, spoken by the first speaker.\n"
        "Second line of the dialogue, answered by the second.\n"
        "\n"
        "\n"
        "Third line of the dialogue, after a run of extra blank lines.\n"
        "Fourth line of the dialogue, with trailing spaces.\n"
        "Last line of the dialogue.\n"
    )


@pytest.mark.skipif(not (RAW_DIR / "plato_apology.txt").exists(), reason="run scripts/download.py first")
def test_cleaned_apology_has_no_gutenberg_text(tmp_path: Path) -> None:
    text = clean_file(APOLOGY, clean_dir=tmp_path).read_text(encoding="utf-8")

    assert "gutenberg" not in text.lower()
    assert text.startswith("How you, O Athenians, have been affected by my accusers")
