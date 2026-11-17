"""Strip Project Gutenberg boilerplate from data/raw/<id>.txt into data/clean/<id>.txt."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from download import RAW_DIR, ROOT, load_sources  # noqa: E402

CLEAN_DIR = ROOT / "data" / "clean"

# Older ebooks say "THIS PROJECT GUTENBERG EBOOK", newer ones "THE PROJECT GUTENBERG EBOOK".
START_MARKER = re.compile(r"^\*\*\* ?START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*$", re.MULTILINE | re.IGNORECASE)
END_MARKER = re.compile(r"^\*\*\* ?END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*$", re.MULTILINE | re.IGNORECASE)
END_TRAILER = re.compile(r"^End of (?:the )?Project Gutenberg.*$", re.MULTILINE | re.IGNORECASE)
PRODUCED_BY = re.compile(r"^Produced by .*?(?:\n[ \t]*\n|\Z)", re.MULTILINE | re.DOTALL)
INTRODUCTION = re.compile(r"^INTRODUCTION\.?[ \t]*$", re.MULTILINE)


def clean(text: str, work: str) -> str:
    text = text.removeprefix("﻿").replace("\r\n", "\n").replace("\r", "\n")

    start = START_MARKER.search(text)
    end = END_MARKER.search(text)
    if not start or not end:
        raise ValueError("Gutenberg START/END marker not found")
    text = text[start.end() : end.start()]

    trailer = END_TRAILER.search(text)
    if trailer:
        text = text[: trailer.start()]

    text = PRODUCED_BY.sub("", text, count=1)

    # Jowett's introduction runs from an unindented "INTRODUCTION." heading up to the
    # dialogue's own title heading (e.g. "APOLOGY"). Drop it along with the title page
    # and contents above it, so the text starts with the dialogue itself.
    intro = INTRODUCTION.search(text)
    if intro:
        heading = re.compile(rf"^{re.escape(work.upper())}\.?[ \t]*$", re.MULTILINE)
        dialogue = heading.search(text, intro.end())
        if not dialogue:
            raise ValueError(f"introduction found but no {work.upper()!r} heading after it")
        text = text[dialogue.end() :]

    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


def clean_file(source: dict[str, str], raw_dir: Path = RAW_DIR, clean_dir: Path = CLEAN_DIR) -> Path:
    raw = raw_dir / f"{source['id']}.txt"
    target = clean_dir / f"{source['id']}.txt"
    text = clean(raw.read_text(encoding="utf-8"), source["work"])
    clean_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    print(f"clean {source['id']} -> {target} ({len(text):,} chars)")
    return target


def main() -> None:
    for source in load_sources():
        clean_file(source)


if __name__ == "__main__":
    main()
