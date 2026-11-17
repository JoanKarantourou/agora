"""Split data/clean/<id>.txt into retrieval chunks at data/chunks/<id>.jsonl."""

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clean import CLEAN_DIR  # noqa: E402
from download import ROOT, load_sources  # noqa: E402

CHUNKS_DIR = ROOT / "data" / "chunks"
SHORT_WORDS = 40
MAX_WORDS = 300

PART_BREAK = re.compile(r"\n[ \t]*\n[ \t]*\n")
PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
# A sentence ends at . ? or ! (plus any closing quotes or brackets) followed by whitespace.
SENTENCE_END = re.compile(r"(?<=[.?!])[”’\"')\]]*\s+")
QUESTION_END = re.compile(r"\?[”’\"')\]]*$")


def word_count(text: str) -> int:
    return len(text.split())


def split_parts(text: str, labels: list[str]) -> list[tuple[str, list[str]]]:
    """Return (label, paragraphs) for each part, each paragraph unwrapped onto one line."""
    parts = [p for p in PART_BREAK.split(text.strip()) if p.strip()]
    if len(parts) != len(labels):
        raise ValueError(f"found {len(parts)} parts but the manifest labels {len(labels)}: {labels}")
    return [
        (label, [" ".join(p.split()) for p in PARAGRAPH_BREAK.split(part) if p.strip()])
        for label, part in zip(labels, parts)
    ]


def split_sentences(paragraph: str) -> list[str]:
    sentences, start = [], 0
    for match in SENTENCE_END.finditer(paragraph):
        sentences.append(paragraph[start : match.start() + len(match.group().rstrip())])
        start = match.end()
    if start < len(paragraph):
        sentences.append(paragraph[start:])
    return sentences


def split_long(paragraph: str) -> list[str]:
    """Cut a long paragraph at sentence ends into pieces of roughly equal size, each <= MAX_WORDS
    unless a single sentence is longer than that."""
    sentences = split_sentences(paragraph)
    pieces_left = math.ceil(word_count(paragraph) / MAX_WORDS)
    remaining = word_count(paragraph)
    pieces: list[list[str]] = [[]]
    for sentence in sentences:
        current, size = word_count(" ".join(pieces[-1])), word_count(sentence)
        target = remaining / pieces_left
        # Cut before this sentence if that lands closer to the target size than cutting after it.
        if pieces[-1] and pieces_left > 1 and (
            current + size > MAX_WORDS or abs(current - target) <= abs(current + size - target)
        ):
            remaining -= current
            pieces_left -= 1
            pieces.append([])
        pieces[-1].append(sentence)
    return [" ".join(piece) for piece in pieces]


def chunk_part(paragraphs: list[str], first_para: int) -> list[tuple[int, int, str]]:
    """Return (para_start, para_end, text) chunks for one part. Paragraph numbers are global."""
    # Each chunk is [para_start, para_end, [paragraph texts], kind], where kind is "short" (a group
    # of short turns), "para" (one whole paragraph of SHORT_WORDS..MAX_WORDS) or "piece" (part of a
    # split long paragraph).
    chunks: list[list] = []
    for number, paragraph in enumerate(paragraphs, first_para):
        words = word_count(paragraph)
        if words < SHORT_WORDS:
            last = chunks[-1] if chunks else None
            fits = last and word_count(" ".join(last[2])) + words <= MAX_WORDS
            # A whole paragraph followed by a short reply is the question that reply answers, so the
            # exchange starts there. A short turn that is itself a question opens a new exchange
            # instead (e.g. "Come hither, Meletus…" after the end of a speech).
            answers_para = last and last[3] == "para" and (len(last[2]) > 1 or not QUESTION_END.search(paragraph))
            if fits and (last[3] == "short" or answers_para):
                last[1], last[2] = number, last[2] + [paragraph]
            else:
                chunks.append([number, number, [paragraph], "short"])
        elif words > MAX_WORDS:
            chunks.extend([number, number, [piece], "piece"] for piece in split_long(paragraph))
        else:
            chunks.append([number, number, [paragraph], "para"])

    # A group of short turns that is still under SHORT_WORDS joins a neighbour, as long as the
    # result fits. A group that opens with a question joins the chunk after it (the rest of the
    # exchange); any other group joins the chunk before it (usually the question it answers).
    merged: list[list] = []
    for i, chunk in enumerate(chunks):
        size = word_count(" ".join(chunk[2]))
        if chunk[3] == "short" and size < SHORT_WORDS:
            prev = merged[-1] if merged else None
            nxt = chunks[i + 1] if i + 1 < len(chunks) else None
            fits_prev = prev and word_count(" ".join(prev[2])) + size <= MAX_WORDS
            fits_next = nxt and word_count(" ".join(nxt[2])) + size <= MAX_WORDS
            if QUESTION_END.search(chunk[2][0]) and fits_next or not fits_prev and fits_next:
                nxt[0], nxt[2] = chunk[0], chunk[2] + nxt[2]
                continue
            if fits_prev:
                prev[1], prev[2] = chunk[1], prev[2] + chunk[2]
                continue
        merged.append(chunk)
    return [(start, end, "\n\n".join(texts)) for start, end, texts, _ in merged]


def chunk_text(text: str, source: dict) -> list[dict]:
    records, first_para = [], 1
    for part, paragraphs in split_parts(text, source["parts"]):
        for start, end, chunk in chunk_part(paragraphs, first_para):
            records.append(
                {
                    "chunk_id": f"{source['id']}-{len(records) + 1:03d}",
                    "source_id": source["id"],
                    "author": source["author"],
                    "work": source["work"],
                    "translator": source["translator"],
                    "part": part,
                    "para_start": start,
                    "para_end": end,
                    "word_count": word_count(chunk),
                    "text": chunk,
                }
            )
        first_para += len(paragraphs)
    return records


def chunk_file(source: dict, clean_dir: Path = CLEAN_DIR, chunks_dir: Path = CHUNKS_DIR) -> Path:
    text = (clean_dir / f"{source['id']}.txt").read_text(encoding="utf-8")
    records = chunk_text(text, source)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    target = chunks_dir / f"{source['id']}.jsonl"
    with target.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"chunk {source['id']} -> {target} ({len(records)} chunks)")
    return target


def main() -> None:
    for source in load_sources():
        chunk_file(source)


if __name__ == "__main__":
    main()
