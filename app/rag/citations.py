"""Reading the [n] passage citations out of an answer."""

import re

# [2] and [2, 3] are both citations, as the chat page links them.
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited_numbers(answer: str) -> list[int]:
    """Every cited passage number, in order of first appearance."""
    seen: list[int] = []
    for match in CITATION.finditer(answer):
        for n in match.group(1).split(","):
            if int(n) not in seen:
                seen.append(int(n))
    return seen
