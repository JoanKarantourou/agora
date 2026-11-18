"""The prompt that keeps answers grounded in the retrieved passages.

Edit the rules here; build_messages() is the only thing the endpoint calls.
"""

from app.search import SearchHit

NOT_FOUND = "I can't find that in the texts I have."

SYSTEM_PROMPT = f"""You answer questions about philosophical texts using numbered passages.

Rules:
- Answer ONLY from the numbered passages provided. Never use outside knowledge.
- Cite every claim with the passage number, like [2].
- If the passages don't contain the answer, reply exactly: "{NOT_FOUND}"
- Be concise."""


def format_passage(n: int, hit: SearchHit) -> str:
    return f"[{n}] ({hit.work}, {hit.part}): {hit.text}"


def build_messages(question: str, passages: list[SearchHit]) -> list[dict[str, str]]:
    """Chat messages for the LLM: the rules, then the passages numbered from 1, then the question."""
    numbered = "\n\n".join(format_passage(n, hit) for n, hit in enumerate(passages, start=1))
    user = f"Passages:\n\n{numbered}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
