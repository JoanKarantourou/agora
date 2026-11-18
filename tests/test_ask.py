import json
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.rag.prompt import NOT_FOUND, build_messages
from app.search import SearchHit

HITS = [
    SearchHit(chunk_id="a-001", author="Plato", work="Apology", part="penalty",
              similarity=0.61, text="The unexamined life is not worth living."),
    SearchHit(chunk_id="a-002", author="Plato", work="Apology", part="final",
              similarity=0.55, text="Death is either a dreamless sleep or a journey."),
]


def test_prompt_numbers_and_labels_passages_and_states_the_rules() -> None:
    system, user = build_messages("Is death bad?", HITS)

    assert system["role"] == "system"
    for rule in [
        "Answer ONLY from the numbered passages provided. Never use outside knowledge.",
        "Cite every claim with the passage number, like [2].",
        f'If the passages don\'t contain the answer, reply exactly: "{NOT_FOUND}"',
        "Be concise.",
    ]:
        assert rule in system["content"]

    assert user["role"] == "user"
    assert user["content"] == (
        "Passages:\n\n"
        "[1] (Apology, penalty): The unexamined life is not worth living.\n\n"
        "[2] (Apology, final): Death is either a dreamless sleep or a journey.\n\n"
        "Question: Is death bad?"
    )


TOKENS = ["It is not", " worth\nliving", " [1]."]  # one holds a newline, to check SSE framing


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.reply

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self.calls.append(messages)
        for token in TOKENS:
            yield token


def parse_sse(body: str) -> list[tuple[str, object]]:
    """(event, JSON-decoded data) for each event in a text/event-stream body."""
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((fields["event"], json.loads(fields["data"])))
    return events


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """/ask with search and the LLM faked: no model, no database, no network."""
    searched: list[tuple[str, int]] = []

    def fake_search(question: str, k: int, embedder) -> list[SearchHit]:
        searched.append((question, k))
        return HITS

    monkeypatch.setattr(main, "search_chunks", fake_search)
    llm = FakeLLM("It is not worth living [1].")
    main.app.dependency_overrides[main.get_embedder] = lambda: None
    main.app.dependency_overrides[main.get_llm] = lambda: llm
    try:
        # Not used as a context manager, so the startup hook (real model, Ollama) never runs.
        yield TestClient(main.app), llm, searched
    finally:
        main.app.dependency_overrides.clear()


def test_ask_sends_the_grounded_prompt_and_returns_numbered_sources(client) -> None:
    http, llm, searched = client

    response = http.post("/ask", json={"question": "Is death bad?"})

    assert response.status_code == 200
    assert searched == [("Is death bad?", 5)]
    assert llm.calls == [build_messages("Is death bad?", HITS)]
    body = response.json()
    assert body["answer"] == "It is not worth living [1]."
    assert body["sources"] == [
        {"n": 1, "work": "Apology", "section": "penalty",
         "text": "The unexamined life is not worth living.", "score": 0.61},
        {"n": 2, "work": "Apology", "section": "final",
         "text": "Death is either a dreamless sleep or a journey.", "score": 0.55},
    ]
    assert set(body["timings"]) == {"search_ms", "llm_ms"}


def test_ask_stream_sends_sources_then_tokens_then_done(client) -> None:
    http, llm, searched = client

    response = http.post("/ask/stream", json={"question": "Is death bad?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert searched == [("Is death bad?", 5)]
    assert llm.calls == [build_messages("Is death bad?", HITS)]
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["sources", "token", "token", "token", "done"]
    assert events[0][1] == [
        {"n": 1, "work": "Apology", "section": "penalty",
         "text": "The unexamined life is not worth living.", "score": 0.61},
        {"n": 2, "work": "Apology", "section": "final",
         "text": "Death is either a dreamless sleep or a journey.", "score": 0.55},
    ]
    assert [data for _, data in events[1:4]] == TOKENS
    done = events[-1][1]
    assert set(done) == {"search_ms", "first_token_ms", "total_ms"}
    assert done["search_ms"] <= done["first_token_ms"] <= done["total_ms"]


def test_ask_stream_without_passages_says_not_found_and_skips_the_llm(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    http, llm, _ = client
    monkeypatch.setattr(main, "search_chunks", lambda question, k, embedder: [])

    events = parse_sse(http.post("/ask/stream", json={"question": "Capital of France?"}).text)

    assert [name for name, _ in events] == ["sources", "token", "done"]
    assert events[0][1] == []
    assert events[1][1] == NOT_FOUND
    assert llm.calls == []


def test_ask_rejects_an_empty_question(client) -> None:
    http, llm, _ = client

    assert http.post("/ask", json={"question": ""}).status_code == 422
    assert llm.calls == []
