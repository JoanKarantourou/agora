import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from fastapi import Depends, FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field

from app.embeddings import Embedder
from app.rag.llm import OllamaLLM
from app.rag.prompt import NOT_FOUND, build_messages
from app.search import SearchHit, search_chunks

ASK_PASSAGES = 5
CHAT_PAGE = Path(__file__).resolve().parent / "static" / "index.html"


class ChatsWithLLM(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str: ...

    def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]: ...


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Loaded once at startup; every request reuses them.
    app.state.embedder = Embedder.load()
    app.state.llm = OllamaLLM()
    yield


app = FastAPI(title="Agora", lifespan=lifespan)


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_llm(request: Request) -> ChatsWithLLM:
    return request.app.state.llm


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class Source(BaseModel):
    n: int
    work: str
    section: str
    text: str
    score: float


class Timings(BaseModel):
    search_ms: float
    llm_ms: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    timings: Timings


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def to_sources(hits: list[SearchHit]) -> list[Source]:
    return [
        Source(n=n, work=h.work, section=h.part, text=h.text, score=h.similarity)
        for n, h in enumerate(hits, start=1)
    ]


@app.get("/", include_in_schema=False)
def chat_page() -> FileResponse:
    return FileResponse(CHAT_PAGE, media_type="text/html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/search")
def search(
    q: str = Query(min_length=1),
    k: int = Query(5, ge=1, le=50),
    embedder: Embedder = Depends(get_embedder),
) -> list[SearchHit]:
    return search_chunks(q, k, embedder)


class Answered(BaseModel):
    """What the /ask pipeline produced, including the retrieved chunks (the API shows less)."""

    hits: list[SearchHit]
    answer: str
    timings: Timings


def answer_question(question: str, embedder: Embedder, llm: ChatsWithLLM) -> Answered:
    """The /ask pipeline: retrieve passages, then answer from them. Also used by the evals."""
    start = time.perf_counter()
    hits = search_chunks(question, ASK_PASSAGES, embedder)
    search_ms = elapsed_ms(start)

    start = time.perf_counter()
    # With nothing to ground an answer in, skip the LLM rather than let it answer from memory.
    answer = llm.chat(build_messages(question, hits)) if hits else NOT_FOUND
    llm_ms = elapsed_ms(start) if hits else 0.0

    return Answered(hits=hits, answer=answer, timings=Timings(search_ms=search_ms, llm_ms=llm_ms))


@app.post("/ask")
def ask(
    body: AskRequest,
    embedder: Embedder = Depends(get_embedder),
    llm: ChatsWithLLM = Depends(get_llm),
) -> AskResponse:
    result = answer_question(body.question, embedder, llm)
    return AskResponse(answer=result.answer, sources=to_sources(result.hits), timings=result.timings)


@app.post("/ask/stream", response_class=EventSourceResponse)
async def ask_stream(
    body: AskRequest,
    embedder: Embedder = Depends(get_embedder),
    llm: ChatsWithLLM = Depends(get_llm),
) -> AsyncIterator[ServerSentEvent]:
    """Events, in order: "sources" (list), "token" (one per text piece), "done" (timings).

    Every data field is JSON, so a token is a quoted string and may contain newlines.
    """
    start = time.perf_counter()
    # Search is blocking (embedding + psycopg), so it runs in a worker thread.
    hits = await run_in_threadpool(search_chunks, body.question, ASK_PASSAGES, embedder)
    search_ms = elapsed_ms(start)
    yield ServerSentEvent(event="sources", data=[s.model_dump() for s in to_sources(hits)])

    first_token_ms = None
    if hits:
        async for token in llm.stream(build_messages(body.question, hits)):
            if first_token_ms is None:
                first_token_ms = elapsed_ms(start)
            yield ServerSentEvent(event="token", data=token)
    else:
        # With nothing to ground an answer in, skip the LLM rather than let it answer from memory.
        first_token_ms = elapsed_ms(start)
        yield ServerSentEvent(event="token", data=NOT_FOUND)

    yield ServerSentEvent(
        event="done",
        data={"search_ms": search_ms, "first_token_ms": first_token_ms, "total_ms": elapsed_ms(start)},
    )
