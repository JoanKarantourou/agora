import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import Depends, FastAPI, Query, Request
from pydantic import BaseModel, Field

from app.embeddings import Embedder
from app.rag.llm import OllamaLLM
from app.rag.prompt import NOT_FOUND, build_messages
from app.search import SearchHit, search_chunks

ASK_PASSAGES = 5


class ChatsWithLLM(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str: ...


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


@app.post("/ask")
def ask(
    body: AskRequest,
    embedder: Embedder = Depends(get_embedder),
    llm: ChatsWithLLM = Depends(get_llm),
) -> AskResponse:
    start = time.perf_counter()
    hits = search_chunks(body.question, ASK_PASSAGES, embedder)
    search_ms = elapsed_ms(start)

    start = time.perf_counter()
    # With nothing to ground an answer in, skip the LLM rather than let it answer from memory.
    answer = llm.chat(build_messages(body.question, hits)) if hits else NOT_FOUND
    llm_ms = elapsed_ms(start) if hits else 0.0

    sources = [
        Source(n=n, work=h.work, section=h.part, text=h.text, score=h.similarity)
        for n, h in enumerate(hits, start=1)
    ]
    return AskResponse(
        answer=answer, sources=sources, timings=Timings(search_ms=search_ms, llm_ms=llm_ms)
    )
