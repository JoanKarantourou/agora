"""Smoke test for the local LLM: stream one answer from Ollama and time it.

Model and server come from OLLAMA_MODEL / OLLAMA_URL (defaults: qwen2.5:7b, http://127.0.0.1:11434).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ollama import Client  # noqa: E402

from app.config import get_ollama_keep_alive, get_ollama_model, get_ollama_url  # noqa: E402

QUESTION = "In one sentence, who was Socrates?"


def main() -> None:
    model, url = get_ollama_model(), get_ollama_url()
    print(f"Model: {model} at {url}")
    print(f"Q: {QUESTION}\nA: ", end="", flush=True)

    start = time.perf_counter()
    first_token = None
    stream = Client(host=url).chat(
        model=model,
        messages=[{"role": "user", "content": QUESTION}],
        keep_alive=get_ollama_keep_alive(),
        stream=True,
    )
    for part in stream:
        text = part.message.content
        if text and first_token is None:
            first_token = time.perf_counter() - start
        print(text, end="", flush=True)
    total = time.perf_counter() - start

    print(f"\n\nTime to first token: {first_token:.2f}s" if first_token is not None else "\n\nNo tokens received")
    print(f"Total time:          {total:.2f}s")


if __name__ == "__main__":
    main()
