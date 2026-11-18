"""The local LLM, served by Ollama."""

from collections.abc import AsyncIterator

from ollama import AsyncClient, Client

from app.config import get_ollama_model, get_ollama_url

TEMPERATURE = 0.1  # low: stick to the passages rather than improvise


class OllamaLLM:
    def __init__(self, model: str | None = None, url: str | None = None) -> None:
        self.model = model or get_ollama_model()
        host = url or get_ollama_url()
        self.client = Client(host=host)
        # Used by streaming endpoints, so a long answer doesn't hold up other requests.
        self.async_client = AsyncClient(host=host)

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat(
            model=self.model, messages=messages, options={"temperature": TEMPERATURE}
        )
        return response.message.content

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """The answer's text pieces as Ollama produces them."""
        parts = await self.async_client.chat(
            model=self.model, messages=messages, options={"temperature": TEMPERATURE}, stream=True
        )
        async for part in parts:
            if part.message.content:
                yield part.message.content
