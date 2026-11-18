"""The local LLM, served by Ollama."""

from ollama import Client

from app.config import get_ollama_model, get_ollama_url

TEMPERATURE = 0.1  # low: stick to the passages rather than improvise


class OllamaLLM:
    def __init__(self, model: str | None = None, url: str | None = None) -> None:
        self.model = model or get_ollama_model()
        self.client = Client(host=url or get_ollama_url())

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat(
            model=self.model, messages=messages, options={"temperature": TEMPERATURE}
        )
        return response.message.content
