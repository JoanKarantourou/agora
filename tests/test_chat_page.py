from fastapi.testclient import TestClient

from app.main import app

# Not used as a context manager, so the startup hook (real model, Ollama) never runs.
client = TestClient(app)


def test_root_serves_the_chat_page() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<input id="question"' in response.text
    assert "/ask/stream" in response.text
