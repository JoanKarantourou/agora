from fastapi import FastAPI

app = FastAPI(title="Agora")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
