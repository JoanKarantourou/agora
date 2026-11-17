# Agora

A philosophy companion that answers from original texts and cites its sources.

See [docs/plan.md](docs/plan.md) for the project plan.

## Ingesting books

Sources are listed in [data/sources.yaml](data/sources.yaml). To fetch, clean, chunk and store them:

```sh
cp .env.example .env            # then set real credentials
docker compose up -d --wait     # Postgres + pgvector on 127.0.0.1:5432
uv run alembic upgrade head     # create the schema
uv run python scripts/ingest.py
```

`ingest.py` runs four steps in order and stops at the first failure:

1. **download**: saves each text to `data/raw/<id>.txt` (skipped if already there)
2. **clean**: strips the Gutenberg header, licence and translator's introduction into `data/clean/<id>.txt`
3. **chunk**: splits the text into passages of up to 300 words in `data/chunks/<id>.jsonl`
4. **load**: replaces that source's rows in the `chunks` table

Re-running it is safe: each source's rows are replaced, not duplicated. Each step can also be run on its own, e.g. `uv run python scripts/chunk.py`.
