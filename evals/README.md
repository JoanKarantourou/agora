# Evals

`golden_set.jsonl` holds 30 retrieval test questions about Plato's *Apology*, one JSON object per line: `id`, `question`, `type`, `expected_chunk_ids` (the `chunks.chunk_id` values that answer it) and `answer_notes` (what a correct answer should say).
Types: `quote` (8) asks for a famous line worded loosely; `paraphrase` (10) states an idea in words the text doesn't use; `concept` (8) is a broader question whose answer may span several chunks.
`unanswerable` (4) asks about things the *Apology* doesn't cover, so `expected_chunk_ids` is `[]` and the right response is a refusal.
The labels refer to the chunk ids currently in the database; if the chunker changes and the ids shift, re-check them.
Run retrieval scoring with `uv run python evals/run_retrieval.py [--mode vector|keyword|hybrid|rerank]` (needs Postgres with embedded chunks); results go to `evals/results/retrieval_<mode>_<date>.json`, and `evals/compare_retrieval.py A.json B.json [C.json ...]` compares runs.
Run faithfulness scoring with `uv run python evals/run_faithfulness.py` (needs Postgres and Ollama; the local model judges its own answers); results go to `evals/results/faithfulness_<date>.json`.
