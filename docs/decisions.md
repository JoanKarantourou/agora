# Decision log

## D1: One job: answer from the texts, with citations (2026-10-24)
Why: a narrow job is buildable, testable and explainable.
Rejected: general philosophy chatbot (can't measure truthfulness).

## D2: Grounding rule, two tiers (2026-10-24)
Why: trust is the product. Books are primary and always cited;
commentary comes later, clearly labelled, from named sources.
Rejected: answering from model memory; anonymous forum content.

## D3: Small core scope (2026-10-24)
Why: accounts, voice, personas and deployment teach little about RAG.
Parked in ideas.md with a roadmap home each.

## D4: Three thinkers to start (2026-10-24)
Why: Heraclitus for short quotes, Plato for long dialogues,
Aristotle to set up the Aquinas story. Small corpus = fast experiments.
Rejected: loading everything at once (more data hides bugs).

## D5: Python + FastAPI (2026-10-24)
Why: the AI ecosystem lives in Python; it's what the market asks for.
Rejected: .NET (comfort zone, but fights the AI tooling).

## D6: Postgres + pgvector for all storage (2026-10-24)
Why: one database for text, labels, vectors and later keyword search.
Rejected: dedicated vector DB like Qdrant (extra moving part).

## D7: Local, free models (2026-10-24)
Why: zero cost; RTX 3070 (8 GB VRAM) fits a 7–8B quantized model via Ollama.
Model sits behind one interface so Claude can be swapped in for comparison.
Trade-off accepted: small models ground less reliably, so we measure it in Phase 5.
Rejected: Claude API for answers (costs money per use).

## D8: Hosting decided in Phase 10; Docker from day one (2026-10-24)
Why: containers make hosting a late, cheap choice (self-host or paid).