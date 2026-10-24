# Agora: plan

## What it is
A philosophy companion. Ask what a thinker said or thought, and Agora answers
from the original texts, citing the exact passage.

## Who it's for
Me first (local). Public launch later (Phase 10).

## The core promise
Agora only says what the sources support. Every answer cites a passage.
If the texts don't cover it, Agora says so instead of guessing.

## Starting corpus
Heraclitus (Burnet), Plato (Jowett: Apology, Phaedo, Symposium),
Aristotle (Nicomachean Ethics). English, public domain, Project Gutenberg.

## Out of scope for the core (Phases 0–7)
User accounts, languages other than English, fine-tuning,
voice, persona chat, deployment. See ideas.md.

## How we'll know it works
A 30-question test set (Phase 5) measuring:
- does search find the right passage?
- does the answer stick to its sources?

## Stack
Python 3.12, FastAPI, Postgres + pgvector, local LLM via Ollama,
local embedding model, Docker, GitHub Actions, OpenTelemetry later.