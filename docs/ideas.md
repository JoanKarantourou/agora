# Agora: ideas parking lot

Ideas we want, but not yet. Each has a home on the roadmap.
Rule: nothing here gets built before the core (Phases 0–7) works and is measured.

## Trust and sources
- Two-tier answers: primary sources (the books) always cited; commentary clearly labelled as such.
- Commentary from named academic sources only. No anonymous forum posts.
- Web search fallback when the books aren't enough → Phase 8 (agent), still cited and labelled.

## Experience
- Dark academia / sacred-text design (solemn, archival feel) → Phase 4. Styling only, so it can start with the first chat page.
- Philosopher profiles: short bio, personality notes, public-domain portrait or avatar → Phase 9, next to the influence map.
- Talk directly to a philosopher (persona chat, grounded in their own texts) → Phase 11.
- Voice conversations (speech in, speech out) → new bonus phase after 11. Prefer a local text-to-speech model to stay free; ElevenLabs only if paid is OK.

## Users
- User accounts → after Phase 10.
- Per-user memory and preferences (e.g. how Agora explains things) → after accounts.

## Developer side
- Developer page for logs, traces and health checks → Phase 7 (observability).

## More thinkers (after the core works)
- Stoics: Marcus Aurelius, Epictetus, Seneca.
- Augustine (Confessions) and Aquinas (Summa excerpts) → power the agent and influence map.
- Eastern philosophy: Confucius, Tao Te Ching (public-domain translations, e.g. Legge).

## Hosting → Phase 10
- Docker everything, so hosting stays a late, cheap decision.
- Option A: self-host on my PC via a tunnel (e.g. Cloudflare Tunnel). Free, uses my RTX 3070. Offline when the PC is off.
- Option B: paid host for the app plus a managed Postgres, with the answering model swapped to a hosted API.

## Models
- Local LLM via Ollama (7–8B quantized, fits 8 GB VRAM). Exact model picked in Phase 4.
- Keep the model behind one interface, so swapping to Claude for comparison is a config change.