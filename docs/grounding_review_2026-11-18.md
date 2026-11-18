# Grounding review, 2026-11-18

Written by hand; not generated. It reviews the run saved in [grounding_check_2026-11-18.md](grounding_check_2026-11-18.md), produced by `scripts/grounding_check.py`.

The automatic checks only test the *form* of an answer: a refusal is the exact text, and every
`[n]` is a real source number. They do not test whether the cited passage supports the claim.
Checking the cited chunks by hand:

| # | Format check | Supported by the cited passage? |
|---|---|---|
| 1 | PASS | **Yes.** [3] is `plato_apology-059`, the "sleep undisturbed even by dreams" / journey passage. |
| 2 | PASS | **No, wrong number.** The quote is real but comes from [1] (`plato_apology-054`, the "unexamined life" chunk). The answer cites [2] (`plato_apology-033`), which does not contain it. |
| 3 | PASS | **No.** The oracle passage (`plato_apology-010`, Chaerephon asks the oracle) is not among the 5 retrieved. "The oracle stated that Socrates was the wisest man" is outside knowledge; the cited [2] (`plato_apology-009`) never mentions the oracle. |
| 4–7 | PASS | Yes: exact refusal, including the prompt-injection trap (7). |

So: refusals 4/4 hold up; grounded, correctly cited answers are **1/3**, not 3/3.
Q3 is the same retrieval miss as the Phase 3 search smoke check (the oracle chunk ranks below
the top 5 for questions that name Socrates).

Follow-up, planned for Phase 5: an automatic check that quoted text appears in a cited source,
which would have caught Q2 and Q3.
