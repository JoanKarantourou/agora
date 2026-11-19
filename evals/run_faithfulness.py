"""Score how faithfully /ask answers stick to their retrieved passages.

Runs every golden-set question through the same pipeline as POST /ask (search + LLM), then:
- answerable questions: a local judge (Ollama, temperature 0) splits the answer into claims and
  marks each supported or not by the retrieved passages; faithfulness = supported / claims;
- unanswerable questions: the judge decides whether the answer clearly refused;
- every answer: a citation check (no LLM) that each cited [n] is one of the retrieved passages;
- every claim that carries a citation: the judge, shown only that cited passage, decides whether it
  supports the claim; citation accuracy = correct citations / citations.
Prints a summary, saves full results to evals/results/faithfulness_<date>.json, and writes
evals/results/judge_review_<date>.md with 5 random citation verdicts for a person to check.

Caveat: the judge is the same model that wrote the answers, so it may go easy on its own mistakes.
"""

import json
import random
import sys
from datetime import date, datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ollama import Client  # noqa: E402

from app.config import get_ollama_keep_alive, get_ollama_model, get_ollama_url  # noqa: E402
from app.embeddings import Embedder  # noqa: E402
from app.main import answer_question  # noqa: E402
from app.rag.citations import CITATION, cited_numbers  # noqa: E402
from app.rag.llm import OllamaLLM  # noqa: E402
from app.rag.prompt import NOT_FOUND, format_passage  # noqa: E402
from app.search import SearchHit  # noqa: E402

GOLDEN_SET = ROOT / "evals" / "golden_set.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"
TYPES = ["quote", "paraphrase", "concept"]
WORST = 5
REVIEW_SAMPLE = 5

CLAIMS_JUDGE = """You check whether an ANSWER is supported by numbered source PASSAGES.

1. Split the ANSWER into its separate factual claims. A claim is one statement about what someone
   said, thought or did, or about what something is. Ignore citation markers like [2] and filler
   such as "According to the passages".
2. For each claim, set "supported" to true only if the PASSAGES state it or it follows directly
   from them. If it needs outside knowledge, goes beyond the passages, or contradicts them, set
   "supported" to false.
3. For every claim, give a short "reason" that names the passage number that supports it, or says
   what is missing.
4. For every claim, list in "citations" the [n] numbers the ANSWER attaches to it: markers in the
   same sentence as the claim. A single marker at the very start or end of a one-sentence answer
   applies to that sentence. Use [] if the claim has no marker.

Judge only against the PASSAGES, never your own knowledge. Reply with JSON only."""

CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "reason": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["claim", "supported", "reason", "citations"],
            },
        }
    },
    "required": ["claims"],
}

CITATION_JUDGE = """You check whether one source PASSAGE supports one CLAIM.

Set "supported" to true only if this PASSAGE states the claim or it follows directly from it.
If the claim needs anything that is not in this passage, set "supported" to false. Give a short
"reason". Judge only against this PASSAGE, never your own knowledge. Reply with JSON only."""

CITATION_SCHEMA = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["supported", "reason"],
}

REFUSAL_JUDGE = """You check whether an ANSWER refuses a QUESTION.

Set "refused" to true if the answer clearly tells the user that the available texts do not cover
the question, in any wording. Set it to false if the answer attempts to answer the question in
any way, even partly or with caveats. Give a short "reason". Reply with JSON only."""

REFUSAL_SCHEMA = {
    "type": "object",
    "properties": {"refused": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["refused", "reason"],
}


class Judge:
    def __init__(self) -> None:
        self.model = get_ollama_model()
        self.client = Client(host=get_ollama_url())
        self.keep_alive = get_ollama_keep_alive()

    def ask(self, system: str, user: str, schema: dict) -> dict:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            format=schema,
            options={"temperature": 0},
            keep_alive=self.keep_alive,
        )
        return json.loads(response.message.content)

    def claims(self, question: str, answer: str, hits: list[SearchHit]) -> list[dict]:
        passages = "\n\n".join(format_passage(n, h) for n, h in enumerate(hits, start=1))
        user = f"PASSAGES:\n\n{passages}\n\nQUESTION: {question}\n\nANSWER:\n{answer}"
        return self.ask(CLAIMS_JUDGE, user, CLAIMS_SCHEMA)["claims"]

    def citation(self, claim: str, n: int, hit: SearchHit) -> dict:
        user = f"PASSAGE:\n{format_passage(n, hit)}\n\nCLAIM: {claim}"
        return self.ask(CITATION_JUDGE, user, CITATION_SCHEMA)

    def refusal(self, question: str, answer: str) -> dict:
        return self.ask(REFUSAL_JUDGE, f"QUESTION: {question}\n\nANSWER:\n{answer}", REFUSAL_SCHEMA)


def is_refusal(answer: str) -> bool:
    """The not-found reply, even with stray citation markers around it (e.g. "[1] I can't find...")."""
    return " ".join(CITATION.sub(" ", answer).split()) == NOT_FOUND


def is_empty(answer: str) -> bool:
    """Nothing but citation markers, e.g. an answer of just "[1]"."""
    return not CITATION.sub(" ", answer).strip()


def check_citations(answer: str, hits: list[SearchHit]) -> dict:
    """Map each cited [n] to the n-th retrieved chunk; numbers with no such passage are invalid."""
    cited = cited_numbers(answer)
    valid = [n for n in cited if 1 <= n <= len(hits)]
    return {
        "cited_numbers": cited,
        "cited_chunk_ids": [hits[n - 1].chunk_id for n in valid],
        "invalid_citations": [n for n in cited if n not in valid],
        "citations_valid": None if not cited else len(valid) == len(cited),
    }


def check_claim_citations(
    claims: list[dict], answer: str, hits: list[SearchHit], judge: Judge
) -> list[dict]:
    """Judge each (claim, cited passage) pair against that passage alone."""
    in_answer = set(cited_numbers(answer))
    checks = []
    for i, claim in enumerate(claims):
        # Keep only numbers that really appear in the answer: the judge may invent citations.
        claim["citations"] = [n for n in dict.fromkeys(claim.get("citations", [])) if n in in_answer]
        for n in claim["citations"]:
            check = {"claim_index": i, "claim": claim["claim"], "n": n}
            if 1 <= n <= len(hits):
                verdict = judge.citation(claim["claim"], n, hits[n - 1])
                check.update(chunk_id=hits[n - 1].chunk_id, correct=verdict["supported"],
                             reason=verdict["reason"])
            else:
                check.update(chunk_id=None, correct=False, reason=f"there is no passage [{n}]")
            checks.append(check)
    return checks


def evaluate(entry: dict, embedder: Embedder, llm: OllamaLLM, judge: Judge) -> dict:
    result = answer_question(entry["question"], embedder, llm)
    record = {
        "id": entry["id"],
        "type": entry["type"],
        "question": entry["question"],
        "expected_chunk_ids": entry["expected_chunk_ids"],
        "retrieved_chunk_ids": [h.chunk_id for h in result.hits],
        "answer": result.answer,
        "exact_not_found": result.answer.strip() == NOT_FOUND,
        "is_refusal": is_refusal(result.answer),
        "is_empty": is_empty(result.answer),
        **check_citations(result.answer, result.hits),
        "timings": result.timings.model_dump(),
    }
    if entry["type"] == "unanswerable":
        verdict = judge.refusal(entry["question"], result.answer)
        record.update(refused=verdict["refused"], refusal_reason=verdict["reason"])
    elif record["is_refusal"] or record["is_empty"]:
        # A refusal or an empty answer makes no claims: nothing to score, so no faithfulness score.
        record.update(claims=[], faithfulness=None, citation_checks=[], citation_accuracy=None)
    else:
        claims = judge.claims(entry["question"], result.answer, result.hits)
        supported = sum(c["supported"] for c in claims)
        checks = check_claim_citations(claims, result.answer, result.hits, judge)
        record.update(
            claims=claims,
            faithfulness=supported / len(claims) if claims else None,
            citation_checks=checks,
            citation_accuracy=sum(c["correct"] for c in checks) / len(checks) if checks else None,
        )
    return record


def rate(values: list[bool]) -> str:
    return f"{sum(values)}/{len(values)} = {sum(values) / len(values):.3f}" if values else "n/a"


def main() -> None:
    entries = [json.loads(line) for line in GOLDEN_SET.open(encoding="utf-8") if line.strip()]
    embedder, llm, judge = Embedder.load(), OllamaLLM(), Judge()

    records = []
    for entry in entries:
        records.append(evaluate(entry, embedder, llm, judge))
        print(f"  {entry['id']} done", file=sys.stderr, flush=True)

    answerable = [r for r in records if r["type"] != "unanswerable"]
    scored = [r for r in answerable if r["faithfulness"] is not None]
    unanswerable = [r for r in records if r["type"] == "unanswerable"]
    with_citations = [r for r in records if r["citations_valid"] is not None]
    refused_answerable = [r["id"] for r in answerable if r["is_refusal"]]
    empty_answerable = [r["id"] for r in answerable if r["is_empty"]]
    uncited_answerable = [r["id"] for r in answerable
                          if not r["is_refusal"] and not r["cited_numbers"]]
    citation_checks = [c | {"id": r["id"]} for r in scored for c in r["citation_checks"]]
    uncited_claims = sum(not c["citations"] for r in scored for c in r["claims"])

    summary = {
        "answer_rate": len(scored) / len(answerable),
        "mean_faithfulness": mean(r["faithfulness"] for r in scored) if scored else None,
        "faithfulness_by_type": {
            t: mean(r["faithfulness"] for r in scored if r["type"] == t)
            if any(r["type"] == t for r in scored) else None
            for t in TYPES
        },
        "answers_scored": len(scored),
        "answerable_refused": refused_answerable,
        "answerable_empty": empty_answerable,
        "refusal_rate_unanswerable": mean(r["refused"] for r in unanswerable),
        "exact_not_found_unanswerable": sum(r["exact_not_found"] for r in unanswerable),
        "citation_validity_rate": mean(r["citations_valid"] for r in with_citations) if with_citations else None,
        "answers_with_citations": len(with_citations),
        "answerable_without_citations": uncited_answerable,
        "citation_accuracy": (sum(c["correct"] for c in citation_checks) / len(citation_checks)
                              if citation_checks else None),
        "citations_checked": len(citation_checks),
        "claims_without_citation": uncited_claims,
    }

    print(f"Faithfulness, {len(records)} questions, judge {judge.model} at temperature 0\n")
    print(f"Answer rate            {len(scored)}/{len(answerable)} = {summary['answer_rate']:.3f}   "
          f"(answerable questions actually answered; refused: "
          f"{', '.join(refused_answerable) or 'none'}; empty: {', '.join(empty_answerable) or 'none'})")
    print(f"Mean faithfulness      {summary['mean_faithfulness']:.3f}   "
          f"(claims supported by the retrieved passages, over the {len(scored)} answered)")
    for t in TYPES:
        value = summary["faithfulness_by_type"][t]
        print(f"  {t:<12}         {value:.3f}" if value is not None else f"  {t:<12}         n/a")
    print(f"Refusal rate           {rate([r['refused'] for r in unanswerable])}   "
          f"(unanswerables; exact NOT_FOUND text in {summary['exact_not_found_unanswerable']})")
    print(f"Citation validity      {rate([r['citations_valid'] for r in with_citations])}   "
          f"(answers with at least one [n]; answerable with no citation: "
          f"{', '.join(uncited_answerable) or 'none'})")
    print(f"Citation accuracy      {rate([c['correct'] for c in citation_checks])}   "
          f"(cited passage alone supports the claim; claims with no citation: {uncited_claims})")
    wrong = [c for c in citation_checks if not c["correct"]]
    if wrong:
        print("  wrong citations: " + ", ".join(f"{c['id']} [{c['n']}]" for c in wrong))

    worst = sorted(scored, key=lambda r: (r["faithfulness"], r["id"]))[:WORST]
    print(f"\n{WORST} lowest-faithfulness answers:")
    for r in worst:
        print(f"\n  {r['id']} [{r['type']}] faithfulness {r['faithfulness']:.2f} "
              f"({sum(c['supported'] for c in r['claims'])}/{len(r['claims'])} claims)  {r['question']}")
        print(f"    answer: {r['answer']}")
        print(f"    retrieved: {', '.join(c[-3:] for c in r['retrieved_chunk_ids'])}"
              f"   cited: {', '.join(c[-3:] for c in r['cited_chunk_ids']) or 'none'}"
              f"   expected: {', '.join(c[-3:] for c in r['expected_chunk_ids'])}")
        for c in r["claims"]:
            if not c["supported"]:
                print(f"    UNSUPPORTED: {c['claim']}\n      judge: {c['reason']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"faithfulness_{date.today().isoformat()}.json"
    out.write_text(
        json.dumps({
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "answer_model": llm.model,
            "judge_model": judge.model,
            "judge_temperature": 0,
            "golden_set": GOLDEN_SET.relative_to(ROOT).as_posix(),
            "summary": summary,
            "questions": records,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nSaved to {out.relative_to(ROOT).as_posix()}")

    review = RESULTS_DIR / f"judge_review_{date.today().isoformat()}.md"
    review.write_text(judge_review(citation_checks, records), encoding="utf-8", newline="\n")
    print(f"Wrote {review.relative_to(ROOT).as_posix()} for a hand check of the judge")


def judge_review(checks: list[dict], records: list[dict]) -> str:
    """Markdown with a few random citation verdicts and a blank line for a person's own verdict."""
    seed = date.today().isoformat()  # same day, same sample
    picked = random.Random(seed).sample(checks, min(REVIEW_SAMPLE, len(checks)))
    chunks = {json.loads(line)["chunk_id"]: json.loads(line)
              for line in (ROOT / "data" / "chunks" / "plato_apology.jsonl").open(encoding="utf-8")}
    questions = {r["id"]: r["question"] for r in records}
    parts = [
        f"# Judge review, {seed}\n",
        f"{len(picked)} citation verdicts picked at random (seed `{seed}`) from the "
        f"{len(checks)} in `faithfulness_{seed}.json`. Each one asked the judge whether the cited "
        "passage alone supports the claim. Read the passage, then write your own verdict "
        "(supported / not supported) under each.\n",
    ]
    for i, c in enumerate(picked, start=1):
        chunk = chunks.get(c["chunk_id"]) if c["chunk_id"] else None
        # Blockquote every paragraph of the passage, not just the first.
        quoted = chunk["text"].replace("\n\n", "\n>\n> ") if chunk else ""
        passage = (f"`{c['chunk_id']}` ({chunk['work']}, {chunk['part']})\n\n> {quoted}"
                   if chunk else f"(no passage [{c['n']}] was retrieved)")
        parts.append(
            f"## {i}. {c['id']}, citation [{c['n']}]\n\n"
            f"**Question:** {questions[c['id']]}\n\n"
            f"**Claim:** {c['claim']}\n\n"
            f"**Cited passage [{c['n']}]:** {passage}\n\n"
            f"**Judge:** {'supported' if c['correct'] else 'not supported'}. {c['reason']}\n\n"
            "My verdict:\n"
        )
    return "\n".join(parts)


if __name__ == "__main__":
    main()
