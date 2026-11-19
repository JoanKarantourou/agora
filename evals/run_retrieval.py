"""Score retrieval against the golden set: hit@5, recall@5 and MRR, overall and per question type.

--mode vector (default) uses the same search function as /search and /ask, with the real model;
--mode keyword uses BM25 keyword search (app.keyword_search). Both read the local database.
Unanswerable questions are skipped: they have no expected chunks to find.
Prints a summary and saves full results to evals/results/retrieval_<mode>_<date>.json.
"""

import argparse
import json
import sys
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.embeddings import MODEL_NAME, Embedder  # noqa: E402
from app.keyword_search import keyword_search  # noqa: E402
from app.search import SearchHit, search_chunks  # noqa: E402

GOLDEN_SET = ROOT / "evals" / "golden_set.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"
TOP_K = 5
TYPES = ["quote", "paraphrase", "concept"]
MODES = ["vector", "keyword"]


def score(expected: list[str], returned: list[str]) -> dict[str, float]:
    found = [cid for cid in expected if cid in returned]
    first_rank = next((rank for rank, cid in enumerate(returned, start=1) if cid in expected), None)
    return {
        "hit_at_5": 1.0 if found else 0.0,
        "recall_at_5": len(found) / len(expected),
        "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
    }


def summarise(records: list[dict]) -> dict[str, float | int]:
    return {
        "n": len(records),
        "hit_at_5": mean(r["hit_at_5"] for r in records),
        "recall_at_5": mean(r["recall_at_5"] for r in records),
        "mrr": mean(r["reciprocal_rank"] for r in records),
    }


def fmt(s: dict) -> str:
    return f"hit@5 {s['hit_at_5']:.3f}   recall@5 {s['recall_at_5']:.3f}   MRR {s['mrr']:.3f}   (n={s['n']})"


def searcher(mode: str) -> Callable[[str], list[SearchHit]]:
    if mode == "keyword":
        return lambda question: keyword_search(question, TOP_K)
    embedder = Embedder.load()
    return lambda question: search_chunks(question, TOP_K, embedder)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=MODES, default="vector")
    mode = parser.parse_args().mode

    entries = [json.loads(line) for line in GOLDEN_SET.open(encoding="utf-8") if line.strip()]
    scored = [e for e in entries if e["type"] != "unanswerable"]
    search = searcher(mode)

    records = []
    for e in scored:
        hits = search(e["question"])
        returned = [h.chunk_id for h in hits]
        records.append({
            "id": e["id"],
            "type": e["type"],
            "question": e["question"],
            "expected_chunk_ids": e["expected_chunk_ids"],
            "returned_chunk_ids": returned,
            # cosine similarity in vector mode, BM25 score in keyword mode
            "returned_scores": [round(h.similarity, 4) for h in hits],
            **score(e["expected_chunk_ids"], returned),
        })

    overall = summarise(records)
    by_type = {t: summarise([r for r in records if r["type"] == t]) for t in TYPES}
    misses = [r for r in records if not r["hit_at_5"]]

    print(f"Retrieval ({mode}), top {TOP_K}, {len(records)} questions "
          f"({len(entries) - len(scored)} unanswerable skipped)\n")
    print(f"Overall      {fmt(overall)}")
    for t in TYPES:
        print(f"{t:<12} {fmt(by_type[t])}")
    print(f"\nMisses (no expected chunk in the top {TOP_K}): {len(misses)}")
    for r in misses:
        print(f"  {r['id']} [{r['type']}] {r['question']}")
        print(f"    expected: {', '.join(r['expected_chunk_ids'])}")
        print(f"    returned: {', '.join(r['returned_chunk_ids'])}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"retrieval_{mode}_{date.today().isoformat()}.json"
    out.write_text(
        json.dumps({
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "model": MODEL_NAME if mode == "vector" else "BM25Okapi (rank_bm25)",
            "top_k": TOP_K,
            "golden_set": GOLDEN_SET.relative_to(ROOT).as_posix(),
            "skipped_unanswerable": len(entries) - len(scored),
            "summary": {"overall": overall, "by_type": by_type},
            "misses": [r["id"] for r in misses],
            "questions": records,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nSaved to {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
