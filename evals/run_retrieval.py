"""Score retrieval against the golden set: hit@5, recall@5 and MRR, overall and per question type.

--mode vector (default) uses the same search function as /search and /ask, with the real model;
--mode keyword uses BM25 keyword search (app.keyword_search);
--mode hybrid fuses the two with Reciprocal Rank Fusion (app.hybrid_search);
--mode rerank pools both searches' candidates and re-sorts them with a cross-encoder (app.rerank),
and also reports pool recall (the ceiling for reranking) and the reranker's timings.
All read the local database.
Unanswerable questions are skipped: they have no expected chunks to find.
Prints a summary and saves full results to evals/results/retrieval_<mode>_<date>.json.
"""

import argparse
import json
import sys
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from statistics import mean, quantiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.embeddings import MODEL_NAME, Embedder  # noqa: E402
from app.hybrid_search import CANDIDATES, RRF_K, hybrid_search  # noqa: E402
from app.keyword_search import keyword_search  # noqa: E402
from app.rerank import POOL_PER_SEARCH, RERANK_MODEL, Reranker, rerank_search_details  # noqa: E402
from app.search import SearchHit, search_chunks  # noqa: E402

GOLDEN_SET = ROOT / "evals" / "golden_set.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"
TOP_K = 5
TYPES = ["quote", "paraphrase", "concept"]
MODES = ["vector", "keyword", "hybrid", "rerank"]


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


Search = Callable[[str], tuple[list[SearchHit], dict]]


def searcher(mode: str) -> Search:
    """A function from question to (hits, extra per-question fields for the results file)."""
    if mode == "keyword":
        return lambda question: (keyword_search(question, TOP_K), {})
    embedder = Embedder.load()
    if mode == "hybrid":
        return lambda question: (hybrid_search(question, TOP_K, embedder), {})
    if mode == "rerank":
        reranker = Reranker.load()
        # One untimed call first, so GPU start-up isn't counted in the timings.
        rerank_search_details("warm-up", TOP_K, embedder, reranker)

        def search(question: str) -> tuple[list[SearchHit], dict]:
            run = rerank_search_details(question, TOP_K, embedder, reranker)
            return run.hits, {"pool_chunk_ids": [h.chunk_id for h in run.pool],
                              "rerank_ms": round(run.rerank_ms, 1)}

        return search
    return lambda question: (search_chunks(question, TOP_K, embedder), {})


def pool_stats(records: list[dict]) -> dict:
    """For rerank: how often the expected chunks were in the pool at all, and how fast it was."""
    times = sorted(r["rerank_ms"] for r in records)
    return {
        "pool_hit": mean(r["pool_hit"] for r in records),
        "pool_recall": mean(r["pool_recall"] for r in records),
        "pool_misses": [r["id"] for r in records if not r["pool_hit"]],
        "mean_pool_size": mean(len(r["pool_chunk_ids"]) for r in records),
        "rerank_ms_mean": round(mean(times), 1),
        "rerank_ms_p95": round(quantiles(times, n=20, method="inclusive")[-1], 1),
        "rerank_ms_max": times[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=MODES, default="vector")
    mode = parser.parse_args().mode

    entries = [json.loads(line) for line in GOLDEN_SET.open(encoding="utf-8") if line.strip()]
    scored = [e for e in entries if e["type"] != "unanswerable"]
    search = searcher(mode)

    records = []
    for e in scored:
        hits, extra = search(e["question"])
        returned = [h.chunk_id for h in hits]
        if "pool_chunk_ids" in extra:
            in_pool = [c for c in e["expected_chunk_ids"] if c in extra["pool_chunk_ids"]]
            extra |= {"pool_hit": bool(in_pool),
                      "pool_recall": len(in_pool) / len(e["expected_chunk_ids"])}
        records.append({
            "id": e["id"],
            "type": e["type"],
            "question": e["question"],
            "expected_chunk_ids": e["expected_chunk_ids"],
            "returned_chunk_ids": returned,
            # cosine similarity (vector), BM25 (keyword), RRF (hybrid) or reranker score (rerank)
            "returned_scores": [round(h.similarity, 4) for h in hits],
            **score(e["expected_chunk_ids"], returned),
            **extra,
        })

    overall = summarise(records)
    by_type = {t: summarise([r for r in records if r["type"] == t]) for t in TYPES}
    misses = [r for r in records if not r["hit_at_5"]]
    pool = pool_stats(records) if mode == "rerank" else None

    print(f"Retrieval ({mode}), top {TOP_K}, {len(records)} questions "
          f"({len(entries) - len(scored)} unanswerable skipped)\n")
    print(f"Overall      {fmt(overall)}")
    for t in TYPES:
        print(f"{t:<12} {fmt(by_type[t])}")
    if pool:
        n = len(records)
        print()
        print(f"Pool recall  {round(pool['pool_hit'] * n)}/{n} = {pool['pool_hit']:.3f} questions have an "
              f"expected chunk in the pool (all expected chunks: {pool['pool_recall']:.3f}); "
              f"not in pool: {', '.join(pool['pool_misses']) or 'none'}")
        print(f"Pool size    mean {pool['mean_pool_size']:.1f} chunks (top {POOL_PER_SEARCH} vector "
              f"+ top {POOL_PER_SEARCH} keyword, deduplicated)")
        print(f"Rerank time  mean {pool['rerank_ms_mean']:.1f} ms, p95 {pool['rerank_ms_p95']:.1f} ms, "
              f"max {pool['rerank_ms_max']:.1f} ms per question (reranker call only)")
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
            "model": {
                "vector": MODEL_NAME,
                "keyword": "BM25Okapi (rank_bm25)",
                "hybrid": f"RRF(k={RRF_K}) of {MODEL_NAME} and BM25Okapi, top {CANDIDATES} each",
                "rerank": f"{RERANK_MODEL} over top {POOL_PER_SEARCH} {MODEL_NAME} "
                          f"+ top {POOL_PER_SEARCH} BM25Okapi",
            }[mode],
            "top_k": TOP_K,
            "golden_set": GOLDEN_SET.relative_to(ROOT).as_posix(),
            "skipped_unanswerable": len(entries) - len(scored),
            "summary": {"overall": overall, "by_type": by_type} | ({"pool": pool} if pool else {}),
            "misses": [r["id"] for r in misses],
            "questions": records,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nSaved to {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
