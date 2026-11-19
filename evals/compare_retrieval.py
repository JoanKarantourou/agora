"""Compare two retrieval results files side by side, and list the questions only one of them hits.

Usage: uv run python evals/compare_retrieval.py A.json B.json
(e.g. evals/results/retrieval_vector_<date>.json evals/results/retrieval_keyword_<date>.json)
"""

import json
import sys
from pathlib import Path

ROWS = ["overall", "quote", "paraphrase", "concept"]
METRICS = [("hit_at_5", "hit@5"), ("recall_at_5", "recall@5"), ("mrr", "MRR")]


def load(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["label"] = data.get("mode", Path(path).stem)
    return data


def summary_row(data: dict, row: str) -> dict:
    s = data["summary"]
    return s["overall"] if row == "overall" else s["by_type"][row]


def only_hits(a: dict, b: dict) -> list[dict]:
    """Questions a hits (an expected chunk in its top 5) that b misses."""
    b_hits = {q["id"]: q["hit_at_5"] for q in b["questions"]}
    return [q for q in a["questions"] if q["hit_at_5"] and not b_hits[q["id"]]]


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])

    width = 26
    print(f"{'':<12}" + "".join(f"{d['label']:<{width}}" for d in (a, b)))
    print(f"{'':<12}" + "".join(f"{'hit@5  recall@5  MRR':<{width}}" for _ in (a, b)))
    for row in ROWS:
        cells = []
        for d in (a, b):
            s = summary_row(d, row)
            cells.append(f"{s['hit_at_5']:.3f}  {s['recall_at_5']:.3f}     {s['mrr']:.3f}".ljust(width))
        n = summary_row(a, row)["n"]
        print(f"{row:<12}" + "".join(cells) + f"(n={n})")

    for first, second in ((b, a), (a, b)):
        found = only_hits(first, second)
        print(f"\n{first['label']} hits, {second['label']} misses: {len(found)}")
        for q in found:
            rank = next(i for i, c in enumerate(q["returned_chunk_ids"], 1) if c in q["expected_chunk_ids"])
            print(f"  {q['id']} [{q['type']}] rank {rank} in {first['label']}  {q['question']}")


if __name__ == "__main__":
    main()
