"""Compare retrieval results files side by side, and list the questions only some of them hit.

Usage: uv run python evals/compare_retrieval.py A.json B.json [C.json ...]
(e.g. evals/results/retrieval_hybrid_<date>.json retrieval_vector_<date>.json retrieval_keyword_<date>.json)

With three or more files, also reports how many of the questions any of the other files hit
the first file keeps, and which it loses.
"""

import json
import sys
from pathlib import Path

ROWS = ["overall", "quote", "paraphrase", "concept"]
WIDTH = 26


def load(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["label"] = data.get("mode", Path(path).stem)
    data["hits"] = {q["id"] for q in data["questions"] if q["hit_at_5"]}
    data["by_id"] = {q["id"]: q for q in data["questions"]}
    return data


def summary_row(data: dict, row: str) -> dict:
    s = data["summary"]
    return s["overall"] if row == "overall" else s["by_type"][row]


def first_rank(q: dict) -> int:
    return next(i for i, c in enumerate(q["returned_chunk_ids"], 1) if c in q["expected_chunk_ids"])


def print_question(data: dict, qid: str, note: str) -> None:
    q = data["by_id"][qid]
    print(f"  {qid} [{q['type']}] {note}  {q['question']}")


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    runs = [load(p) for p in sys.argv[1:]]

    print(f"{'':<12}" + "".join(f"{d['label']:<{WIDTH}}" for d in runs))
    print(f"{'':<12}" + "".join(f"{'hit@5  recall@5  MRR':<{WIDTH}}" for _ in runs))
    for row in ROWS:
        cells = []
        for d in runs:
            s = summary_row(d, row)
            cells.append(f"{s['hit_at_5']:.3f}  {s['recall_at_5']:.3f}     {s['mrr']:.3f}".ljust(WIDTH))
        print(f"{row:<12}" + "".join(cells) + f"(n={summary_row(runs[0], row)['n']})")

    first = runs[0]
    for other in runs[1:]:
        for a, b in ((first, other), (other, first)):
            only = sorted(a["hits"] - b["hits"])
            print(f"\n{a['label']} hits, {b['label']} misses: {len(only)}")
            for qid in only:
                print_question(a, qid, f"rank {first_rank(a['by_id'][qid])} in {a['label']}")

    if len(runs) >= 3:
        union = set().union(*(d["hits"] for d in runs[1:]))
        lost = sorted(union - first["hits"])
        labels = " or ".join(d["label"] for d in runs[1:])
        print(f"\n{first['label']} keeps {len(union) - len(lost)}/{len(union)} of the questions "
              f"{labels} hits; lost: {len(lost)}")
        for qid in lost:
            found_by = ", ".join(f"{d['label']} rank {first_rank(d['by_id'][qid])}"
                                 for d in runs[1:] if qid in d["hits"])
            print_question(first, qid, f"(found by {found_by})")
        gained = sorted(first["hits"] - union)
        if gained:
            print(f"{first['label']} also hits {len(gained)} that none of the others hit: {', '.join(gained)}")


if __name__ == "__main__":
    main()
