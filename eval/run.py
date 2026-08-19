"""Run the hand-built query set against the current index and report:
- hit@k: did the expected advisor show up in the top-k results
- MRR: mean reciprocal rank of the expected advisor across all queries

Intentionally simple (10 queries, exact name match) so it stays easy to
reason about - the point is a real, measured number against the actual
index, not a fancy harness.

Run: python -m eval.run
"""

from pathlib import Path

import yaml

from embed.model import get_embedder
from search.es_client import INDEX_NAME, build_client
from search.query import hybrid_search

QUERIES_PATH = Path("eval/queries.yaml")


def load_queries() -> list[dict]:
    with open(QUERIES_PATH) as f:
        return yaml.safe_load(f)["queries"]


def run(client=None, index_name: str = INDEX_NAME, k: int = 5) -> dict:
    client = client or build_client()
    embedder = get_embedder()
    queries = load_queries()

    hits = 0
    reciprocal_ranks = []

    for q in queries:
        query_embedding = embedder.embed_text(q["query"])
        results = hybrid_search(client, q["query"], query_embedding, k=k, index_name=index_name)
        names = [r["name"] for r in results]

        hit = q["expected_name"] in names
        hits += hit
        rank = names.index(q["expected_name"]) + 1 if hit else 0
        reciprocal_ranks.append(1 / rank if rank else 0.0)

        status = "HIT " if hit else "MISS"
        print(f"[{status}] {q['query']!r} -> expected {q['expected_name']!r}")

    n = len(queries)
    hit_rate = hits / n
    mrr = sum(reciprocal_ranks) / n
    print(f"\nhit@{k}: {hits}/{n} ({hit_rate:.0%})")
    print(f"MRR: {mrr:.2f}")

    return {"hit_rate": hit_rate, "mrr": mrr, "n": n}


if __name__ == "__main__":
    run()
