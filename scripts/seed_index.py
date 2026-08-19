"""Cleans the seed advisor CSV, embeds every bio, and bulk-indexes the result.

Run: python -m scripts.seed_index
"""

from embed.model import get_embedder
from preprocess.clean import clean_advisors, load_advisors
from search.es_client import build_client
from search.indexing import ensure_index, index_advisors


def main() -> None:
    client = build_client()
    ensure_index(client)

    advisors = clean_advisors(load_advisors("data/advisors.csv"))
    embedder = get_embedder()
    embeddings = embedder.embed_texts(advisors["bio"].tolist())

    count = index_advisors(client, advisors, embeddings)
    print(f"Indexed {count} advisors.")


if __name__ == "__main__":
    main()
