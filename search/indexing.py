"""Elasticsearch index mapping and bulk indexing for advisor profiles."""

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from embed.model import EMBEDDING_DIM
from search.es_client import INDEX_NAME

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "advisor_id": {"type": "keyword"},
            "name": {"type": "text"},
            "bio": {"type": "text"},
            "tags": {"type": "keyword"},
            "years_experience": {"type": "float"},
            "experience_bucket": {"type": "keyword"},
            "embedding": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIM,
                "index": True,
                "similarity": "cosine",
            },
        }
    }
}


def ensure_index(client: Elasticsearch, index_name: str = INDEX_NAME) -> None:
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=INDEX_MAPPING)


def index_advisors(
    client: Elasticsearch,
    advisors: pd.DataFrame,
    embeddings,
    index_name: str = INDEX_NAME,
) -> int:
    """Bulk-indexes a cleaned advisors DataFrame alongside its bio embeddings."""
    actions = [
        {
            "_index": index_name,
            "_id": str(row.id),
            "_source": {
                "advisor_id": str(row.id),
                "name": row.name,
                "bio": row.bio,
                "tags": row.tags,
                "years_experience": None if pd.isna(row.years_experience) else float(row.years_experience),
                "experience_bucket": row.experience_bucket,
                "embedding": embeddings[i].tolist(),
            },
        }
        for i, row in enumerate(advisors.itertuples(index=False))
    ]

    success_count, _errors = bulk(client, actions, refresh="wait_for")
    return success_count
