"""Hybrid search: kNN over the bio embedding blended with BM25 text match.

Pure vector search misses an advisor whose bio uses the exact term a client
searched for but phrases the rest of their expertise differently; pure BM25
misses an advisor whose bio is semantically on-topic but doesn't share
vocabulary with the query. Running both in one request and letting
Elasticsearch combine the scores catches what either alone would miss.
"""

import numpy as np
from elasticsearch import Elasticsearch

from search.es_client import INDEX_NAME


def hybrid_search(
    client: Elasticsearch,
    query_text: str,
    query_embedding: np.ndarray,
    k: int = 5,
    index_name: str = INDEX_NAME,
) -> list[dict]:
    response = client.search(
        index=index_name,
        knn={
            "field": "embedding",
            "query_vector": query_embedding.tolist(),
            "k": k,
            "num_candidates": max(k * 10, 50),
        },
        query={
            "multi_match": {
                "query": query_text,
                "fields": ["name^2", "bio", "tags"],
            }
        },
        # The 384-dim vector is what makes the match, not something a client
        # of this API needs back - excluding it at the ES level saves the
        # round trip too, not just the response payload.
        source_excludes=["embedding"],
        size=k,
    )

    return [
        {"advisor_id": hit["_source"]["advisor_id"], "score": hit["_score"], **hit["_source"]}
        for hit in response["hits"]["hits"]
    ]
