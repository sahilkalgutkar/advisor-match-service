import os

from elasticsearch import Elasticsearch

INDEX_NAME = "advisors"


def build_client(url: str | None = None) -> Elasticsearch:
    return Elasticsearch(url or os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200"))
