import pandas as pd
import pytest
from testcontainers.community.elasticsearch import ElasticSearchContainer

from embed.model import get_embedder
from search.indexing import ensure_index, index_advisors
from search.query import hybrid_search

TEST_INDEX = "advisors-test"


@pytest.fixture(scope="module")
def es_client():
    container = ElasticSearchContainer(image="docker.elastic.co/elasticsearch/elasticsearch:8.15.0")
    container.with_env("xpack.security.enabled", "false")
    container.with_env("discovery.type", "single-node")
    container.start()

    from elasticsearch import Elasticsearch

    url = f"http://{container.get_container_host_ip()}:{container.get_exposed_port(9200)}"
    client = Elasticsearch(url)

    yield client

    container.stop()


@pytest.fixture(scope="module")
def indexed_client(es_client):
    ensure_index(es_client, index_name=TEST_INDEX)

    advisors = pd.DataFrame(
        [
            {
                "id": 1,
                "name": "Dr. Elena Vasquez",
                "bio": "Cardiologist specializing in heart surgery and cardiac care.",
                "tags": ["healthcare", "cardiology"],
                "years_experience": 18.0,
                "experience_bucket": "principal",
            },
            {
                "id": 2,
                "name": "Erik Johansson",
                "bio": "Maritime shipping and freight logistics expert.",
                "tags": ["shipping", "logistics"],
                "years_experience": 9.0,
                "experience_bucket": "mid",
            },
            {
                "id": 3,
                "name": "Helena Fischer",
                "bio": "Pharmaceutical commercialization expert focused on oncology drug launches.",
                "tags": ["pharma", "oncology"],
                "years_experience": 22.0,
                "experience_bucket": "principal",
            },
        ]
    )

    embedder = get_embedder()
    embeddings = embedder.embed_texts(advisors["bio"].tolist())
    index_advisors(es_client, advisors, embeddings, index_name=TEST_INDEX)

    return es_client


def test_hybrid_search_ranks_the_semantically_relevant_advisor_first(indexed_client):
    embedder = get_embedder()
    query_embedding = embedder.embed_text("Looking for an expert in heart disease and cardiac treatment.")

    results = hybrid_search(indexed_client, "heart disease cardiac treatment", query_embedding, k=3, index_name=TEST_INDEX)

    assert len(results) > 0
    assert results[0]["name"] == "Dr. Elena Vasquez"


def test_hybrid_search_respects_the_k_limit(indexed_client):
    embedder = get_embedder()
    query_embedding = embedder.embed_text("Any kind of business advisor.")

    results = hybrid_search(indexed_client, "business advisor", query_embedding, k=2, index_name=TEST_INDEX)

    assert len(results) <= 2


def test_index_advisors_reports_the_number_indexed(es_client):
    ensure_index(es_client, index_name="advisors-count-test")
    advisors = pd.DataFrame(
        [
            {
                "id": 100,
                "name": "Test Advisor",
                "bio": "A test bio for counting.",
                "tags": [],
                "years_experience": 5.0,
                "experience_bucket": "mid",
            }
        ]
    )
    embedder = get_embedder()
    embeddings = embedder.embed_texts(advisors["bio"].tolist())

    count = index_advisors(es_client, advisors, embeddings, index_name="advisors-count-test")

    assert count == 1
