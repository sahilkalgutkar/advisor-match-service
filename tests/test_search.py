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


def test_ensure_index_is_idempotent_when_the_index_already_exists(es_client):
    """Two Gunicorn workers both calling create_app() at startup can both see
    exists()=False before either has created the index - ensure_index must
    treat the resulting "already exists" error as success, not raise."""
    ensure_index(es_client, index_name="advisors-idempotent-test")

    ensure_index(es_client, index_name="advisors-idempotent-test")


def test_index_advisors_handles_a_missing_years_experience_and_bucket(es_client):
    """An advisor with no years-of-experience has no experience_bucket either
    (both come out as NaN from preprocess.clean.experience_bucket, not a
    plain Python None) - indexing must convert that to valid JSON null rather
    than sending the literal NaN token, which Elasticsearch's strict parser
    rejects outright."""
    ensure_index(es_client, index_name="advisors-missing-years-test")
    advisors = pd.DataFrame(
        [
            {
                "id": 200,
                "name": "No Experience Listed",
                "bio": "A bio with no years of experience on file.",
                "tags": [],
                "years_experience": float("nan"),
                "experience_bucket": float("nan"),
            }
        ]
    )
    embedder = get_embedder()
    embeddings = embedder.embed_texts(advisors["bio"].tolist())

    count = index_advisors(es_client, advisors, embeddings, index_name="advisors-missing-years-test")

    assert count == 1
    doc = es_client.get(index="advisors-missing-years-test", id="200")
    assert doc["_source"]["years_experience"] is None
    assert doc["_source"]["experience_bucket"] is None


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
