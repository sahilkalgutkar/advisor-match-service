import pytest
from testcontainers.community.elasticsearch import ElasticSearchContainer

from api.app import create_app

TEST_INDEX = "advisors-api-test"


@pytest.fixture(scope="module")
def client():
    container = ElasticSearchContainer(image="docker.elastic.co/elasticsearch/elasticsearch:8.15.0")
    container.with_env("xpack.security.enabled", "false")
    container.with_env("discovery.type", "single-node")
    container.start()

    from elasticsearch import Elasticsearch

    url = f"http://{container.get_container_host_ip()}:{container.get_exposed_port(9200)}"
    es_client = Elasticsearch(url)

    app = create_app(es_client=es_client, index_name=TEST_INDEX)
    with app.test_client() as test_client:
        yield test_client

    container.stop()


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_create_advisor_requires_a_name_and_bio(client):
    response = client.post("/advisors", json={"name": "Jane Doe"})

    assert response.status_code == 400


def test_create_and_fetch_an_advisor(client):
    created = client.post(
        "/advisors",
        json={
            "name": "Dr. Elena Vasquez",
            "bio": "Cardiologist specializing in heart surgery and cardiac care.",
            "tags": ["healthcare", "cardiology"],
            "years_experience": 18,
        },
    )
    assert created.status_code == 201
    advisor_id = created.get_json()["advisor_id"]

    fetched = client.get(f"/advisors/{advisor_id}")

    assert fetched.status_code == 200
    body = fetched.get_json()
    assert body["name"] == "Dr. Elena Vasquez"
    assert body["experience_bucket"] == "senior"
    assert "embedding" not in body


def test_get_unknown_advisor_returns_404(client):
    response = client.get("/advisors/does-not-exist")

    assert response.status_code == 404


def test_match_requires_a_query(client):
    response = client.post("/match", json={})

    assert response.status_code == 400


def test_match_ranks_the_semantically_relevant_advisor_first(client):
    client.post(
        "/advisors",
        json={
            "name": "Erik Johansson",
            "bio": "Maritime shipping and freight logistics expert.",
            "tags": ["shipping", "logistics"],
            "years_experience": 9,
        },
    )

    response = client.post("/match", json={"query": "heart disease and cardiac treatment expert", "k": 5})

    assert response.status_code == 200
    results = response.get_json()["results"]
    assert len(results) > 0
    assert results[0]["name"] == "Dr. Elena Vasquez"
    assert "embedding" not in results[0]
