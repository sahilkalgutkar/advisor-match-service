import pytest
from testcontainers.community.elasticsearch import ElasticSearchContainer

from embed.model import get_embedder
from eval.run import run
from preprocess.clean import clean_advisors, load_advisors
from search.indexing import ensure_index, index_advisors

TEST_INDEX = "advisors-eval-test"


@pytest.fixture(scope="module")
def indexed_client():
    container = ElasticSearchContainer(image="docker.elastic.co/elasticsearch/elasticsearch:8.15.0")
    container.with_env("xpack.security.enabled", "false")
    container.with_env("discovery.type", "single-node")
    container.start()

    from elasticsearch import Elasticsearch

    url = f"http://{container.get_container_host_ip()}:{container.get_exposed_port(9200)}"
    client = Elasticsearch(url)

    ensure_index(client, index_name=TEST_INDEX)
    advisors = clean_advisors(load_advisors("data/advisors.csv"))
    embedder = get_embedder()
    embeddings = embedder.embed_texts(advisors["bio"].tolist())
    index_advisors(client, advisors, embeddings, index_name=TEST_INDEX)

    yield client

    container.stop()


def test_eval_hit_rate_against_the_real_seed_dataset_is_reasonably_high(indexed_client):
    report = run(client=indexed_client, index_name=TEST_INDEX, k=5)

    # Not 100% - hybrid search over a small, hand-written dataset won't be
    # perfect, and the point of this eval is a real measured number rather
    # than a rubber-stamped one. But most of these 10 queries were written
    # to closely paraphrase their expected advisor's bio, so a reasonable
    # hybrid search should find most of them in the top 5.
    assert report["hit_rate"] >= 0.7
