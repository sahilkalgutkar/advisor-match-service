import numpy as np
import pytest
import torch

from embed.model import EMBEDDING_DIM, AdvisorEmbedder, get_embedder, resolve_device


@pytest.fixture(scope="module")
def embedder() -> AdvisorEmbedder:
    return get_embedder()


def test_resolve_device_returns_cpu_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device() == torch.device("cpu")


def test_embed_texts_returns_the_expected_shape(embedder):
    embeddings = embedder.embed_texts(["A cardiologist with 20 years of experience.", "A shipping logistics expert."])

    assert embeddings.shape == (2, EMBEDDING_DIM)
    assert embeddings.dtype == np.float32


def test_embed_texts_returns_empty_array_for_no_input(embedder):
    embeddings = embedder.embed_texts([])

    assert embeddings.shape == (0, EMBEDDING_DIM)


def test_embeddings_are_l2_normalized(embedder):
    embeddings = embedder.embed_texts(["Some advisor bio about healthcare payer strategy."])

    norm = np.linalg.norm(embeddings[0])
    assert norm == pytest.approx(1.0, abs=1e-3)


def test_semantically_similar_bios_embed_closer_together_than_dissimilar_ones(embedder):
    cardiology_a = embedder.embed_text("Cardiologist specializing in heart surgery and cardiac care.")
    cardiology_b = embedder.embed_text("Expert in cardiac medicine and cardiovascular treatment.")
    shipping = embedder.embed_text("Maritime shipping and freight logistics expert.")

    similar_score = float(np.dot(cardiology_a, cardiology_b))
    dissimilar_score = float(np.dot(cardiology_a, shipping))

    assert similar_score > dissimilar_score


def test_get_embedder_returns_the_same_cached_instance():
    assert get_embedder() is get_embedder()
