"""PyTorch/Hugging Face embeddings for advisor bios and client queries.

Uses a small sentence-transformers model (all-MiniLM-L6-v2, 384 dims) so it
stays fast on CPU in CI, but runs on CUDA automatically when it's available -
the actual inference is a real PyTorch forward pass through a Hugging Face
transformer, not a hosted API call.
"""

from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class AdvisorEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: torch.device | None = None):
        self.device = device or resolve_device()
        self.model = SentenceTransformer(model_name, device=str(self.device))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return embeddings.astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> AdvisorEmbedder:
    """Loading the model takes a couple seconds, so share one instance."""
    return AdvisorEmbedder()
