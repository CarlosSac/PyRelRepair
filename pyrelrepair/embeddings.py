"""Embeddings via Ollama (nomic-embed-text or any Ollama embedding model)."""
from __future__ import annotations

import logging

import numpy as np
import requests

logger = logging.getLogger(__name__)


def encode(
    texts: list[str],
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> np.ndarray:
    """Encode a batch of texts using an Ollama embedding model.

    Returns array of shape (len(texts), embedding_dim).
    """
    try:
        resp = requests.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        embeddings = resp.json()["embeddings"]
        return np.array(embeddings, dtype=np.float32)
    except requests.RequestException as e:
        logger.error("Ollama embed request failed: %s", e)
        raise
