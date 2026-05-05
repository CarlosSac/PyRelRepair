"""Similarity search and retrieval for SigRepair and SnipRepair."""
from __future__ import annotations

import logging

import numpy as np

from .code_parser import FunctionInfo
from .embeddings import encode

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a @ b.T


def retrieve_similar_signatures(
    query_text: str,
    candidate_functions: list[FunctionInfo],
    top_k: int = 25,
    embed_model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> list[tuple[FunctionInfo, float]]:
    """Retrieve top-K most similar function signatures using nomic-embed-text.

    Used by SigRepair. Encodes the rewritten query and each candidate's
    signature+docstring, then ranks by cosine similarity.
    """
    if not candidate_functions:
        return []

    candidate_texts = [f.signature_with_doc for f in candidate_functions]
    all_texts = [query_text] + candidate_texts
    all_embs = encode(all_texts, model=embed_model, base_url=base_url)

    query_emb = all_embs[:1]
    candidate_embs = all_embs[1:]

    sims = _cosine_similarity(query_emb, candidate_embs).flatten()
    top_indices = np.argsort(sims)[::-1][:top_k]
    results = [(candidate_functions[i], float(sims[i])) for i in top_indices]

    logger.info(
        "SigRetrieval: retrieved %d signatures (top sim=%.4f, bottom sim=%.4f)",
        len(results),
        results[0][1] if results else 0,
        results[-1][1] if results else 0,
    )
    return results


def retrieve_similar_snippets(
    buggy_code: str,
    buggy_comments: str,
    candidate_functions: list[FunctionInfo],
    top_k: int = 15,
    embed_model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> list[tuple[FunctionInfo, float]]:
    """Retrieve top-K most similar code snippets using nomic-embed-text.

    Used by SnipRepair. Encodes the buggy function's code+comments and
    each candidate, then ranks by cosine similarity.
    """
    if not candidate_functions:
        return []

    query_text = buggy_code + "\n" + buggy_comments
    candidate_texts = [f.code + "\n" + f.comments for f in candidate_functions]
    all_texts = [query_text] + candidate_texts
    all_embs = encode(all_texts, model=embed_model, base_url=base_url)

    query_emb = all_embs[:1]
    candidate_embs = all_embs[1:]

    sims = _cosine_similarity(query_emb, candidate_embs).flatten()
    top_indices = np.argsort(sims)[::-1][:top_k]
    results = [(candidate_functions[i], float(sims[i])) for i in top_indices]

    logger.info(
        "SnipRetrieval: retrieved %d snippets (top sim=%.4f)",
        len(results),
        results[0][1] if results else 0,
    )
    return results
