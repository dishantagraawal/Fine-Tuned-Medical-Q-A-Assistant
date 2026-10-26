"""Sentence-transformer embeddings (BGE), with deterministic mock fallback."""

from __future__ import annotations

import hashlib
import logging
import re
from threading import Lock

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

_model = None
_lock = Lock()
_mock = False


def set_mock_embeddings(enabled: bool = True) -> None:
    global _mock, _model
    _mock = enabled
    if enabled:
        _model = None


def get_embedding_model():
    global _model
    settings = get_settings()
    if _mock or settings.mock_inference:
        return None
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s", settings.embedding_model)
            _model = SentenceTransformer(settings.embedding_model)
        return _model


def _stable_token_index(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % dim


def _mock_embed(texts: list[str], dim: int) -> np.ndarray:
    """Hashing bag-of-words embeddings so similar texts retrieve each other offline."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        vec = np.zeros(dim, dtype=np.float32)
        if not tokens:
            tokens = ["empty"]
        for tok in tokens:
            vec[_stable_token_index(tok, dim)] += 1.0
        norm = np.linalg.norm(vec) + 1e-9
        out[i] = vec / norm
    return out

def embed_texts(texts: list[str]) -> np.ndarray:
    settings = get_settings()
    dim = settings.embedding_dim
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    model = get_embedding_model()
    if model is None:
        return _mock_embed(texts, dim)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0].tolist()
