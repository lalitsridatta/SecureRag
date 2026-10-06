"""
rag/embeddings.py
Generate sentence embeddings using a local sentence-transformers model.
No API calls — runs fully offline.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# Lightweight, fast model — good balance of quality and speed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_model: SentenceTransformer = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of strings.
    Returns an ndarray of shape (len(texts), embedding_dim).
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalise for cosine similarity via dot product
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (embedding_dim,)."""
    return embed_texts([query])[0]
