"""
rag/vector_store.py
Build, save, and load a FAISS vector store for chunk retrieval.
"""

import os
import json
import pickle
from typing import List, Dict, Tuple

import numpy as np
import faiss

from rag.embeddings import embed_texts, embed_query

INDEX_DIR      = "vector_store"
INDEX_FILE     = os.path.join(INDEX_DIR, "faiss.index")
METADATA_FILE  = os.path.join(INDEX_DIR, "metadata.pkl")
STATS_FILE     = os.path.join(INDEX_DIR, "stats.json")


def build_index(chunks: List[Dict]) -> Tuple[faiss.Index, List[Dict]]:
    """
    Embed all chunks and build a FAISS flat inner-product index.
    (Embeddings are L2-normalised so inner product == cosine similarity.)
    """
    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks …")
    embeddings = embed_texts(texts)               # (N, D)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)                # inner product on normalised vectors
    index.add(embeddings.astype(np.float32))

    return index, chunks


def save_index(index: faiss.Index, chunks: List[Dict], doc_count: int) -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_FILE)

    with open(METADATA_FILE, "wb") as f:
        pickle.dump(chunks, f)

    stats = {
        "doc_count":   doc_count,
        "chunk_count": len(chunks),
        "embedding_model": "all-MiniLM-L6-v2",
    }
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Index saved → {INDEX_DIR}/  ({len(chunks)} chunks, {doc_count} docs)")


def load_index() -> Tuple[faiss.Index, List[Dict]]:
    """Load existing FAISS index and chunk metadata from disk."""
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(
            "Vector index not found. Please build the index first by clicking "
            "'Rebuild Knowledge Base' in the sidebar."
        )
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def index_exists() -> bool:
    return os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE)


def get_stats() -> Dict:
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE) as f:
            return json.load(f)
    return {"doc_count": 0, "chunk_count": 0}


def search(
    query: str,
    index: faiss.Index,
    chunks: List[Dict],
    top_k: int = 5,
    score_threshold: float = 0.0,
) -> List[Dict]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Returns a list of chunk dicts with an added "score" key.
    """
    q_vec = embed_query(query).astype(np.float32).reshape(1, -1)
    scores, indices = index.search(q_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:                    # FAISS sentinel for "no result"
            continue
        if float(score) < score_threshold:
            continue
        chunk = dict(chunks[idx])        # copy to avoid mutation
        chunk["score"] = float(score)
        results.append(chunk)

    return results
