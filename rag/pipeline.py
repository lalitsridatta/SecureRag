"""
rag/pipeline.py
Main RAG pipeline — clean interface for the frontend and future security layer.

Usage:
    from rag.pipeline import RAGPipeline

    rag = RAGPipeline()
    result = rag.answer("What are the attendance requirements?")

    # result = {
    #     "answer":           "The minimum attendance requirement is 75% ...",
    #     "sources":          ["academic_regulations.pdf", "attendance_policy.pdf"],
    #     "retrieved_chunks": [ {...}, {...} ],
    #     "model":            "gemini-2.5-flash",
    #     "error":            None,
    # }

The transformer security gate and NLI validation layer will be inserted
AROUND this pipeline later — this function remains unchanged.
"""

import os
from typing import Dict, List, Optional

from rag.document_loader import load_documents
from rag.chunker         import chunk_documents
from rag.vector_store    import (
    build_index, save_index, load_index,
    index_exists, search, get_stats,
)
from rag.generator       import generate_answer


class RAGPipeline:
    """
    Encapsulates the full RAG flow:
      query → embedding → vector search → context → LLM → answer + sources
    """

    def __init__(
        self,
        knowledge_base_dir: str = "knowledge_base",
        top_k: int = 5,
        score_threshold: float = 0.0,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ):
        self.knowledge_base_dir = knowledge_base_dir
        self.top_k              = top_k
        self.score_threshold    = score_threshold
        self.chunk_size         = chunk_size
        self.chunk_overlap      = chunk_overlap

        self._index  = None
        self._chunks: List[Dict] = []

        # Auto-load existing index if available
        if index_exists():
            self._load()

    # ── Index management ─────────────────────────────────────────────────────

    def build(self) -> Dict:
        """
        Load documents, chunk them, embed, and save the vector index.
        Returns a summary dict.
        """
        print("Building RAG index …")

        docs   = load_documents(self.knowledge_base_dir)
        chunks = chunk_documents(docs, self.chunk_size, self.chunk_overlap)

        self._index, self._chunks = build_index(chunks)
        save_index(self._index, self._chunks, doc_count=len(docs))

        return {
            "doc_count":   len(docs),
            "chunk_count": len(chunks),
        }

    def _load(self) -> None:
        """Load a pre-built index from disk."""
        self._index, self._chunks = load_index()

    def is_ready(self) -> bool:
        return self._index is not None and len(self._chunks) > 0

    def get_stats(self) -> Dict:
        return get_stats()

    # ── Core answer method ───────────────────────────────────────────────────

    def answer(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Dict:
        """
        Run the full RAG pipeline for a user query.

        This is the clean interface that the security layer will wrap later.

        Returns:
            {
                "answer":           str,
                "sources":          List[str],   # unique filenames
                "retrieved_chunks": List[Dict],  # full chunk dicts with scores
                "model":            str,
                "error":            Optional[str],
            }
        """
        query = (query or "").strip()
        if not query:
            return {
                "answer": "Please enter a question.",
                "sources": [],
                "retrieved_chunks": [],
                "model": "none",
                "error": "Empty query.",
            }

        if not self.is_ready():
            return {
                "answer": (
                    "The knowledge base has not been indexed yet. "
                    "Please click 'Rebuild Knowledge Base' in the sidebar."
                ),
                "sources": [],
                "retrieved_chunks": [],
                "model": "none",
                "error": "Index not built.",
            }

        k = top_k if top_k is not None else self.top_k

        # 1. Retrieve relevant chunks
        retrieved = search(
            query,
            self._index,
            self._chunks,
            top_k=k,
            score_threshold=self.score_threshold,
        )

        # 2. Collect unique sources
        sources = list(dict.fromkeys(c["source"] for c in retrieved))

        # 3. Generate answer
        result = generate_answer(query, retrieved)

        return {
            "answer":           result["answer"],
            "sources":          sources,
            "retrieved_chunks": retrieved,
            "model":            result["model"],
            "error":            result["error"],
        }
