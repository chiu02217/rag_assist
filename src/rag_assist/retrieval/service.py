"""Lifecycle-managed retrieval service reused by chat/API."""
from __future__ import annotations
from typing import Any
from qdrant_client import QdrantClient
from rag_assist.config import get_settings
from rag_assist.retrieval.hybrid import (
    FilteredBM25Cache, bm25_search, build_qdrant_filter, dense_search,
    expand_to_parents, load_chunks, load_embedding_model, load_reranker,
    reciprocal_rank_fusion, rerank, resolve_scope,
)
from rag_assist.storage.parent_store import ParentStore

class HybridRetriever:
    def __init__(self, source_type: str = "all"):
        self.settings = get_settings()
        self.source_type = source_type
        self.scope = resolve_scope(source_type)
        self.chunks = load_chunks()
        if not self.chunks:
            raise RuntimeError(f"No child chunks loaded from {self.settings.chunks_file}")

        self.bm25_cache = FilteredBM25Cache(self.chunks)
        self.bm25_index = self.bm25_cache.get(
            category=self.scope["category"],
            doc_type=self.scope["doc_type"],
            source_type=self.scope["source_type"],
        )
        self.query_filter = build_qdrant_filter(
            category=self.scope["category"],
            doc_type=self.scope["doc_type"],
            source_type=self.scope["source_type"],
        )
        self.embedding_model = load_embedding_model()
        self.reranker = load_reranker()
        self.parent_store = ParentStore()
        if self.parent_store.count() == 0:
            self.parent_store.close()
            raise RuntimeError("Parent store is empty. Run the Qdrant indexer first.")
        self.client = QdrantClient(url=self.settings.qdrant_url)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        dense_results = dense_search(
            self.client, self.embedding_model, query,
            self.settings.dense_top_k, query_filter=self.query_filter,
        )
        lexical_results = bm25_search(
            self.bm25_index, query, self.settings.bm25_top_k,
        )
        fused = reciprocal_rank_fusion(
            dense_results, lexical_results,
            self.settings.rrf_top_k, self.settings.rrf_k,
        )
        reranked = rerank(
            self.reranker, query, fused, self.settings.rerank_top_k,
        )
        return expand_to_parents(
            reranked, self.parent_store,
            top_k or self.settings.final_top_parents,
        )

    def close(self) -> None:
        try:
            self.parent_store.close()
        finally:
            close = getattr(self.client, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "HybridRetriever":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
