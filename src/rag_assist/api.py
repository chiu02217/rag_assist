"""Internal HTTP API for retrieval and document indexing."""

from __future__ import annotations

import json
import os
import secrets
import threading
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from rag_assist.chunking.parent_child import create_parent_child_data
from rag_assist.config import get_settings
from rag_assist.retrieval.service import HybridRetriever
from rag_assist.storage.parent_store import ParentStore
from rag_assist.storage.qdrant_indexer import (
    ensure_collection,
    ensure_payload_indexes,
    index_chunks,
)


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)
    top_k: int = Field(default=5, ge=1, le=50)
    source_type: Literal["documentation", "github_issue", "all", "uploaded"] = "all"


class IndexRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=2_000_000)
    source_url: str = Field(default="", max_length=2_000)
    source_id: str = Field(default="", max_length=500)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _require_internal_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("RAG_ASSIST_API_KEY", "").strip()
    if expected and (not x_api_key or not secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")


class RagRuntime:
    """Keeps models, Qdrant and SQLite open for the lifetime of the API process."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.lock = threading.RLock()
        self.retriever = HybridRetriever(source_type="all")
        self.index_client = QdrantClient(url=self.settings.qdrant_url)
        ensure_collection(self.index_client)
        ensure_payload_indexes(self.index_client)
        self.index_model = self.retriever.embedding_model
        self.parent_store: ParentStore = self.retriever.parent_store

    def retrieve(self, query: str, top_k: int, source_type: str) -> list[dict[str, Any]]:
        with self.lock:
            self.retriever.scope = self._scope(source_type)
            from rag_assist.retrieval.hybrid import build_qdrant_filter

            self.retriever.query_filter = build_qdrant_filter(**self.retriever.scope)
            self.retriever.bm25_index = self.retriever.bm25_cache.get(
                **self.retriever.scope
            )
            results = self.retriever.retrieve(query, top_k=top_k)
            normalized = []
            for item in results[:top_k]:
                parent = item["parent"]
                child = item["best_child"]
                normalized.append(
                    {
                        "parent_id": item["parent_id"],
                        "document_id": parent["document_id"],
                        "title": parent["title"],
                        "section": parent["section"],
                        "content": parent["content"],
                        "source": parent["source"],
                        "source_type": parent["source_type"],
                        "source_url": parent.get("source_url") or None,
                        "relative_path": parent.get("relative_path", ""),
                        "chunk_id": child["chunk_id"],
                        "score": float(child["reranker_score"]),
                        "support_count": item["support_count"],
                    }
                )
            return normalized

    @staticmethod
    def _scope(source_type: str) -> dict[str, str | None]:
        from rag_assist.retrieval.hybrid import resolve_scope

        if source_type not in {"all", "documentation", "github_issue", "uploaded"}:
            raise ValueError(f"Unsupported source_type: {source_type}")
        if source_type == "uploaded":
            return {"source_type": "uploaded", "category": "uploaded", "doc_type": "uploaded"}
        return resolve_scope(source_type)

    def index_document(self, request: IndexRequest) -> dict[str, Any]:
        document = {
            "document_id": request.document_id,
            "title": request.title,
            "content": request.content,
            "source": "AI Workflow RAG uploads",
            "source_type": "uploaded",
            "source_id": request.source_id or request.document_id,
            "source_url": request.source_url,
            "relative_path": request.source_url or request.document_id,
            "category": "uploaded",
            "doc_type": "uploaded",
            "metadata": request.metadata,
        }
        parents, chunks = create_parent_child_data([document])
        if not chunks:
            raise ValueError("No non-empty text chunks were produced")
        parent_rows = [vars(parent) for parent in parents]
        chunk_rows = [vars(chunk) for chunk in chunks]
        with self.lock:
            self.parent_store.upsert_parents(parent_rows)
            index_chunks(self.index_client, self.index_model, chunk_rows)
            known_ids = {chunk["chunk_id"] for chunk in self.retriever.chunks}
            fresh_chunks = [chunk for chunk in chunk_rows if chunk["chunk_id"] not in known_ids]
            if fresh_chunks:
                self.settings.chunks_file.parent.mkdir(parents=True, exist_ok=True)
                with self.settings.chunks_file.open(
                    "a", encoding="utf-8"
                ) as handle:
                    for chunk in fresh_chunks:
                        handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            self.retriever.chunks.extend(fresh_chunks)
            from rag_assist.retrieval.hybrid import FilteredBM25Cache

            self.retriever.bm25_cache = FilteredBM25Cache(self.retriever.chunks)
        return {
            "document_id": request.document_id,
            "parents": len(parent_rows),
            "chunks": len(chunk_rows),
            "collection": self.settings.qdrant_collection,
        }

    def close(self) -> None:
        self.retriever.close()
        self.index_client.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = RagRuntime()
    yield
    app.state.runtime.close()


app = FastAPI(
    title="rag_assist Retrieval API",
    version="1.0.0",
    description="Internal hybrid retrieval and legacy-index ingestion API.",
    lifespan=lifespan,
)


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def ready(
    runtime: RagRuntime = Depends(lambda: app.state.runtime),
) -> dict[str, Any]:
    info = runtime.index_client.get_collection(runtime.settings.qdrant_collection)
    return {
        "status": "ready",
        "collection": runtime.settings.qdrant_collection,
        "points": info.points_count,
        "parent_contexts": runtime.parent_store.count(),
    }


@app.post("/api/v1/retrieve", dependencies=[Depends(_require_internal_key)])
@app.post("/retrieve", dependencies=[Depends(_require_internal_key)], include_in_schema=False)
def retrieve(
    request: RetrieveRequest,
    runtime: RagRuntime = Depends(lambda: app.state.runtime),
) -> dict[str, Any]:
    try:
        results = runtime.retrieve(request.query, request.top_k, request.source_type)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Retrieval failed: {type(exc).__name__}",
        ) from exc
    return {"query": request.query, "results": results}


@app.post("/documents", dependencies=[Depends(_require_internal_key)])
def index_document(
    request: IndexRequest,
    runtime: RagRuntime = Depends(lambda: app.state.runtime),
) -> dict[str, Any]:
    try:
        return runtime.index_document(request)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Indexing failed: {type(exc).__name__}",
        ) from exc
