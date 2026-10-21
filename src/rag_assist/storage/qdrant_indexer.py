"""Qdrant indexing for the unified multi-source child corpus."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from rag_assist.config import get_settings
from rag_assist.storage.parent_store import ParentStore


settings = get_settings()

INPUT_FILE = settings.chunks_file
PARENTS_FILE = settings.parents_file

QDRANT_URL = settings.qdrant_url
COLLECTION_NAME = settings.qdrant_collection

MODEL_NAME = settings.embedding_model
VECTOR_SIZE = settings.vector_size

EMBED_BATCH_SIZE = settings.embed_batch_size
MAX_CHUNKS = settings.max_chunks


def load_chunks() -> list[dict]:
    chunks = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if line.strip():
                chunks.append(json.loads(line))

            if (
                MAX_CHUNKS is not None
                and len(chunks) >= MAX_CHUNKS
            ):
                break

    return chunks


def stable_point_id(chunk_id: str) -> str:
    return str(
        uuid5(
            NAMESPACE_URL,
            f"rag_assist:{chunk_id}",
        )
    )


def ensure_collection(
    client: QdrantClient,
) -> None:
    exists = any(
        item.name == COLLECTION_NAME
        for item in client.get_collections().collections
    )

    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(
            f"Created collection: {COLLECTION_NAME}"
        )
    else:
        print(
            f"Using existing collection: {COLLECTION_NAME}"
        )


def ensure_payload_indexes(
    client: QdrantClient,
) -> None:
    for field_name in (
        "category",
        "doc_type",
        "parent_id",
        "source_type",
        "source_id",
        "status",
    ):
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception as exc:
            # Qdrant may report "already exists"; keep indexing.
            print(
                f"Payload index {field_name}: {exc}"
            )


def load_embedding_model():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Embedding device: {device}"
    )

    model_kwargs = (
        {"torch_dtype": torch.float16}
        if device == "cuda"
        else {}
    )

    return SentenceTransformer(
        MODEL_NAME,
        device=device,
        model_kwargs=model_kwargs,
    )


def index_chunks(
    client: QdrantClient,
    model: SentenceTransformer,
    chunks: list[dict],
) -> None:
    total = len(chunks)

    for start in range(
        0,
        total,
        EMBED_BATCH_SIZE,
    ):
        batch = chunks[
            start:start + EMBED_BATCH_SIZE
        ]

        embeddings = model.encode(
            [
                chunk["content"]
                for chunk in batch
            ],
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        points = []

        for chunk, embedding in zip(
            batch,
            embeddings,
        ):
            metadata = dict(
                chunk.get("metadata") or {}
            )

            payload = {
                "chunk_id": chunk["chunk_id"],
                "parent_id": chunk["parent_id"],
                "document_id": chunk["document_id"],
                "parent_index": chunk["parent_index"],
                "chunk_index": chunk["chunk_index"],
                "title": chunk["title"],
                "section": chunk["section"],
                "source": chunk["source"],
                "source_type": chunk.get(
                    "source_type",
                    "documentation",
                ),
                "source_id": chunk.get(
                    "source_id",
                    "",
                ),
                "source_url": chunk.get(
                    "source_url",
                    "",
                ),
                "relative_path": chunk["relative_path"],
                "category": chunk["category"],
                "doc_type": chunk["doc_type"],
                "content": chunk["content"],
            }

            for key in (
                "status",
                "labels",
                "github_repo",
                "github_issue_number",
                "created_at",
                "updated_at",
            ):
                if key in metadata:
                    payload[key] = metadata[key]

            points.append(
                PointStruct(
                    id=stable_point_id(
                        chunk["chunk_id"]
                    ),
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        current = min(
            start + len(batch),
            total,
        )

        print(
            f"Indexed {current}/{total}"
        )


def sync_parent_store() -> None:
    store = ParentStore()

    try:
        count = (
            store.upsert_parents_from_jsonl(
                PARENTS_FILE
            )
        )

        print(
            f"Synced parent contexts: {count}"
        )
        print(
            f"Persistent parent records: "
            f"{store.count()}"
        )
    finally:
        store.close()


def main() -> None:
    print("=" * 80)
    print("Multi-source Parent-Child RAG Indexing")
    print("=" * 80)

    chunks = load_chunks()
    print(
        f"Loaded child chunks: {len(chunks)}"
    )

    sync_parent_store()

    client = QdrantClient(
        url=QDRANT_URL
    )

    print(
        "Connected to Qdrant."
    )

    ensure_collection(client)
    ensure_payload_indexes(client)

    model = load_embedding_model()

    index_chunks(
        client,
        model,
        chunks,
    )

    info = client.get_collection(
        COLLECTION_NAME
    )

    print(
        f"Qdrant points: {info.points_count}"
    )
    print(
        "Indexing completed."
    )


if __name__ == "__main__":
    main()
