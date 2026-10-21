import json
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

INPUT_FILE = Path("data/processed/chunks_structured.jsonl")
PARENTS_FILE = Path("data/processed/parents.jsonl")

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "kubernetes_docs"

MODEL_NAME = "BAAI/bge-m3"
VECTOR_SIZE = 1024

EMBED_BATCH_SIZE = 8
MAX_CHUNKS = None

def load_chunks():
    chunks = []
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
            if MAX_CHUNKS is not None and len(chunks) >= MAX_CHUNKS:
                break
    return chunks

def stable_point_id(chunk_id):
    return str(uuid5(NAMESPACE_URL, f"rag_assist:{chunk_id}"))

def ensure_collection(client):
    exists = any(
        c.name == COLLECTION_NAME
        for c in client.get_collections().collections
    )

    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Using existing collection: {COLLECTION_NAME}")

def ensure_payload_indexes(client):
    for field_name in ("category", "doc_type", "parent_id"):
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception as exc:
            print(f"Payload index {field_name}: {exc}")

def sync_parent_store():
    from parent_store import ParentStore

    store = ParentStore()
    count = store.upsert_parents_from_jsonl(PARENTS_FILE)

    print(f"Synced parent contexts: {count}")
    print(f"Persistent parent records: {store.count()}")

    store.close()

def load_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding device: {device}")

    model_kwargs = (
        {"torch_dtype": torch.float16}
        if device == "cuda"
        else {}
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        model_kwargs=model_kwargs,
    )

    print(
        f"Embedding dimension: "
        f"{model.get_sentence_embedding_dimension()}"
    )
    return model

def index_chunks(client, model, chunks):
    total = len(chunks)

    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[start:start + EMBED_BATCH_SIZE]

        embeddings = model.encode(
            [chunk["content"] for chunk in batch],
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        points = []

        for chunk, embedding in zip(batch, embeddings):
            payload = {
                "chunk_id": chunk["chunk_id"],
                "parent_id": chunk["parent_id"],
                "document_id": chunk["document_id"],
                "parent_index": chunk["parent_index"],
                "chunk_index": chunk["chunk_index"],
                "title": chunk["title"],
                "section": chunk["section"],
                "source": chunk["source"],
                "relative_path": chunk["relative_path"],
                "category": chunk["category"],
                "doc_type": chunk["doc_type"],
                "content": chunk["content"],
            }

            points.append(
                PointStruct(
                    id=stable_point_id(chunk["chunk_id"]),
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        current = min(start + len(batch), total)
        print(f"Indexed {current}/{total}")

def main():
    print("=" * 70)
    print("Parent-Child RAG Indexing")
    print("=" * 70)

    chunks = load_chunks()
    print(f"Loaded child chunks: {len(chunks)}")

    sync_parent_store()

    client = QdrantClient(url=QDRANT_URL)
    print("Connected to Qdrant.")

    ensure_collection(client)
    ensure_payload_indexes(client)

    model = load_embedding_model()
    index_chunks(client, model, chunks)

    info = client.get_collection(COLLECTION_NAME)
    print(f"Qdrant points: {info.points_count}")
    print("Indexing completed.")

if __name__ == "__main__":
    main()
