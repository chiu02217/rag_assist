"""Interactive source-aware Parent/Child hybrid retrieval."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag_assist.config import get_settings
from rag_assist.storage.parent_store import ParentStore


settings = get_settings()

INPUT_FILE = settings.chunks_file

QDRANT_URL = settings.qdrant_url
COLLECTION_NAME = settings.qdrant_collection

MODEL_NAME = settings.embedding_model

DENSE_TOP_K = settings.dense_top_k
BM25_TOP_K = settings.bm25_top_k

RRF_TOP_K = settings.rrf_top_k
RRF_K = settings.rrf_k

RERANKER_MODEL_NAME = settings.reranker_model
RERANKER_BATCH_SIZE = settings.reranker_batch_size
RERANKER_MAX_LENGTH = settings.reranker_max_length

FINAL_TOP_PARENTS = settings.final_top_parents


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunks: list[dict]


def load_chunks() -> list[dict]:
    chunks = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(
        r"[a-zA-Z0-9_./:-]+",
        text.lower(),
    )


def build_bm25_index(
    chunks: list[dict],
) -> BM25Index | None:
    if not chunks:
        return None

    corpus = [
        tokenize(chunk["content"])
        for chunk in chunks
    ]

    return BM25Index(
        bm25=BM25Okapi(corpus),
        chunks=chunks,
    )


def filter_chunks(
    chunks: list[dict],
    *,
    category: str | None = None,
    doc_type: str | None = None,
    source_type: str | None = None,
) -> list[dict]:
    filtered = []

    for chunk in chunks:
        if (
            category is not None
            and chunk.get("category") != category
        ):
            continue

        if (
            doc_type is not None
            and chunk.get("doc_type") != doc_type
        ):
            continue

        if (
            source_type is not None
            and chunk.get("source_type")
            != source_type
        ):
            continue

        filtered.append(chunk)

    return filtered


class FilteredBM25Cache:
    """In-memory filtered BM25 indexes for local development."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.cache: dict[
            tuple[
                str | None,
                str | None,
                str | None,
            ],
            BM25Index | None,
        ] = {}

    def get(
        self,
        *,
        category: str | None = None,
        doc_type: str | None = None,
        source_type: str | None = None,
    ) -> BM25Index | None:
        key = (
            category,
            doc_type,
            source_type,
        )

        if key not in self.cache:
            candidates = filter_chunks(
                self.chunks,
                category=category,
                doc_type=doc_type,
                source_type=source_type,
            )

            print(
                f"Building BM25 candidate index "
                f"for {key}: "
                f"{len(candidates)} chunks"
            )

            self.cache[key] = (
                build_bm25_index(candidates)
            )

        return self.cache[key]


def bm25_search(
    bm25_index: BM25Index | None,
    query: str,
    top_k: int,
) -> list[dict]:
    if bm25_index is None:
        return []

    scores = bm25_index.bm25.get_scores(
        tokenize(query)
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]

    return [
        {
            "chunk_id": bm25_index.chunks[index][
                "chunk_id"
            ],
            "chunk": bm25_index.chunks[index],
            "score": float(scores[index]),
        }
        for index in ranked_indices
    ]


def load_embedding_model():
    print(
        "Query embedding device: CPU"
    )

    return SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )


def build_qdrant_filter(
    *,
    category: str | None = None,
    doc_type: str | None = None,
    source_type: str | None = None,
):
    conditions = []

    if category is not None:
        conditions.append(
            models.FieldCondition(
                key="category",
                match=models.MatchValue(
                    value=category
                ),
            )
        )

    if doc_type is not None:
        conditions.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchValue(
                    value=doc_type
                ),
            )
        )

    if source_type is not None:
        conditions.append(
            models.FieldCondition(
                key="source_type",
                match=models.MatchValue(
                    value=source_type
                ),
            )
        )

    if not conditions:
        return None

    return models.Filter(
        must=conditions
    )


def dense_search(
    client: QdrantClient,
    model: SentenceTransformer,
    query: str,
    top_k: int,
    query_filter=None,
):
    vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector.tolist(),
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    ).points


def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    top_k: int,
    rrf_k: int = RRF_K,
):
    fused = {}

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):
        chunk_id = result.payload[
            "chunk_id"
        ]

        fused.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "chunk": result.payload,
                "dense_rank": None,
                "bm25_rank": None,
                "rrf_score": 0.0,
            },
        )

        fused[chunk_id][
            "dense_rank"
        ] = rank

        fused[chunk_id][
            "rrf_score"
        ] += 1.0 / (
            rrf_k + rank
        )

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):
        chunk_id = result[
            "chunk_id"
        ]

        fused.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "chunk": result["chunk"],
                "dense_rank": None,
                "bm25_rank": None,
                "rrf_score": 0.0,
            },
        )

        fused[chunk_id][
            "bm25_rank"
        ] = rank

        fused[chunk_id][
            "rrf_score"
        ] += 1.0 / (
            rrf_k + rank
        )

    return sorted(
        fused.values(),
        key=lambda item: item[
            "rrf_score"
        ],
        reverse=True,
    )[:top_k]


def load_reranker():
    device = (
        settings.reranker_device
    )

    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print(
        f"Reranker device: {device}"
    )

    return CrossEncoder(
        RERANKER_MODEL_NAME,
        device=device,
        max_length=RERANKER_MAX_LENGTH,
    )


def rerank(
    reranker,
    query: str,
    results: list[dict],
    top_k: int,
) -> list[dict]:
    if not results:
        return []

    pairs = [
        [
            query,
            result["chunk"]["content"],
        ]
        for result in results
    ]

    scores = reranker.predict(
        pairs,
        batch_size=RERANKER_BATCH_SIZE,
        show_progress_bar=False,
    )

    ranked = []

    for result, score in zip(
        results,
        scores,
    ):
        item = result.copy()
        item["reranker_score"] = float(
            score
        )
        ranked.append(item)

    ranked.sort(
        key=lambda item: item[
            "reranker_score"
        ],
        reverse=True,
    )

    return ranked[:top_k]


def expand_to_parents(
    reranked_children: list[dict],
    parent_store: ParentStore,
    top_k: int,
) -> list[dict]:
    grouped = {}

    for rank, child_result in enumerate(
        reranked_children,
        start=1,
    ):
        parent_id = child_result[
            "chunk"
        ]["parent_id"]

        grouped.setdefault(
            parent_id,
            {
                "parent_id": parent_id,
                "best_child": None,
                "best_child_rank": None,
                "support_count": 0,
            },
        )

        group = grouped[parent_id]

        group["support_count"] += 1

        if (
            group["best_child"] is None
            or child_result[
                "reranker_score"
            ]
            > group[
                "best_child"
            ][
                "reranker_score"
            ]
        ):
            group["best_child"] = (
                child_result
            )
            group["best_child_rank"] = (
                rank
            )

    parents = parent_store.get_parents(
        grouped.keys()
    )

    final = []

    for parent_id, group in grouped.items():
        parent = parents.get(parent_id)

        if parent is None:
            continue

        final.append(
            {
                "parent_id": parent_id,
                "parent": parent,
                "best_child": group[
                    "best_child"
                ],
                "support_count": group[
                    "support_count"
                ],
            }
        )

    final.sort(
        key=lambda item: (
            item["best_child"][
                "reranker_score"
            ],
            item["support_count"],
        ),
        reverse=True,
    )

    return final[:top_k]


def print_parent_results(
    results: list[dict],
) -> None:
    print(
        "\n"
        + "=" * 80
    )
    print(
        "PARENT CONTEXT RESULTS"
    )
    print(
        "=" * 80
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        parent = result[
            "parent"
        ]
        child = result[
            "best_child"
        ]

        print(
            f"\n[{rank}] "
            f"Reranker: "
            f"{child['reranker_score']:.4f} "
            f"| Supporting child chunks: "
            f"{result['support_count']}"
        )

        print(
            f"Source Type: "
            f"{parent.get('source_type', 'unknown')}"
        )
        print(
            f"Title: {parent['title']}"
        )
        print(
            f"Section: {parent['section']}"
        )
        print(
            f"Source URL: "
            f"{parent.get('source_url', '')}"
        )
        print(
            f"Source ID: "
            f"{parent.get('source_id', '')}"
        )
        print(
            f"Path: {parent['relative_path']}"
        )

        print(
            "\n--- Parent Context ---"
        )
        print(
            parent["content"]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive source-aware "
            "Parent/Child hybrid retrieval."
        )
    )

    parser.add_argument(
        "--source-type",
        choices=(
            "documentation",
            "github_issue",
            "all",
        ),
        default=settings.default_source_type,
    )

    return parser.parse_args()


def resolve_scope(
    source_type: str,
):
    if source_type == "documentation":
        return {
            "source_type": "documentation",
            "category": settings.default_category,
            "doc_type": settings.default_doc_type,
        }

    if source_type == "github_issue":
        return {
            "source_type": "github_issue",
            "category": "github_issue",
            "doc_type": "github_issue",
        }

    return {
        "source_type": None,
        "category": None,
        "doc_type": None,
    }


def main() -> None:
    args = parse_args()
    scope = resolve_scope(
        args.source_type
    )

    print("=" * 80)
    print("Source-aware Parent-Child Hybrid RAG")
    print("=" * 80)

    print(
        f"Retrieval scope: "
        f"{args.source_type}"
    )

    chunks = load_chunks()

    print(
        f"Loaded child chunks: "
        f"{len(chunks)}"
    )

    query_filter = build_qdrant_filter(
        category=scope["category"],
        doc_type=scope["doc_type"],
        source_type=scope["source_type"],
    )

    bm25_cache = FilteredBM25Cache(
        chunks
    )

    bm25_index = bm25_cache.get(
        category=scope["category"],
        doc_type=scope["doc_type"],
        source_type=scope["source_type"],
    )

    embedding_model = load_embedding_model()
    reranker = load_reranker()

    parent_store = ParentStore()

    if parent_store.count() == 0:
        parent_store.close()
        raise RuntimeError(
            "Parent store is empty. "
            "Run qdrant_indexer first."
        )

    client = QdrantClient(
        url=QDRANT_URL
    )

    print("\nReady.")

    try:
        while True:
            query = input(
                "Query > "
            ).strip()

            if query.lower() == "exit":
                break

            if not query:
                continue

            dense_results = dense_search(
                client,
                embedding_model,
                query,
                DENSE_TOP_K,
                query_filter=query_filter,
            )

            bm25_results = bm25_search(
                bm25_index,
                query,
                BM25_TOP_K,
            )

            rrf_results = reciprocal_rank_fusion(
                dense_results,
                bm25_results,
                RRF_TOP_K,
                RRF_K,
            )

            reranked = rerank(
                reranker,
                query,
                rrf_results,
                settings.rerank_top_k,
            )

            parent_results = expand_to_parents(
                reranked,
                parent_store,
                FINAL_TOP_PARENTS,
            )

            print_parent_results(
                parent_results
            )

    finally:
        parent_store.close()


if __name__ == "__main__":
    main()
