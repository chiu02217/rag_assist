import json
import re
from pathlib import Path
from dataclasses import dataclass

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

INPUT_FILE = Path("data/processed/chunks_structured.jsonl")
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "kubernetes_docs"

MODEL_NAME = "BAAI/bge-m3"
DENSE_TOP_K = 30
BM25_TOP_K = 30

RRF_TOP_K = 20
RRF_K = 60

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_BATCH_SIZE = 4
RERANKER_MAX_LENGTH = 512

FINAL_TOP_PARENTS = 5

@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunks: list

def load_chunks():
    chunks = []
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def tokenize(text: str):
    return re.findall(
        r"[a-zA-Z0-9_./:-]+",
        text.lower(),
    )

def build_bm25_index(chunks):
    if not chunks:
        return None
    corpus = [tokenize(c["content"]) for c in chunks]
    return BM25Index(BM25Okapi(corpus), chunks)

def filter_chunks(chunks, category=None, doc_type=None):
    filtered = []
    for chunk in chunks:
        if category is not None and chunk.get("category") != category:
            continue
        if doc_type is not None and chunk.get("doc_type") != doc_type:
            continue
        filtered.append(chunk)
    return filtered

class FilteredBM25Cache:
    """
    Cache filtered lexical indexes by metadata scope.
    For production at larger scale, replace rank-bm25 with
    OpenSearch/Elasticsearch while keeping the same interface.
    """
    def __init__(self, chunks):
        self.chunks = chunks
        self.cache = {}

    def get(self, category=None, doc_type=None):
        key = (category, doc_type)
        if key not in self.cache:
            candidates = filter_chunks(
                self.chunks,
                category=category,
                doc_type=doc_type,
            )
            print(
                f"Building BM25 candidate index for {key}: "
                f"{len(candidates)} chunks"
            )
            self.cache[key] = build_bm25_index(candidates)
        return self.cache[key]

def bm25_search(bm25_index, query, top_k):
    if bm25_index is None:
        return []

    scores = bm25_index.bm25.get_scores(tokenize(query))

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    return [
        {
            "chunk_id": bm25_index.chunks[i]["chunk_id"],
            "chunk": bm25_index.chunks[i],
            "score": float(scores[i]),
        }
        for i in ranked_indices
    ]

def load_embedding_model():
    # Retrieval query is tiny; keep GPU free for reranker.
    print("Query embedding device: CPU")
    return SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

def build_qdrant_filter(category=None, doc_type=None):
    conditions = []

    if category:
        conditions.append(
            models.FieldCondition(
                key="category",
                match=models.MatchValue(value=category),
            )
        )

    if doc_type:
        conditions.append(
            models.FieldCondition(
                key="doc_type",
                match=models.MatchValue(value=doc_type),
            )
        )

    return models.Filter(must=conditions) if conditions else None

def dense_search(client, model, query, top_k, query_filter=None):
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

def reciprocal_rank_fusion(dense_results, bm25_results, top_k, rrf_k=60):
    fused = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result.payload["chunk_id"]
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
        fused[chunk_id]["dense_rank"] = rank
        fused[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank)

    for rank, result in enumerate(bm25_results, start=1):
        chunk_id = result["chunk_id"]
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
        fused[chunk_id]["bm25_rank"] = rank
        fused[chunk_id]["rrf_score"] += 1.0 / (rrf_k + rank)

    return sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )[:top_k]

def load_reranker():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Reranker device: {device}")

    return CrossEncoder(
        RERANKER_MODEL_NAME,
        device=device,
        max_length=RERANKER_MAX_LENGTH,
    )

def rerank(reranker, query, results, top_k):
    if not results:
        return []

    pairs = [
        [query, result["chunk"]["content"]]
        for result in results
    ]

    scores = reranker.predict(
        pairs,
        batch_size=RERANKER_BATCH_SIZE,
        show_progress_bar=False,
    )

    reranked = []

    for result, score in zip(results, scores):
        item = result.copy()
        item["reranker_score"] = float(score)
        reranked.append(item)

    reranked.sort(
        key=lambda x: x["reranker_score"],
        reverse=True,
    )

    return reranked[:top_k]

def expand_to_parents(reranked_children, parent_store, top_k):
    """
    Parent-child retrieval:
      1. Retrieve/rerank small child chunks.
      2. Deduplicate by parent_id.
      3. Fetch each parent once from persistent storage.
      4. Return parent contexts to the next LLM stage.
    """
    grouped = {}

    for rank, child_result in enumerate(
        reranked_children,
        start=1,
    ):
        parent_id = get_parent_id(
            child_result["chunk"]
        )
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
            or child_result["reranker_score"]
            > group["best_child"]["reranker_score"]
        ):
            group["best_child"] = child_result
            group["best_child_rank"] = rank

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
                "best_child": group["best_child"],
                "support_count": group["support_count"],
            }
        )

    final.sort(
        key=lambda x: (
            x["best_child"]["reranker_score"],
            x["support_count"],
        ),
        reverse=True,
    )

    return final[:top_k]

def print_child_results(title, results, limit=10):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for rank, result in enumerate(results[:limit], start=1):
        if "reranker_score" in result:
            print(
                f"\n[{rank}] "
                f"Reranker: {result['reranker_score']:.4f} "
                f"| RRF: {result['rrf_score']:.5f} "
                f"| Dense: {result.get('dense_rank')} "
                f"| BM25: {result.get('bm25_rank')}"
            )
        elif "rrf_score" in result:
            print(
                f"\n[{rank}] "
                f"RRF: {result['rrf_score']:.5f} "
                f"| Dense: {result.get('dense_rank')} "
                f"| BM25: {result.get('bm25_rank')}"
            )
        elif "score" in result:
            print(
                f"\n[{rank}] "
                f"BM25: {result['score']:.4f}"
            )
        else:
            print(
                f"\n[{rank}] "
                f"Dense: {result.score:.4f}"
            )

        payload = result["chunk"] if isinstance(result, dict) else result.payload

        print(f"Title: {payload['title']}")
        print(f"Section: {payload['section']}")
        print(f"Path: {payload['relative_path']}")
        print(f"Parent ID: {payload['parent_id']}")
        print(f"\n{payload['content'][:500]}")

def print_parent_results(results):
    print("\n" + "=" * 80)
    print("PARENT CONTEXT RESULTS")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):
        parent = result["parent"]
        child = result["best_child"]

        print(
            f"\n[{rank}] "
            f"Reranker: {child['reranker_score']:.4f} "
            f"| Supporting child chunks: "
            f"{result['support_count']}"
        )

        print(f"Title: {parent['title']}")
        print(f"Section: {parent['section']}")
        print(f"Path: {parent['relative_path']}")
        print(f"Parent ID: {parent['parent_id']}")

        print("\n--- Parent Context ---")
        print(parent["content"])

def get_parent_id(chunk):
    """
    Get parent_id from payload.

    Fallback:
    derive it from the deterministic chunk_id:
        <parent_id>:chunk:<index>
    """

    parent_id = chunk.get(
        "parent_id"
    )

    if parent_id:
        return parent_id

    chunk_id = chunk.get(
        "chunk_id"
    )

    if chunk_id and ":chunk:" in chunk_id:

        return chunk_id.rsplit(
            ":chunk:",
            1
        )[0]

    raise ValueError(
        "Child chunk is missing parent_id. "
        f"chunk_id={chunk_id}"
    )

def main():
    # For this first experiment, keep the filter explicit.
    # Later move it to the API request / query router.
    category = "concepts"
    doc_type = "documentation"

    query_filter = build_qdrant_filter(
        category=category,
        doc_type=doc_type,
    )

    print("=" * 80)
    print("Parent-Child Hybrid RAG")
    print("=" * 80)

    chunks = load_chunks()
    print(f"Loaded child chunks: {len(chunks)}")

    from parent_store import ParentStore

    parent_store = ParentStore()

    if parent_store.count() == 0:
        raise RuntimeError(
            "Parent store is empty. Run src/indexer.py first."
        )

    bm25_cache = FilteredBM25Cache(chunks)

    bm25_index = bm25_cache.get(
        category=category,
        doc_type=doc_type,
    )

    embedding_model = load_embedding_model()
    reranker = load_reranker()

    client = QdrantClient(url=QDRANT_URL)

    print("\nReady.")
    print(
        f"Metadata filter: "
        f"category={category}, "
        f"doc_type={doc_type}"
    )

    try:
        while True:
            query = input("Query > ").strip()

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

            reranked_children = rerank(
                reranker,
                query,
                rrf_results,
                RRF_TOP_K,
            )

            parent_results = expand_to_parents(
                reranked_children,
                parent_store,
                FINAL_TOP_PARENTS,
            )

            print_child_results(
                "DENSE RESULTS",
                dense_results,
            )

            print_child_results(
                "BM25 RESULTS",
                bm25_results,
            )

            print_child_results(
                "HYBRID / RRF RESULTS",
                rrf_results,
            )

            print_child_results(
                "RERANKED CHILD RESULTS",
                reranked_children,
            )

            print_parent_results(
                parent_results
            )

    finally:
        parent_store.close()

if __name__ == "__main__":
    main()
