"""Utilities for retrieval-fusion experiments."""


def _chunk_id_from_dense(result):
    return result.payload["chunk_id"]


def _chunk_from_dense(result):
    return result.payload


def weighted_reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    top_k,
    rrf_k=60,
    dense_weight=1.0,
    bm25_weight=1.0,
):
    """Fuse two ranked lists with independently tunable weights.

    The ranking contribution is:

        dense_weight / (rrf_k + dense_rank)
        bm25_weight / (rrf_k + bm25_rank)

    A weight of 0 disables that source. The function preserves the same
    result shape as the existing reciprocal_rank_fusion implementation.
    """
    if dense_weight < 0 or bm25_weight < 0:
        raise ValueError("RRF weights must be non-negative")
    if dense_weight == 0 and bm25_weight == 0:
        raise ValueError("At least one RRF weight must be greater than zero")
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")

    fused = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = _chunk_id_from_dense(result)
        fused.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "chunk": _chunk_from_dense(result),
                "dense_rank": None,
                "bm25_rank": None,
                "rrf_score": 0.0,
            },
        )
        fused[chunk_id]["dense_rank"] = rank
        fused[chunk_id]["rrf_score"] += dense_weight / (rrf_k + rank)

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
        fused[chunk_id]["rrf_score"] += bm25_weight / (rrf_k + rank)

    return sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )[:top_k]
