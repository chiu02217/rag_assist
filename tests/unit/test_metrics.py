import math

from rag_assist.evaluation.metrics import (
    compute_metrics,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k():
    relevant = {"p1", "p3"}
    retrieved = ["p2", "p1", "p4", "p3"]

    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert recall_at_k(retrieved, relevant, 5) == 1.0


def test_reciprocal_rank():
    relevant = {"p1", "p3"}
    retrieved = ["p2", "p1", "p3"]

    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert reciprocal_rank(retrieved, relevant, 1) == 0.0


def test_ndcg_at_k():
    relevant = {"p1", "p3"}
    retrieved = ["p2", "p1", "p3"]

    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3)

    assert math.isclose(
        ndcg_at_k(retrieved, relevant, 3),
        dcg / idcg,
    )


def test_compute_metrics():
    result = compute_metrics(
        ["p1", "p2", "p3"],
        ["p1", "p3"],
    )

    assert result["recall@1"] == 0.5
    assert result["recall@3"] == 1.0
    assert result["mrr@10"] == 1.0
    assert result["ndcg@5"] > 0.9
