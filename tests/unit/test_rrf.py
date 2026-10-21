from dataclasses import dataclass

import pytest

from rag_assist.retrieval.fusion import weighted_reciprocal_rank_fusion


@dataclass
class DenseResult:
    payload: dict


def test_dense_weight_can_dominate():
    dense = [
        DenseResult({"chunk_id": "A", "content": "a"}),
        DenseResult({"chunk_id": "B", "content": "b"}),
    ]
    bm25 = [
        {"chunk_id": "B", "chunk": {"chunk_id": "B", "content": "b"}},
        {"chunk_id": "A", "chunk": {"chunk_id": "A", "content": "a"}},
    ]

    result = weighted_reciprocal_rank_fusion(
        dense,
        bm25,
        top_k=2,
        rrf_k=60,
        dense_weight=0.8,
        bm25_weight=0.2,
    )

    assert result[0]["chunk_id"] == "A"


def test_zero_weight_disables_source():
    dense = [
        DenseResult({"chunk_id": "A", "content": "a"}),
        DenseResult({"chunk_id": "B", "content": "b"}),
    ]
    bm25 = [
        {"chunk_id": "B", "chunk": {"chunk_id": "B", "content": "b"}},
        {"chunk_id": "A", "chunk": {"chunk_id": "A", "content": "a"}},
    ]

    result = weighted_reciprocal_rank_fusion(
        dense,
        bm25,
        top_k=2,
        dense_weight=1.0,
        bm25_weight=0.0,
    )

    assert [x["chunk_id"] for x in result] == ["A", "B"]


def test_both_zero_is_invalid():
    with pytest.raises(ValueError):
        weighted_reciprocal_rank_fusion(
            [],
            [],
            top_k=5,
            dense_weight=0.0,
            bm25_weight=0.0,
        )
