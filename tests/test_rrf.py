import unittest
from dataclasses import dataclass

from src.rrf_utils import weighted_reciprocal_rank_fusion


@dataclass
class DenseResult:
    payload: dict


class WeightedRRFTests(unittest.TestCase):
    def test_dense_weight_can_dominate(self):
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
        self.assertEqual(result[0]["chunk_id"], "A")

    def test_zero_weight_disables_source(self):
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
        self.assertEqual([x["chunk_id"] for x in result], ["A", "B"])

    def test_both_zero_is_invalid(self):
        with self.assertRaises(ValueError):
            weighted_reciprocal_rank_fusion([], [], 5, dense_weight=0, bm25_weight=0)


if __name__ == "__main__":
    unittest.main()
