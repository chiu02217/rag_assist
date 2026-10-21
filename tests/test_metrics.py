import math
import sys
from pathlib import Path
import unittest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from retrieval_metrics import recall_at_k, reciprocal_rank, ndcg_at_k, compute_metrics


class RetrievalMetricTests(unittest.TestCase):
    def setUp(self):
        self.relevant = {"p1", "p3"}

    def test_recall(self):
        retrieved = ["p2", "p1", "p4", "p3"]
        self.assertEqual(recall_at_k(retrieved, self.relevant, 1), 0.0)
        self.assertEqual(recall_at_k(retrieved, self.relevant, 3), 0.5)
        self.assertEqual(recall_at_k(retrieved, self.relevant, 5), 1.0)

    def test_mrr(self):
        retrieved = ["p2", "p1", "p3"]
        self.assertAlmostEqual(reciprocal_rank(retrieved, self.relevant), 0.5)
        self.assertEqual(reciprocal_rank(retrieved, self.relevant, 1), 0.0)

    def test_ndcg(self):
        retrieved = ["p2", "p1", "p3"]
        dcg = 1 / math.log2(3) + 1 / math.log2(4)
        idcg = 1 + 1 / math.log2(3)
        self.assertAlmostEqual(ndcg_at_k(retrieved, self.relevant, 3), dcg / idcg)

    def test_compute_metrics(self):
        result = compute_metrics(["p1", "p2", "p3"], ["p1", "p3"])
        self.assertEqual(result["recall@1"], 0.5)
        self.assertEqual(result["recall@3"], 1.0)
        self.assertEqual(result["mrr@10"], 1.0)
        self.assertGreater(result["ndcg@5"], 0.9)


if __name__ == "__main__":
    unittest.main()
