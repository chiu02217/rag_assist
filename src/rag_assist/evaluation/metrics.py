import math

EVAL_KS = (1, 3, 5)
NDCG_K = 5
MRR_K = 10


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(
    retrieved: list[str],
    relevant: set[str],
    k: int | None = None,
) -> float:
    limit = len(retrieved) if k is None else min(k, len(retrieved))
    for rank, item in enumerate(retrieved[:limit], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(k, len(relevant))
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )
    return dcg / idcg if idcg else 0.0


def compute_metrics(retrieved: list[str], relevant_ids: list[str]) -> dict:
    relevant = set(relevant_ids)
    metrics = {
        f"recall@{k}": recall_at_k(retrieved, relevant, k)
        for k in EVAL_KS
    }
    metrics[f"mrr@{MRR_K}"] = reciprocal_rank(
        retrieved,
        relevant,
        MRR_K,
    )
    metrics[f"ndcg@{NDCG_K}"] = ndcg_at_k(
        retrieved,
        relevant,
        NDCG_K,
    )
    return metrics
