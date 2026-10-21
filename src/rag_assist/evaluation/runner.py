import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from rag_assist.config import get_settings
# Allow importing the existing retrieval implementation from src/.
# SRC_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = get_settings().project_root
# if str(SRC_DIR) not in sys.path:
#     sys.path.insert(0, str(SRC_DIR))
PROJECT_ROOT = Path.cwd()
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "data" / "eval" / "golden_dataset.jsonl"
)

from qdrant_client import QdrantClient

from rag_assist.retrieval.hybrid import (
    BM25_TOP_K,
    DENSE_TOP_K,
    FINAL_TOP_PARENTS,
    RRF_K,
    RRF_TOP_K,
    FilteredBM25Cache,
    build_qdrant_filter,
    dense_search,
    bm25_search,
    load_chunks,
    load_embedding_model,
    load_reranker,
    reciprocal_rank_fusion,
    rerank,
    expand_to_parents,
)
from rag_assist.storage.parent_store import ParentStore

DATASET_FILE = get_settings().eval_dataset_file
RESULT_JSON = get_settings().eval_results_json
RESULT_CSV = get_settings().eval_results_csv
PARENTS_FILE = get_settings().parents_file


def normalize_path(value: str) -> str:
    return str(value).replace("\\", "/").strip().lower().lstrip("./")


def normalize_section(value: str) -> str:
    # Current chunker stores the heading text without the leading # markers.
    # Remove optional Hugo/Markdown anchor suffixes for robust label matching.
    value = str(value).strip()
    if "{#" in value:
        value = value.split("{#", 1)[0].rstrip()
    return " ".join(value.split()).lower()


def stable_section_key(relative_path: str, section: str) -> str:
    return f"{normalize_path(relative_path)}::{normalize_section(section)}"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def validate_dataset(dataset: list[dict], parents: list[dict]) -> dict:
    index = defaultdict(list)
    for parent in parents:
        key = stable_section_key(
            parent["relative_path"],
            parent["section"],
        )
        index[key].append(parent)

    errors = []
    resolved = []

    for query_idx, item in enumerate(dataset, start=1):
        if not item.get("query"):
            errors.append(f"query #{query_idx}: missing query")
            continue

        relevant_keys = []
        resolved_debug_parent_ids = []
        for label_idx, label in enumerate(item.get("relevant", []), start=1):
            path = label.get("relative_path")
            section = label.get("section")
            if not path or not section:
                errors.append(
                    f"query #{query_idx} label #{label_idx}: "
                    "relative_path and section are required"
                )
                continue

            matches = index.get(
                stable_section_key(path, section),
                [],
            )
            if not matches:
                errors.append(
                    f"query #{query_idx} label #{label_idx}: "
                    f"no parent found for {path} / {section}"
                )
                continue

            relevant_keys.append(stable_section_key(path, section))
            resolved_debug_parent_ids.extend(p["parent_id"] for p in matches)

        relevant_keys = list(dict.fromkeys(relevant_keys))
        if not relevant_keys:
            errors.append(f"query #{query_idx}: no resolvable relevant sections")
            continue

        resolved.append(
            {
                "query_index": query_idx,
                "query": item["query"],
                "filters": item.get("filters", {}),
                "relevant_keys": relevant_keys,
                "resolved_parent_ids": list(dict.fromkeys(resolved_debug_parent_ids)),
            }
        )

    return {
        "valid": not errors,
        "errors": errors,
        "resolved": resolved,
    }


def stable_keys_from_child_results(results: list[dict]) -> list[str]:
    """Collapse child results into unique path+section labels in first-hit order."""
    seen = set()
    keys = []

    for result in results:
        chunk = result.get("chunk", {})
        relative_path = chunk.get("relative_path")
        section = chunk.get("section")
        if not relative_path or not section:
            continue
        key = stable_section_key(relative_path, section)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)

    return keys


def stable_keys_from_parent_results(results: list[dict]) -> list[str]:
    keys = []
    seen = set()

    for result in results:
        parent = result.get("parent", {})
        relative_path = parent.get("relative_path")
        section = parent.get("section")
        if not relative_path or not section:
            continue
        key = stable_section_key(relative_path, section)
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)

    return keys


from rag_assist.evaluation.metrics import (
    EVAL_KS,
    NDCG_K,
    MRR_K,
    compute_metrics,
)
from rag_assist.retrieval.fusion import weighted_reciprocal_rank_fusion
from rag_assist.config import get_settings


RRF_EXPERIMENTS = [
    ("dense_only", 1.0, 0.0),
    ("rrf_0.8_0.2", 0.8, 0.2),
    ("rrf_0.7_0.3", 0.7, 0.3),
    ("rrf_0.6_0.4", 0.6, 0.4),
    ("rrf_equal", 1.0, 1.0),
]


def aggregate(stage_rows: dict[str, list[dict]]) -> dict[str, dict]:
    output = {}
    for stage, rows in stage_rows.items():
        if not rows:
            continue
        metric_names = rows[0]["metrics"].keys()
        output[stage] = {
            name: sum(row["metrics"][name] for row in rows) / len(rows)
            for name in metric_names
        }
    return output


def run_evaluation(dataset: list[dict], parents: list[dict]) -> dict:
    validation = validate_dataset(dataset, parents)
    if not validation["valid"]:
        message = "\n".join(validation["errors"])
        raise RuntimeError(
            "Evaluation dataset validation failed. Fix the labels first:\n" + message
        )

    resolved = validation["resolved"]
    chunks = load_chunks()
    parent_store = ParentStore()

    if parent_store.count() == 0:
        parent_store.close()
        raise RuntimeError("Parent store is empty. Run src/indexer.py first.")

    bm25_cache = FilteredBM25Cache(chunks)
    embedding_model = load_embedding_model()
    reranker = load_reranker()
    client = QdrantClient(url="http://localhost:6333")

    stage_rows = defaultdict(list)
    query_results = []

    try:
        for item_number, item in enumerate(resolved, start=1):
            query = item["query"]
            filters = item.get("filters", {})
            category = filters.get("category")
            doc_type = filters.get("doc_type")
            relevant_ids = item["relevant_keys"]

            print(f"\n[{item_number}/{len(resolved)}] {query}")
            print(f"  Filter: category={category}, doc_type={doc_type}")
            print(f"  Relevant sections: {len(relevant_ids)}")

            qdrant_filter = build_qdrant_filter(
                category=category,
                doc_type=doc_type,
            )
            bm25_index = bm25_cache.get(
                category=category,
                doc_type=doc_type,
            )

            dense_results = dense_search(
                client,
                embedding_model,
                query,
                DENSE_TOP_K,
                query_filter=qdrant_filter,
            )
            bm25_results = bm25_search(
                bm25_index,
                query,
                BM25_TOP_K,
            )
            dense_child_results = [
                {"chunk": point.payload} for point in dense_results
            ]

            per_query = {
                "query_index": item["query_index"],
                "query": query,
                "filters": filters,
                "relevant_sections": relevant_ids,
                "stages": {},
                "rrf_experiments": {},
            }

            ranked = {
                "dense": stable_keys_from_child_results(dense_child_results),
                "bm25": stable_keys_from_child_results(bm25_results),
            }

            # Baseline stages. Keep these comparable to the original evaluation.
            baseline_rrf = reciprocal_rank_fusion(
                dense_results,
                bm25_results,
                RRF_TOP_K,
                RRF_K,
            )
            baseline_reranked = rerank(
                reranker,
                query,
                baseline_rrf,
                RRF_TOP_K,
            )
            baseline_parent = expand_to_parents(
                baseline_reranked,
                parent_store,
                FINAL_TOP_PARENTS,
            )
            ranked.update({
                "rrf": stable_keys_from_child_results(baseline_rrf),
                "reranker": stable_keys_from_child_results(baseline_reranked),
                "parent": stable_keys_from_parent_results(baseline_parent),
            })

            for experiment_name, dense_weight, bm25_weight in RRF_EXPERIMENTS:
                fused = weighted_reciprocal_rank_fusion(
                    dense_results,
                    bm25_results,
                    RRF_TOP_K,
                    RRF_K,
                    dense_weight=dense_weight,
                    bm25_weight=bm25_weight,
                )
                rrf_keys = stable_keys_from_child_results(fused)
                rrf_metrics = compute_metrics(rrf_keys, relevant_ids)
                per_query["rrf_experiments"][experiment_name] = {
                    "dense_weight": dense_weight,
                    "bm25_weight": bm25_weight,
                    "metrics": rrf_metrics,
                }
                stage_rows[f"{experiment_name}"].append({"metrics": rrf_metrics})

            # Rerank the production-default equal-weight RRF separately above.

            for stage, retrieved in ranked.items():
                metrics = compute_metrics(retrieved, relevant_ids)
                per_query["stages"][stage] = {
                    "retrieved_sections": retrieved,
                    "metrics": metrics,
                }
                stage_rows[stage].append({"metrics": metrics})
                print(
                    f"  {stage:12s} "
                    f"R@1={metrics['recall@1']:.3f} "
                    f"R@3={metrics['recall@3']:.3f} "
                    f"R@5={metrics['recall@5']:.3f} "
                    f"MRR@{MRR_K}={metrics[f'mrr@{MRR_K}']:.3f} "
                    f"nDCG@{NDCG_K}={metrics[f'ndcg@{NDCG_K}']:.3f}"
                )

            # Print all fusion experiments after the baseline stages.
            for experiment_name, payload in per_query["rrf_experiments"].items():
                metrics = payload["metrics"]
                print(
                    f"  {experiment_name:12s} "
                    f"R@1={metrics['recall@1']:.3f} "
                    f"R@3={metrics['recall@3']:.3f} "
                    f"R@5={metrics['recall@5']:.3f} "
                    f"MRR@{MRR_K}={metrics[f'mrr@{MRR_K}']:.3f} "
                    f"nDCG@{NDCG_K}={metrics[f'ndcg@{NDCG_K}']:.3f}"
                )

            query_results.append(per_query)
    finally:
        parent_store.close()

    return {
        "dataset_file": str(DATASET_FILE),
        "relevance_unit": "relative_path + section",
        "num_queries": len(resolved),
        "ks": list(EVAL_KS),
        "mrr_cutoff": MRR_K,
        "ndcg_cutoff": NDCG_K,
        "aggregate": aggregate(stage_rows),
        "queries": query_results,
    }


def write_outputs(result: dict) -> None:
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    headers = [
        "query_index",
        "query",
        "stage",
        "recall@1",
        "recall@3",
        "recall@5",
        f"mrr@{MRR_K}",
        f"ndcg@{NDCG_K}",
    ]
    with RESULT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for query in result["queries"]:
            for stage, payload in query["stages"].items():
                row = {
                    "query_index": query["query_index"],
                    "query": query["query"],
                    "stage": stage,
                }
                row.update(payload["metrics"])
                writer.writerow(row)


def print_summary(result: dict) -> None:
    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Queries: {result['num_queries']}")
    print(
        f"Metrics: Recall@{list(EVAL_KS)}, "
        f"MRR@{MRR_K}, nDCG@{NDCG_K}"
    )
    print()

    header = (
        f"{'Stage':<10} "
        f"{'R@1':>8} {'R@3':>8} {'R@5':>8} "
        f"{'MRR':>8} {'nDCG':>8}"
    )
    print(header)
    print("-" * len(header))

    for stage, metrics in result["aggregate"].items():
        print(
            f"{stage:<10} "
            f"{metrics['recall@1']:>8.3f} "
            f"{metrics['recall@3']:>8.3f} "
            f"{metrics['recall@5']:>8.3f} "
            f"{metrics[f'mrr@{MRR_K}']:>8.3f} "
            f"{metrics[f'ndcg@{NDCG_K}']:>8.3f}"
        )

    print("\nSaved:")
    print(f"  JSON: {RESULT_JSON}")
    print(f"  CSV : {RESULT_CSV}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Dense/BM25/RRF/Reranker/Parent retrieval."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to evaluation dataset JSONL",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate that all ground-truth labels resolve to parents.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")
    if not PARENTS_FILE.exists():
        raise FileNotFoundError(
            f"Parent file not found: {PARENTS_FILE}. Run src/chunker.py first."
        )

    dataset = load_jsonl(dataset_path)
    parents = load_jsonl(PARENTS_FILE)

    validation = validate_dataset(dataset, parents)
    print(
        f"Dataset validation: {'PASS' if validation['valid'] else 'FAIL'} "
        f"({len(validation['resolved'])}/{len(dataset)} queries resolvable)"
    )
    if validation["errors"]:
        for error in validation["errors"]:
            print(f"  ERROR: {error}")

    if not validation["valid"]:
        raise SystemExit(1)

    if args.validate_only:
        return

    result = run_evaluation(dataset, parents)
    write_outputs(result)
    print_summary(result)


if __name__ == "__main__":
    main()

