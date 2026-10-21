from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]

TARGET_FILES = [
    "src/rag_assist/ingestion/kubernetes_docs.py",
    "src/rag_assist/chunking/parent_child.py",
    "src/rag_assist/storage/parent_store.py",
    "src/rag_assist/storage/qdrant_indexer.py",
    "src/rag_assist/retrieval/dense.py",
    "src/rag_assist/retrieval/bm25.py",
    "src/rag_assist/retrieval/hybrid.py",
    "src/rag_assist/evaluation/runner.py",
]


def add_import(content: str) -> str:
    if "from rag_assist.config import get_settings" in content:
        return content

    lines = content.splitlines()
    import_lines = [
        i for i, line in enumerate(lines)
        if line.startswith("import ") or line.startswith("from ")
    ]

    if not import_lines:
        return (
            "from rag_assist.config import get_settings\n\n"
            + content
        )

    last_import = max(import_lines)
    lines.insert(last_import + 1, "from rag_assist.config import get_settings")
    return "\n".join(lines) + ("\n" if content.endswith("\n") else "")


def apply_replacements(path: Path, content: str) -> tuple[str, int]:
    replacements: list[tuple[str, str]] = []

    name = path.as_posix()

    if name.endswith("ingestion/kubernetes_docs.py"):
        replacements = [
            (
                'DOCS_DIR = Path("data/raw/kubernetes-website/content/en/docs")',
                "settings = get_settings()\nDOCS_DIR = settings.kubernetes_docs_dir",
            ),
            (
                'OUTPUT_FILE = Path("data/processed/documents.jsonl")',
                "OUTPUT_FILE = settings.documents_file",
            ),
            (
                'source="Kubernetes Documentation"',
                'source=settings.documentation_source_name',
            ),
        ]

    elif name.endswith("chunking/parent_child.py"):
        replacements = [
            (
                'INPUT_FILE = Path("data/processed/documents.jsonl")',
                "settings = get_settings()\nINPUT_FILE = settings.documents_file",
            ),
            (
                'PARENTS_OUTPUT = Path("data/processed/parents.jsonl")',
                "PARENTS_OUTPUT = settings.parents_file",
            ),
            (
                'CHUNKS_OUTPUT = Path("data/processed/chunks_structured.jsonl")',
                "CHUNKS_OUTPUT = settings.chunks_file",
            ),
            ("PARENT_MAX_CHARS = 4000", "PARENT_MAX_CHARS = settings.parent_max_chars"),
            ("PARENT_OVERLAP_CHARS = 400", "PARENT_OVERLAP_CHARS = settings.parent_overlap_chars"),
            ("CHILD_CHUNK_SIZE = 1200", "CHILD_CHUNK_SIZE = settings.child_chunk_size"),
            ("CHILD_OVERLAP = 200", "CHILD_OVERLAP = settings.child_overlap"),
        ]

    elif name.endswith("storage/parent_store.py"):
        replacements = [
            (
                'DB_FILE = Path("data/db/rag_metadata.sqlite3")',
                "settings = get_settings()\nDB_FILE = settings.parent_db_file",
            ),
            (
                'PARENTS_FILE = Path("data/processed/parents.jsonl")',
                "PARENTS_FILE = settings.parents_file",
            ),
        ]

    elif name.endswith("storage/qdrant_indexer.py"):
        replacements = [
            (
                'INPUT_FILE = Path("data/processed/chunks_structured.jsonl")',
                "settings = get_settings()\nINPUT_FILE = settings.chunks_file",
            ),
            (
                'PARENTS_FILE = Path("data/processed/parents.jsonl")',
                "PARENTS_FILE = settings.parents_file",
            ),
            ('QDRANT_URL = "http://localhost:6333"', "QDRANT_URL = settings.qdrant_url"),
            ('COLLECTION_NAME = "kubernetes_docs"', "COLLECTION_NAME = settings.qdrant_collection"),
            ('MODEL_NAME = "BAAI/bge-m3"', "MODEL_NAME = settings.embedding_model"),
            ("VECTOR_SIZE = 1024", "VECTOR_SIZE = settings.vector_size"),
            ("EMBED_BATCH_SIZE = 8", "EMBED_BATCH_SIZE = settings.embed_batch_size"),
            ("MAX_CHUNKS = None", "MAX_CHUNKS = settings.max_chunks"),
            (
                "from parent_store import ParentStore",
                "from rag_assist.storage.parent_store import ParentStore",
            ),
        ]

    elif name.endswith("retrieval/dense.py"):
        replacements = [
            (
                'INPUT_FILE = Path("data/processed/chunks_structured.jsonl")',
                "settings = get_settings()\nINPUT_FILE = settings.chunks_file",
            ),
            ('QDRANT_URL = "http://localhost:6333"', "QDRANT_URL = settings.qdrant_url"),
            ('COLLECTION_NAME = "kubernetes_docs"', "COLLECTION_NAME = settings.qdrant_collection"),
            ('MODEL_NAME = "BAAI/bge-m3"', "MODEL_NAME = settings.embedding_model"),
            ("DENSE_TOP_K = 20", "DENSE_TOP_K = settings.dense_top_k"),
        ]

    elif name.endswith("retrieval/bm25.py"):
        replacements = [
            (
                'INPUT_FILE = Path("data/processed/chunks_structured.jsonl")',
                "settings = get_settings()\nINPUT_FILE = settings.chunks_file",
            ),
            ("BM25_TOP_K = 20", "BM25_TOP_K = settings.bm25_top_k"),
        ]

    elif name.endswith("retrieval/hybrid.py"):
        replacements = [
            (
                'INPUT_FILE = Path("data/processed/chunks_structured.jsonl")',
                "settings = get_settings()\nINPUT_FILE = settings.chunks_file",
            ),
            ('QDRANT_URL = "http://localhost:6333"', "QDRANT_URL = settings.qdrant_url"),
            ('COLLECTION_NAME = "kubernetes_docs"', "COLLECTION_NAME = settings.qdrant_collection"),
            ('MODEL_NAME = "BAAI/bge-m3"', "MODEL_NAME = settings.embedding_model"),
            ('RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"', "RERANKER_MODEL_NAME = settings.reranker_model"),
            ("DENSE_TOP_K = 20", "DENSE_TOP_K = settings.dense_top_k"),
            ("BM25_TOP_K = 20", "BM25_TOP_K = settings.bm25_top_k"),
            ("RRF_K = 60", "RRF_K = settings.rrf_k"),
            ("RRF_TOP_K = 20", "RRF_TOP_K = settings.rrf_top_k"),
            ("RERANK_TOP_K = 20", "RERANK_TOP_K = settings.rerank_top_k"),
            ("FINAL_TOP_K = 5", "FINAL_TOP_K = settings.final_top_k"),
            ("FINAL_TOP_PARENTS = 5", "FINAL_TOP_PARENTS = settings.final_top_parents"),
            ('category = "concepts"', "category = settings.default_category"),
            ('doc_type = "documentation"', "doc_type = settings.default_doc_type"),
            (
                'device = "cuda" if torch.cuda.is_available() else "cpu"',
                'device = settings.embedding_device\n    if device == "auto":\n        device = "cuda" if torch.cuda.is_available() else "cpu"',
            ),
            (
                'device = ("cuda" if torch.cuda.is_available() else "cpu")',
                'device = settings.reranker_device\n    if device == "auto":\n        device = "cuda" if torch.cuda.is_available() else "cpu"',
            ),
        ]

    elif name.endswith("evaluation/runner.py"):
        replacements = [
            (
                'PROJECT_ROOT = SRC_DIR.parent',
                "PROJECT_ROOT = get_settings().project_root",
            ),
            (
                'DATASET_FILE = PROJECT_ROOT / "data/eval/golden_dataset.jsonl"',
                "DATASET_FILE = get_settings().eval_dataset_file",
            ),
            (
                'RESULT_JSON = PROJECT_ROOT / "data/eval/evaluation_results.json"',
                "RESULT_JSON = get_settings().eval_results_json",
            ),
            (
                'RESULT_CSV = PROJECT_ROOT / "data/eval/evaluation_results.csv"',
                "RESULT_CSV = get_settings().eval_results_csv",
            ),
            (
                'PARENTS_FILE = PROJECT_ROOT / "data/processed/parents.jsonl"',
                "PARENTS_FILE = get_settings().parents_file",
            ),
            (
                'DATASET_FILE = PROJECT_ROOT / "data/eval/golden_dataset.jsonl"',
                "DATASET_FILE = get_settings().eval_dataset_file",
            ),
        ]

    count = 0
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new, 1)
            count += 1

    return content, count


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = ROOT / "backups" / f"config_refactor_{stamp}"

    changed = 0
    missing = []

    print(f"Project root: {ROOT}")

    for relative in TARGET_FILES:
        path = ROOT / relative
        if not path.exists():
            missing.append(relative)
            continue

        original = path.read_text(encoding="utf-8")
        updated = add_import(original)
        updated, replacement_count = apply_replacements(path, updated)

        if updated != original:
            backup_path = backup_root / relative
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_path)
            path.write_text(updated, encoding="utf-8")
            changed += 1
            print(f"[UPDATED] {relative} ({replacement_count} config replacements)")
        else:
            print(f"[OK]      {relative}")

    if missing:
        print("\n[NOT FOUND] These files were skipped:")
        for relative in missing:
            print(f"  {relative}")

    print(f"\nChanged files: {changed}")
    if changed:
        print(f"Backup: {backup_root}")

    print("\nVerification:")
    print("  python -m pytest tests/unit -q")
    print("  python -m rag_assist.evaluation.runner --validate-only")
    print("  python -m rag_assist.retrieval.hybrid")

    if missing:
        print("\nSome files were not found. Review the output before continuing.")


if __name__ == "__main__":
    main()
