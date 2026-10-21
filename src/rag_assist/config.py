"""Central runtime configuration for rag_assist."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def find_project_root(start: Path | None = None) -> Path:
    explicit = os.getenv("RAG_PROJECT_ROOT")
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(
                f"RAG_PROJECT_ROOT does not exist: {root}"
            )
        return root

    current = (start or Path.cwd()).resolve()

    for candidate in (current, *current.parents):
        if (
            (candidate / "pyproject.toml").exists()
            and (candidate / "src").is_dir()
            and (candidate / "data").is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not determine rag_assist project root. "
        "Run from the repository or set RAG_PROJECT_ROOT."
    )


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    project_root: Path

    # Data directories
    raw_dir: Path
    processed_dir: Path
    db_dir: Path
    eval_dir: Path

    # Existing documentation corpus
    kubernetes_docs_dir: Path
    documents_file: Path

    # Multi-source corpus
    github_issues_file: Path
    documents_all_file: Path

    # Parent / child outputs remain on the existing paths
    parents_file: Path
    chunks_file: Path

    # Persistent storage / evaluation
    parent_db_file: Path
    eval_dataset_file: Path
    eval_results_json: Path
    eval_results_csv: Path

    # Qdrant
    qdrant_url: str
    qdrant_collection: str

    # Models
    embedding_model: str
    reranker_model: str
    embedding_device: str
    reranker_device: str
    reranker_batch_size: int
    reranker_max_length: int

    # Chunking / indexing
    parent_max_chars: int
    parent_overlap_chars: int
    child_chunk_size: int
    child_overlap: int
    embed_batch_size: int
    max_chunks: int | None
    vector_size: int

    # Retrieval
    dense_top_k: int
    bm25_top_k: int
    rrf_top_k: int
    rrf_k: int
    rerank_top_k: int
    final_top_k: int
    final_top_parents: int

    # Default metadata scope
    default_category: str
    default_doc_type: str
    documentation_source_name: str
    default_source_type: str

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        root = (project_root or find_project_root()).resolve()

        raw_dir = root / "data" / "raw"
        processed_dir = root / "data" / "processed"
        db_dir = root / "data" / "db"
        eval_dir = root / "data" / "eval"

        max_chunks_raw = os.getenv("RAG_MAX_CHUNKS", "").strip()
        max_chunks = int(max_chunks_raw) if max_chunks_raw else None

        return cls(
            project_root=root,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            db_dir=db_dir,
            eval_dir=eval_dir,
            kubernetes_docs_dir=_resolve_path(
                _env_str(
                    "RAG_KUBERNETES_DOCS_DIR",
                    str(
                        raw_dir
                        / "kubernetes-website"
                        / "content"
                        / "en"
                        / "docs"
                    ),
                ),
                root,
            ),
            documents_file=_resolve_path(
                _env_str(
                    "RAG_DOCUMENTS_FILE",
                    str(processed_dir / "documents.jsonl"),
                ),
                root,
            ),
            github_issues_file=_resolve_path(
                _env_str(
                    "RAG_GITHUB_ISSUES_FILE",
                    str(processed_dir / "github_issues.jsonl"),
                ),
                root,
            ),
            documents_all_file=_resolve_path(
                _env_str(
                    "RAG_DOCUMENTS_ALL_FILE",
                    str(processed_dir / "documents_all.jsonl"),
                ),
                root,
            ),
            parents_file=_resolve_path(
                _env_str(
                    "RAG_PARENTS_FILE",
                    str(processed_dir / "parents.jsonl"),
                ),
                root,
            ),
            chunks_file=_resolve_path(
                _env_str(
                    "RAG_CHUNKS_FILE",
                    str(processed_dir / "chunks_structured.jsonl"),
                ),
                root,
            ),
            parent_db_file=_resolve_path(
                _env_str(
                    "RAG_PARENT_DB_FILE",
                    str(db_dir / "rag_metadata.sqlite3"),
                ),
                root,
            ),
            eval_dataset_file=_resolve_path(
                _env_str(
                    "RAG_EVAL_DATASET",
                    str(eval_dir / "golden_dataset.jsonl"),
                ),
                root,
            ),
            eval_results_json=_resolve_path(
                _env_str(
                    "RAG_EVAL_RESULTS_JSON",
                    str(eval_dir / "evaluation_results.json"),
                ),
                root,
            ),
            eval_results_csv=_resolve_path(
                _env_str(
                    "RAG_EVAL_RESULTS_CSV",
                    str(eval_dir / "evaluation_results.csv"),
                ),
                root,
            ),
            qdrant_url=_env_str(
                "RAG_QDRANT_URL",
                "http://localhost:6333",
            ),
            qdrant_collection=_env_str(
                "RAG_QDRANT_COLLECTION",
                "kubernetes_docs",
            ),
            embedding_model=_env_str(
                "RAG_EMBEDDING_MODEL",
                "BAAI/bge-m3",
            ),
            reranker_model=_env_str(
                "RAG_RERANKER_MODEL",
                "BAAI/bge-reranker-v2-m3",
            ),
            embedding_device=_env_str(
                "RAG_EMBEDDING_DEVICE",
                "auto",
            ),
            reranker_device=_env_str(
                "RAG_RERANKER_DEVICE",
                "auto",
            ),
            reranker_batch_size=_env_int(
                "RAG_RERANKER_BATCH_SIZE",
                4,
            ),
            reranker_max_length=_env_int(
                "RAG_RERANKER_MAX_LENGTH",
                512,
            ),
            parent_max_chars=_env_int(
                "RAG_PARENT_MAX_CHARS",
                4000,
            ),
            parent_overlap_chars=_env_int(
                "RAG_PARENT_OVERLAP_CHARS",
                400,
            ),
            child_chunk_size=_env_int(
                "RAG_CHILD_CHUNK_SIZE",
                1200,
            ),
            child_overlap=_env_int(
                "RAG_CHILD_OVERLAP",
                200,
            ),
            embed_batch_size=_env_int(
                "RAG_EMBED_BATCH_SIZE",
                8,
            ),
            max_chunks=max_chunks,
            vector_size=_env_int(
                "RAG_VECTOR_SIZE",
                1024,
            ),
            dense_top_k=_env_int(
                "RAG_DENSE_TOP_K",
                30,
            ),
            bm25_top_k=_env_int(
                "RAG_BM25_TOP_K",
                30,
            ),
            rrf_top_k=_env_int(
                "RAG_RRF_TOP_K",
                20,
            ),
            rrf_k=_env_int(
                "RAG_RRF_K",
                60,
            ),
            rerank_top_k=_env_int(
                "RAG_RERANK_TOP_K",
                20,
            ),
            final_top_k=_env_int(
                "RAG_FINAL_TOP_K",
                5,
            ),
            final_top_parents=_env_int(
                "RAG_FINAL_TOP_PARENTS",
                5,
            ),
            default_category=_env_str(
                "RAG_DEFAULT_CATEGORY",
                "concepts",
            ),
            default_doc_type=_env_str(
                "RAG_DEFAULT_DOC_TYPE",
                "documentation",
            ),
            documentation_source_name=_env_str(
                "RAG_DOCUMENTATION_SOURCE",
                "Kubernetes Documentation",
            ),
            default_source_type=_env_str(
                "RAG_DEFAULT_SOURCE_TYPE",
                "documentation",
            ),
        )

    def ensure_runtime_dirs(self) -> None:
        for directory in (
            self.processed_dir,
            self.db_dir,
            self.eval_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def get_settings(project_root: Path | None = None) -> Settings:
    settings = Settings.from_env(project_root)
    settings.ensure_runtime_dirs()
    return settings
