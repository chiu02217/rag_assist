from rag_assist.config import Settings


def _make_project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='test-rag-assist'\n",
        encoding="utf-8",
    )
    return tmp_path


def test_default_paths(tmp_path):
    root = _make_project(tmp_path)
    settings = Settings.from_env(root)

    assert settings.project_root == root
    assert settings.documents_file == root / "data" / "processed" / "documents.jsonl"
    assert settings.parents_file == root / "data" / "processed" / "parents.jsonl"
    assert settings.chunks_file == root / "data" / "processed" / "chunks_structured.jsonl"
    assert settings.parent_db_file == root / "data" / "db" / "rag_metadata.sqlite3"
    assert settings.eval_dataset_file == root / "data" / "eval" / "golden_dataset.jsonl"


def test_environment_override(monkeypatch, tmp_path):
    root = _make_project(tmp_path)

    monkeypatch.setenv("RAG_QDRANT_URL", "http://example:6333")
    monkeypatch.setenv("RAG_QDRANT_COLLECTION", "demo_docs")
    monkeypatch.setenv("RAG_DENSE_TOP_K", "30")
    monkeypatch.setenv("RAG_EMBED_BATCH_SIZE", "4")
    monkeypatch.setenv("RAG_DEFAULT_CATEGORY", "tasks")

    settings = Settings.from_env(root)

    assert settings.qdrant_url == "http://example:6333"
    assert settings.qdrant_collection == "demo_docs"
    assert settings.dense_top_k == 30
    assert settings.embed_batch_size == 4
    assert settings.default_category == "tasks"


def test_relative_custom_path_is_resolved_from_project_root(monkeypatch, tmp_path):
    root = _make_project(tmp_path)
    monkeypatch.setenv("RAG_DOCUMENTS_FILE", "custom/documents.jsonl")

    settings = Settings.from_env(root)

    assert settings.documents_file == root / "custom" / "documents.jsonl"


def test_runtime_dirs(tmp_path):
    root = _make_project(tmp_path)
    settings = Settings.from_env(root)
    settings.ensure_runtime_dirs()

    assert (root / "data" / "processed").is_dir()
    assert (root / "data" / "db").is_dir()
    assert (root / "data" / "eval").is_dir()
