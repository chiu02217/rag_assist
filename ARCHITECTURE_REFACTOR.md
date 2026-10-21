# rag_assist architecture refactor

## Why this refactor

The current project has all Python modules in one `src/` directory. As the RAG grows
into multi-source ingestion, retrieval, generation, API, and agent tooling, a flat
directory becomes difficult to navigate and creates implicit coupling.

This refactor is intentionally conservative: it reorganizes the existing modules
by responsibility first. It does **not** change retrieval behavior yet.

## Target structure

```text
rag_assist/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  ├─ db/
│  └─ eval/
│
├─ src/
│  └─ rag_assist/
│     ├─ ingestion/
│     │  └─ kubernetes_docs.py
│     ├─ chunking/
│     │  └─ parent_child.py
│     ├─ storage/
│     │  ├─ parent_store.py
│     │  └─ qdrant_indexer.py
│     ├─ retrieval/
│     │  ├─ dense.py
│     │  ├─ bm25.py
│     │  ├─ fusion.py
│     │  └─ hybrid.py
│     ├─ evaluation/
│     │  ├─ metrics.py
│     │  └─ runner.py
│     ├─ diagnostics/
│     │  ├─ check_doc.py
│     │  └─ inspect_doc.py
│     └─ cli/
│
├─ scripts/
│  ├─ refactor_project.ps1
│  ├─ reset_qdrant.py
│  ├─ run_evaluation.ps1
│  └─ run_weighted_rrf.ps1
│
├─ tests/
│  ├─ unit/
│  │  ├─ test_metrics.py
│  │  └─ test_rrf.py
│  └─ integration/
│     ├─ test_embedding.py
│     ├─ test_parent_store.py
│     ├─ test_qdrant.py
│     ├─ test_qdrant_payload.py
│     └─ test_reranker.py
│
├─ evaluations/
│  └─ retrieval_queries.py
├─ notebooks/
├─ requirements.txt
├─ pyproject.toml
└─ README.md
```

## Migration order

1. Run the refactor in dry-run mode.
2. Apply the copy migration.
3. Install the package in editable mode.
4. Run unit tests.
5. Run evaluation validation.
6. Run one real retrieval query.
7. Only then delete legacy files.

## Important design choice

Do not split `hybrid.py` into many tiny files in the same migration. First stabilize
the directory boundaries. When the next feature (GitHub Issues/PR ingestion) is added,
we can then split the retrieval implementation into `dense.py`, `sparse.py`,
`fusion.py`, `reranker.py`, and `pipeline.py` based on actual interfaces.

## Future multi-source layout

The next ingestion feature should become:

```text
src/rag_assist/ingestion/
├─ kubernetes_docs.py
├─ github_issues.py
├─ github_prs.py
├─ release_notes.py
├─ normalizer.py
└─ pipeline.py
```

The normalized record should carry a `source_type`, `source_id`, `source_url`,
and source-specific metadata so all sources can share the same chunk/index/retrieval
pipeline.
