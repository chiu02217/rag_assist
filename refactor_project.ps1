param(
    [switch]$Apply,
    [switch]$DeleteOld
)

$ErrorActionPreference = "Stop"

$root = Get-Location
$map = @{
    "src/doc_loader.py" = "src/rag_assist/ingestion/kubernetes_docs.py"
    "src/chunker.py" = "src/rag_assist/chunking/parent_child.py"
    "src/parent_store.py" = "src/rag_assist/storage/parent_store.py"
    "src/indexer.py" = "src/rag_assist/storage/qdrant_indexer.py"
    "src/retriever.py" = "src/rag_assist/retrieval/dense.py"
    "src/bm25_retriever.py" = "src/rag_assist/retrieval/bm25.py"
    "src/rrf_utils.py" = "src/rag_assist/retrieval/fusion.py"
    "src/hybrid_retriever.py" = "src/rag_assist/retrieval/hybrid.py"
    "src/retrieval_metrics.py" = "src/rag_assist/evaluation/metrics.py"
    "src/evaluate_retrieval.py" = "src/rag_assist/evaluation/runner.py"
    "src/check_doc.py" = "src/rag_assist/diagnostics/check_doc.py"
    "src/inspect_doc.py" = "src/rag_assist/diagnostics/inspect_doc.py"
    "tests/test_metrics.py" = "tests/unit/test_metrics.py"
    "tests/test_rrf.py" = "tests/unit/test_rrf.py"
    "tests/test_embedding.py" = "tests/integration/test_embedding.py"
    "tests/test_payload.py" = "tests/integration/test_qdrant_payload.py"
    "tests/test_qdrant.py" = "tests/integration/test_qdrant.py"
    "tests/test_parent_store.py" = "tests/integration/test_parent_store.py"
    "tests/test_reranker.py" = "tests/integration/test_reranker.py"
    "tests/reset_qdrant.py" = "scripts/reset_qdrant.py"
}

if (-not $Apply) {
    Write-Host "DRY RUN only."
    Write-Host "Review the migration plan first. To apply:"
    Write-Host "  .\scripts\refactor_project.ps1 -Apply"
    Write-Host "After tests pass, optionally remove legacy files:"
    Write-Host "  .\scripts\refactor_project.ps1 -Apply -DeleteOld"
    exit 0
}

foreach ($entry in $map.GetEnumerator()) {
    $src = Join-Path $root $entry.Key
    $dst = Join-Path $root $entry.Value

    if (Test-Path $src) {
        New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
        Copy-Item $src $dst -Force
        Write-Host "Copied $($entry.Key) -> $($entry.Value)"
    }
}

# Add package markers.
$packages = @(
    "src/rag_assist",
    "src/rag_assist/ingestion",
    "src/rag_assist/chunking",
    "src/rag_assist/storage",
    "src/rag_assist/retrieval",
    "src/rag_assist/evaluation",
    "src/rag_assist/diagnostics",
    "src/rag_assist/cli"
)
foreach ($pkg in $packages) {
    New-Item -ItemType Directory -Force -Path (Join-Path $root $pkg) | Out-Null
    $init = Join-Path $root "$pkg/__init__.py"
    if (-not (Test-Path $init)) {
        '"""rag_assist package."""' | Set-Content $init -Encoding UTF8
    }
}

# Targeted import rewrites for the current evaluation runner/tests.
$replacements = @{
    "src/rag_assist/evaluation/runner.py" = @(
        @('from hybrid_retriever import (', 'from rag_assist.retrieval.hybrid import ('),
        @('from parent_store import ParentStore', 'from rag_assist.storage.parent_store import ParentStore'),
        @('from retrieval_metrics import (', 'from rag_assist.evaluation.metrics import ('),
        @('from rrf_utils import weighted_reciprocal_rank_fusion', 'from rag_assist.retrieval.fusion import weighted_reciprocal_rank_fusion')
    )
    "tests/unit/test_metrics.py" = @(
        @('from retrieval_metrics import', 'from rag_assist.evaluation.metrics import')
    )
    "tests/unit/test_rrf.py" = @(
        @('from rrf_utils import', 'from rag_assist.retrieval.fusion import')
    )
}

foreach ($entry in $replacements.GetEnumerator()) {
    $path = Join-Path $root $entry.Key
    if (-not (Test-Path $path)) { continue }

    $text = Get-Content $path -Raw
    foreach ($pair in $entry.Value) {
        $text = $text.Replace($pair[0], $pair[1])
    }
    Set-Content $path $text -Encoding UTF8
}

Write-Host ""
Write-Host "Next verification commands:"
Write-Host "  pip install -e ."
Write-Host "  python -m pytest tests/unit"
Write-Host "  python -m rag_assist.evaluation.runner --validate-only"
Write-Host ""
Write-Host "NOTE: Current retrieval modules remain internally cohesive. After verification,"
Write-Host "we can split hybrid.py into dense.py / sparse.py / reranker.py without changing behavior."

if ($DeleteOld) {
    foreach ($entry in $map.GetEnumerator()) {
        $src = Join-Path $root $entry.Key
        if (Test-Path $src) {
            Remove-Item $src -Force
            Write-Host "Removed legacy $($entry.Key)"
        }
    }
}
