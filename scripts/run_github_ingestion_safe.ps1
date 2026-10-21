param(
    [int]$MaxIssues = 500
)

$ErrorActionPreference = "Stop"

if (-not $env:GITHUB_TOKEN) {
    Write-Warning "GITHUB_TOKEN is not set. Public GitHub API access has stricter rate limits."
}

python -m rag_assist.ingestion.github_issues_paged --max-issues $MaxIssues
python -m rag_assist.ingestion.build_corpus

Write-Host ""
Write-Host "If the corpus looks correct, continue with:"
Write-Host "  python -m rag_assist.chunking.parent_child"
Write-Host "  python -m rag_assist.storage.qdrant_indexer"
