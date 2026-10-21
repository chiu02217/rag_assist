param(
    [int]$MaxIssues = 500
)

$ErrorActionPreference = "Stop"

Write-Host "Step 1/4 - ingest GitHub Issues"
python -m rag_assist.ingestion.github_issues --max-issues $MaxIssues

Write-Host "Step 2/4 - build unified corpus"
python -m rag_assist.ingestion.build_corpus

Write-Host "Step 3/4 - chunk unified corpus"
python -m rag_assist.chunking.parent_child

Write-Host "Step 4/4 - index unified corpus"
python -m rag_assist.storage.qdrant_indexer
