param(
    [int]$MaxIssues = 200
)

$ErrorActionPreference = "Stop"

if (-not $env:GITHUB_TOKEN) {
    Write-Warning "GITHUB_TOKEN is not set. Public unauthenticated API access may have stricter rate limits."
}

python -m rag_assist.ingestion.github_issues --max-issues $MaxIssues

python -m rag_assist.ingestion.build_corpus
