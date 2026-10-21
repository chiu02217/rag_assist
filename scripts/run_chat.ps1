param(
    [ValidateSet("documentation", "github_issue", "all")]
    [string]$SourceType = "all",
    [string]$Model = "qwen3.5:2b",
    [string]$BaseUrl = "http://localhost:11434/v1"
)

$ErrorActionPreference = "Stop"

python -m rag_assist.generation.chat `
    --source-type $SourceType `
    --model $Model `
    --base-url $BaseUrl
