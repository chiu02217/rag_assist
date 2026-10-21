import json

from rag_assist.ingestion.github_issues import (
    deterministic_document_id,
    normalize_issue,
)
from rag_assist.ingestion.build_corpus import merge_documents


def test_issue_id_is_deterministic():
    assert deterministic_document_id("kubernetes/kubernetes", 123) == (
        deterministic_document_id("kubernetes/kubernetes", 123)
    )
    assert deterministic_document_id("kubernetes/kubernetes", 123) != (
        deterministic_document_id("kubernetes/kubernetes", 124)
    )


def test_normalize_issue_has_source_contract():
    issue = {
        "number": 137685,
        "title": "MemoryQoS issue",
        "body": "Example body.",
        "state": "closed",
        "html_url": "https://github.com/kubernetes/kubernetes/issues/137685",
        "labels": [{"name": "kind/bug"}],
        "created_at": "2026-03-12T23:02:26Z",
        "updated_at": "2026-06-01T17:36:52Z",
    }

    record = normalize_issue(issue, "kubernetes/kubernetes")
    payload = record.to_dict()

    assert payload["source_type"] == "github_issue"
    assert payload["source_id"] == "kubernetes/kubernetes#137685"
    assert payload["source_url"].endswith("/137685")
    assert payload["github_issue_number"] == 137685
    assert "MemoryQoS issue" in payload["content"]
    assert "kind/bug" in payload["content"]


def test_merge_documents_deduplicates(tmp_path):
    docs = tmp_path / "docs.jsonl"
    issues = tmp_path / "issues.jsonl"
    output = tmp_path / "all.jsonl"

    docs.write_text(
        json.dumps(
            {
                "document_id": "doc-1",
                "relative_path": "concepts/example.md",
                "title": "Example",
                "content": "doc",
                "category": "concepts",
                "doc_type": "documentation",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    issue = {
        "document_id": "issue-1",
        "source_type": "github_issue",
        "source_id": "kubernetes/kubernetes#1",
        "source_url": "https://github.com/kubernetes/kubernetes/issues/1",
        "title": "Issue",
        "content": "issue",
    }

    issues.write_text(json.dumps(issue) + "\n", encoding="utf-8")

    docs_count, issue_count, merged_count = merge_documents(
        docs, issues, output
    )

    assert (docs_count, issue_count, merged_count) == (1, 1, 2)

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]

    documentation = records[0]
    assert documentation["source_type"] == "documentation"
    assert documentation["source_url"].startswith("https://kubernetes.io/docs/")
