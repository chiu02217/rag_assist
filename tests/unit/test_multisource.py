import json

from rag_assist.chunking.parent_child import create_parent_child_data
from rag_assist.ingestion.github_issues import normalize_issue


def test_multi_source_fields_survive_chunking():
    issue = normalize_issue(
        {
            "number": 123,
            "title": "Example issue",
            "body": "This is an issue about kubelet.",
            "state": "open",
            "html_url": "https://github.com/kubernetes/kubernetes/issues/123",
            "labels": [{"name": "kind/bug"}],
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-02T00:00:00Z",
        },
        "kubernetes/kubernetes",
    )

    parents, chunks = create_parent_child_data(
        [issue.to_dict()]
    )

    assert parents
    assert chunks

    assert parents[0].source_type == "github_issue"
    assert parents[0].source_id == "kubernetes/kubernetes#123"
    assert parents[0].source_url.endswith("/123")

    assert chunks[0].source_type == "github_issue"
    assert chunks[0].source_id == "kubernetes/kubernetes#123"
    assert chunks[0].metadata["github_issue_number"] == 123
