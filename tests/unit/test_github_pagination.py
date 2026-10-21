from pathlib import Path

from rag_assist.ingestion.github_issues_paged import ingest_recent_issues


class FakeClient:
    def __init__(self):
        self.calls = []

    def fetch_page(self, page, per_page, state):
        self.calls.append((page, per_page, state))
        if page == 1:
            return [
                {
                    "number": 1,
                    "title": "Issue one",
                    "body": "body",
                    "state": "open",
                    "html_url": "https://github.com/kubernetes/kubernetes/issues/1",
                    "labels": [],
                }
            ]
        return []


def test_ingest_respects_limit(monkeypatch, tmp_path):
    fake = FakeClient()

    monkeypatch.setattr(
        "rag_assist.ingestion.github_issues_paged.GitHubIssuesClient",
        lambda **kwargs: fake,
    )

    count = ingest_recent_issues(
        repo="kubernetes/kubernetes",
        api_url="https://api.github.com",
        token=None,
        max_issues=1,
        state="all",
        raw_output=tmp_path / "raw.jsonl",
        normalized_output=tmp_path / "normalized.jsonl",
    )

    assert count == 1
    assert fake.calls == [(1, 1, "all")]
