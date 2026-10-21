"""GitHub Issues ingestion for Kubernetes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import requests

from rag_assist.config import get_settings
from rag_assist.ingestion.models import DocumentRecord


DEFAULT_REPO = "kubernetes/kubernetes"
DEFAULT_API_URL = "https://api.github.com"


def deterministic_document_id(repo: str, issue_number: int) -> str:
    raw = f"github_issue:{repo}#{issue_number}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


class GitHubIssuesClient:
    """Small GitHub REST client with pagination and explicit rate-limit errors."""

    def __init__(
        self,
        repo: str,
        api_url: str = DEFAULT_API_URL,
        token: str | None = None,
        timeout: int = 30,
    ) -> None:
        self.repo = repo
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "rag-assist",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        self.session.headers.update(headers)

    def fetch_page(
        self,
        page: int,
        per_page: int = 100,
        state: str = "all",
        sort: str = "updated",
        direction: str = "desc",
    ) -> list[dict[str, Any]]:
        url = f"{self.api_url}/repos/{self.repo}/issues"
        response = self.session.get(
            url,
            params={
                "state": state,
                "sort": sort,
                "direction": direction,
                "per_page": per_page,
                "page": page,
            },
            timeout=self.timeout,
        )

        if response.status_code in {403, 429}:
            remaining = response.headers.get("X-RateLimit-Remaining", "unknown")
            reset = response.headers.get("X-RateLimit-Reset", "unknown")
            raise RuntimeError(
                "GitHub API rate limit or access restriction. "
                f"remaining={remaining}, reset={reset}"
            )

        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, list):
            raise RuntimeError("Unexpected GitHub Issues API response format.")

        return payload

    def iter_issues(
        self,
        max_issues: int | None = None,
        state: str = "all",
    ) -> Iterator[dict[str, Any]]:
        page = 1
        yielded = 0

        while True:
            items = self.fetch_page(
                page=page,
                per_page=100,
                state=state,
            )

            if not items:
                break

            for item in items:
                # GitHub's /issues endpoint includes pull requests.
                if "pull_request" in item:
                    continue

                yield item
                yielded += 1

                if max_issues is not None and yielded >= max_issues:
                    return

            if len(items) < 100:
                break

            page += 1


def normalize_issue(issue: dict[str, Any], repo: str) -> DocumentRecord:
    number = int(issue["number"])
    title = (issue.get("title") or "").strip()
    body = (issue.get("body") or "").strip()

    labels = [
        label.get("name", "").strip()
        for label in issue.get("labels", [])
        if label.get("name")
    ]

    state = issue.get("state", "unknown")
    html_url = issue.get(
        "html_url",
        f"https://github.com/{repo}/issues/{number}",
    )

    metadata_lines = [
        f"GitHub Issue #{number}",
        f"State: {state}",
        f"Labels: {', '.join(labels) if labels else 'none'}",
        "",
    ]

    content = "\n".join(metadata_lines)
    if body:
        content += f"Title: {title}\n\n{body}"
    else:
        content += f"Title: {title}"

    return DocumentRecord(
        document_id=deterministic_document_id(repo, number),
        source="Kubernetes GitHub Issues",
        source_type="github_issue",
        source_id=f"{repo}#{number}",
        source_url=html_url,
        title=title,
        content=content,
        relative_path=f"github://{repo}/issues/{number}",
        category="github_issue",
        doc_type="github_issue",
        metadata={
            "github_repo": repo,
            "github_issue_number": number,
            "status": state,
            "labels": labels,
            "created_at": issue.get("created_at"),
            "updated_at": issue.get("updated_at"),
        },
    )


def write_jsonl(records: Iterator[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def ingest_issues(
    *,
    repo: str,
    api_url: str,
    token: str | None,
    max_issues: int | None,
    state: str,
    raw_output: Path,
    normalized_output: Path,
) -> int:
    client = GitHubIssuesClient(
        repo=repo,
        api_url=api_url,
        token=token,
    )

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    normalized_output.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with (
        raw_output.open("w", encoding="utf-8") as raw_handle,
        normalized_output.open("w", encoding="utf-8") as normalized_handle,
    ):
        for issue in client.iter_issues(
            max_issues=max_issues,
            state=state,
        ):
            raw_handle.write(
                json.dumps(issue, ensure_ascii=False) + "\n"
            )

            record = normalize_issue(issue, repo)
            normalized_handle.write(
                json.dumps(
                    record.to_dict(),
                    ensure_ascii=False,
                )
                + "\n"
            )

            count += 1

            if count % 25 == 0:
                print(f"Ingested GitHub issues: {count}")

    return count


def parse_args() -> argparse.Namespace:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Ingest Kubernetes GitHub Issues into JSONL.",
    )

    parser.add_argument(
        "--repo",
        default=os.getenv("RAG_GITHUB_REPO", DEFAULT_REPO),
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("RAG_GITHUB_API_URL", DEFAULT_API_URL),
    )
    parser.add_argument(
        "--token",
        default=os.getenv("GITHUB_TOKEN"),
        help="GitHub token. Prefer setting GITHUB_TOKEN instead of passing it on the command line.",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=int(os.getenv("RAG_GITHUB_MAX_ISSUES", "200")),
    )
    parser.add_argument(
        "--state",
        choices=("open", "closed", "all"),
        default=os.getenv("RAG_GITHUB_ISSUE_STATE", "all"),
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=settings.raw_dir / "github" / "issues.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.processed_dir / "github_issues.jsonl",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("Kubernetes GitHub Issues Ingestion")
    print("=" * 80)
    print(f"Repository: {args.repo}")
    print(f"State: {args.state}")
    print(f"Max issues: {args.max_issues}")
    print(f"Raw output: {args.raw_output}")
    print(f"Normalized output: {args.output}")
    print()

    count = ingest_issues(
        repo=args.repo,
        api_url=args.api_url,
        token=args.token,
        max_issues=args.max_issues,
        state=args.state,
        raw_output=args.raw_output,
        normalized_output=args.output,
    )

    print()
    print(f"Completed. Issues ingested: {count}")
    print(f"Normalized records: {args.output}")


if __name__ == "__main__":
    main()
