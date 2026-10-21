"""GitHub Issues ingestion with safer bounded pagination."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests

from rag_assist.config import get_settings
from rag_assist.ingestion.github_issues import (
    DEFAULT_API_URL,
    DEFAULT_REPO,
    GitHubIssuesClient,
    normalize_issue,
)


def ingest_recent_issues(
    *,
    repo: str,
    api_url: str,
    token: str | None,
    max_issues: int,
    state: str,
    raw_output: Path,
    normalized_output: Path,
) -> int:
    """Ingest a bounded recent snapshot.

    The GitHub `/issues` endpoint can return HTTP 422 for deeper pagination on
    very large result sets. We use a small per-page request and reduce the page
    size once on a 422 response before failing.
    """
    client = GitHubIssuesClient(
        repo=repo,
        api_url=api_url,
        token=token,
    )

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    normalized_output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    page = 1

    with (
        raw_output.open("w", encoding="utf-8") as raw_handle,
        normalized_output.open("w", encoding="utf-8") as normalized_handle,
    ):
        while count < max_issues:
            remaining = max_issues - count
            per_page = min(50, max(1, remaining))

            try:
                items = client.fetch_page(
                    page=page,
                    per_page=per_page,
                    state=state,
                )
            except requests.HTTPError as exc:
                response = exc.response

                if (
                    response is not None
                    and response.status_code == 422
                    and per_page > 10
                ):
                    per_page = 10
                    print(
                        f"GitHub returned 422 on page {page}; "
                        f"retrying with per_page={per_page}"
                    )
                    items = client.fetch_page(
                        page=page,
                        per_page=per_page,
                        state=state,
                    )
                else:
                    raise

            if not items:
                break

            issue_count_this_page = 0

            for issue in items:
                # `/issues` also returns pull requests.
                if "pull_request" in issue:
                    continue

                raw_handle.write(
                    json.dumps(
                        issue,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                normalized = normalize_issue(
                    issue,
                    repo,
                )

                normalized_handle.write(
                    json.dumps(
                        normalized.to_dict(),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                count += 1
                issue_count_this_page += 1

                if count % 25 == 0:
                    print(
                        f"Ingested GitHub issues: {count}"
                    )

                if count >= max_issues:
                    break

            # If the API returned fewer items than requested, this is the last page.
            if len(items) < per_page:
                break

            page += 1

    return count


def parse_args() -> argparse.Namespace:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Ingest a bounded Kubernetes GitHub Issues snapshot."
    )
    parser.add_argument(
        "--repo",
        default=os.getenv(
            "RAG_GITHUB_REPO",
            DEFAULT_REPO,
        ),
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv(
            "RAG_GITHUB_API_URL",
            DEFAULT_API_URL,
        ),
    )
    parser.add_argument(
        "--token",
        default=os.getenv("GITHUB_TOKEN"),
        help=(
            "GitHub token. Prefer GITHUB_TOKEN environment variable "
            "instead of putting a token in shell history."
        ),
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=int(
            os.getenv(
                "RAG_GITHUB_MAX_ISSUES",
                "500",
            )
        ),
    )
    parser.add_argument(
        "--state",
        choices=("open", "closed", "all"),
        default=os.getenv(
            "RAG_GITHUB_ISSUE_STATE",
            "all",
        ),
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=settings.raw_dir / "github" / "issues.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.github_issues_file,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_issues <= 0:
        raise ValueError(
            "--max-issues must be greater than zero."
        )

    print("=" * 80)
    print("Kubernetes GitHub Issues Ingestion")
    print("=" * 80)
    print(f"Repository: {args.repo}")
    print(f"State: {args.state}")
    print(f"Max issues: {args.max_issues}")
    print(f"Output: {args.output}")
    print()

    count = ingest_recent_issues(
        repo=args.repo,
        api_url=args.api_url,
        token=args.token,
        max_issues=args.max_issues,
        state=args.state,
        raw_output=args.raw_output,
        normalized_output=args.output,
    )

    print()
    print(
        f"Completed. Issues ingested: {count}"
    )
    print(
        f"Normalized records: {args.output}"
    )


if __name__ == "__main__":
    main()
