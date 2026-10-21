"""Build a unified document corpus from existing docs and source connectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_assist.config import get_settings


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    return records


def _with_documentation_source_contract(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Add the new multi-source fields to legacy documentation records."""
    if record.get("source_type"):
        return record

    relative_path = record.get("relative_path", "")
    clean_path = relative_path.replace("\\", "/")

    if clean_path.startswith("/"):
        clean_path = clean_path[1:]

    if clean_path.endswith(".md"):
        clean_path = clean_path[:-3]

    clean_path = clean_path.rstrip("/")

    url = (
        f"https://kubernetes.io/docs/{clean_path}/"
        if clean_path
        else "https://kubernetes.io/docs/"
    )

    output = dict(record)
    output["source_type"] = "documentation"
    output["source_id"] = record.get(
        "document_id",
        relative_path,
    )
    output["source_url"] = url
    output.setdefault("category", "documentation")
    output.setdefault("doc_type", "documentation")

    return output


def merge_documents(
    documentation_file: Path,
    github_issues_file: Path,
    output_file: Path,
) -> tuple[int, int, int]:
    docs = [
        _with_documentation_source_contract(record)
        for record in _read_jsonl(documentation_file)
    ]
    issues = _read_jsonl(github_issues_file)

    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for record in [*docs, *issues]:
        document_id = record.get("document_id")
        if not document_id:
            raise ValueError(
                "Every corpus record must have document_id."
            )

        if document_id in seen_ids:
            continue

        seen_ids.add(document_id)
        merged.append(record)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as handle:
        for record in merged:
            handle.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    return len(docs), len(issues), len(merged)


def parse_args() -> argparse.Namespace:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Merge documentation and GitHub Issues into a unified corpus.",
    )
    parser.add_argument(
        "--documentation",
        type=Path,
        default=settings.documents_file,
    )
    parser.add_argument(
        "--github-issues",
        type=Path,
        default=settings.processed_dir / "github_issues.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.processed_dir / "documents_all.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    docs_count, issue_count, merged_count = merge_documents(
        documentation_file=args.documentation,
        github_issues_file=args.github_issues,
        output_file=args.output,
    )

    print("=" * 80)
    print("Unified Corpus")
    print("=" * 80)
    print(f"Documentation records: {docs_count}")
    print(f"GitHub issue records: {issue_count}")
    print(f"Unified records: {merged_count}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
