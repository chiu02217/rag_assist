"""Structure-aware Parent/Child chunking for a unified multi-source corpus."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rag_assist.config import get_settings


settings = get_settings()

INPUT_FILE = settings.documents_all_file
PARENTS_OUTPUT = settings.parents_file
CHUNKS_OUTPUT = settings.chunks_file

PARENT_MAX_CHARS = settings.parent_max_chars
PARENT_OVERLAP_CHARS = settings.parent_overlap_chars
CHILD_CHUNK_SIZE = settings.child_chunk_size
CHILD_OVERLAP = settings.child_overlap


@dataclass
class ParentSection:
    parent_id: str
    document_id: str
    parent_index: int
    title: str
    section: str
    source: str
    source_type: str
    source_id: str
    source_url: str
    relative_path: str
    category: str
    doc_type: str
    content: str
    metadata: dict[str, Any]


@dataclass
class ChildChunk:
    chunk_id: str
    parent_id: str
    document_id: str
    parent_index: int
    chunk_index: int
    title: str
    section: str
    source: str
    source_type: str
    source_id: str
    source_url: str
    relative_path: str
    category: str
    doc_type: str
    content: str
    metadata: dict[str, Any]


def load_documents() -> list[dict[str, Any]]:
    documents = []

    with INPUT_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                documents.append(json.loads(line))

    return documents


def parse_sections(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    sections = []

    current_heading = "Introduction"
    current_lines: list[str] = []

    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)

        if match:
            content = "\n".join(current_lines).strip()

            if content:
                sections.append(
                    {
                        "heading": current_heading,
                        "content": content,
                    }
                )

            current_heading = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    content = "\n".join(current_lines).strip()

    if content:
        sections.append(
            {
                "heading": current_heading,
                "content": content,
            }
        )

    return sections


def split_long_text(
    text: str,
    max_chars: int,
    overlap: int,
) -> list[str]:
    if len(text) <= max_chars:
        return [text.strip()]

    paragraphs = re.split(r"\n\s*\n", text)
    results: list[str] = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        candidate = (
            paragraph
            if not current
            else current + "\n\n" + paragraph
        )

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            results.append(current.strip())

        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        start = 0

        while start < len(paragraph):
            end = min(start + max_chars, len(paragraph))
            results.append(paragraph[start:end].strip())

            if end >= len(paragraph):
                current = ""
                break

            start = max(0, end - overlap)

    if current:
        results.append(current.strip())

    return results


def split_child_chunks(
    parent_content: str,
    chunk_size: int = CHILD_CHUNK_SIZE,
    overlap: int = CHILD_OVERLAP,
) -> list[str]:
    if len(parent_content) <= chunk_size:
        return [parent_content.strip()]

    paragraphs = re.split(
        r"\n\s*\n",
        parent_content,
    )

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        candidate = (
            paragraph
            if not current
            else current + "\n\n" + paragraph
        )

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())

        if len(paragraph) <= chunk_size:
            current = paragraph
            continue

        start = 0
        pieces: list[str] = []

        while start < len(paragraph):
            end = min(start + chunk_size, len(paragraph))
            pieces.append(paragraph[start:end].strip())

            if end >= len(paragraph):
                break

            start = max(0, end - overlap)

        chunks.extend(pieces[:-1])
        current = pieces[-1]

    if current:
        chunks.append(current.strip())

    return chunks


def make_parent_id(document_id: str, parent_index: int) -> str:
    return f"{document_id}:parent:{parent_index}"


def make_chunk_id(parent_id: str, chunk_index: int) -> str:
    return f"{parent_id}:chunk:{chunk_index}"


def normalize_source_fields(
    document: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(document.get("metadata") or {})

    return {
        "source_type": document.get(
            "source_type",
            "documentation",
        ),
        "source_id": document.get(
            "source_id",
            document.get(
                "document_id",
                "",
            ),
        ),
        "source_url": document.get(
            "source_url",
            "",
        ),
        "source": document.get(
            "source",
            settings.documentation_source_name,
        ),
        "relative_path": document.get(
            "relative_path",
            "",
        ),
        "category": document.get(
            "category",
            "documentation",
        ),
        "doc_type": document.get(
            "doc_type",
            "documentation",
        ),
        "metadata": metadata,
    }


def create_parent_child_data(
    documents: list[dict[str, Any]],
) -> tuple[list[ParentSection], list[ChildChunk]]:
    parents: list[ParentSection] = []
    chunks: list[ChildChunk] = []

    for document in documents:
        source = normalize_source_fields(document)

        relative_path = source["relative_path"]

        # Keep the existing behavior for Kubernetes docs.
        if (
            source["source_type"] == "documentation"
            and Path(relative_path).name.lower() == "_index.md"
        ):
            continue

        parent_index = 0

        sections = parse_sections(
            document.get("content", "").strip()
        )

        # GitHub issue content might not contain Markdown headings.
        if not sections:
            sections = [
                {
                    "heading": document.get(
                        "title",
                        "Document",
                    ),
                    "content": document.get(
                        "content",
                        "",
                    ).strip(),
                }
            ]

        for section in sections:
            if not section["content"].strip():
                continue

            parent_parts = split_long_text(
                section["content"],
                PARENT_MAX_CHARS,
                PARENT_OVERLAP_CHARS,
            )

            for parent_content in parent_parts:
                parent_id = make_parent_id(
                    document["document_id"],
                    parent_index,
                )

                parents.append(
                    ParentSection(
                        parent_id=parent_id,
                        document_id=document["document_id"],
                        parent_index=parent_index,
                        title=document.get(
                            "title",
                            section["heading"],
                        ),
                        section=section["heading"],
                        source=source["source"],
                        source_type=source["source_type"],
                        source_id=source["source_id"],
                        source_url=source["source_url"],
                        relative_path=relative_path,
                        category=source["category"],
                        doc_type=source["doc_type"],
                        content=parent_content,
                        metadata=source["metadata"],
                    )
                )

                child_contents = split_child_chunks(
                    parent_content
                )

                for child_index, child_content in enumerate(
                    child_contents
                ):
                    chunks.append(
                        ChildChunk(
                            chunk_id=make_chunk_id(
                                parent_id,
                                child_index,
                            ),
                            parent_id=parent_id,
                            document_id=document["document_id"],
                            parent_index=parent_index,
                            chunk_index=child_index,
                            title=document.get(
                                "title",
                                section["heading"],
                            ),
                            section=section["heading"],
                            source=source["source"],
                            source_type=source["source_type"],
                            source_id=source["source_id"],
                            source_url=source["source_url"],
                            relative_path=relative_path,
                            category=source["category"],
                            doc_type=source["doc_type"],
                            content=child_content,
                            metadata=source["metadata"],
                        )
                    )

                parent_index += 1

    return parents, chunks


def save_jsonl(records: list[Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    asdict(record),
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    print("=" * 80)
    print("Multi-source Parent-Child Chunking")
    print("=" * 80)

    documents = load_documents()
    print(f"Loaded unified documents: {len(documents)}")

    source_counts: dict[str, int] = {}

    for document in documents:
        source_type = document.get(
            "source_type",
            "documentation",
        )
        source_counts[source_type] = (
            source_counts.get(source_type, 0) + 1
        )

    print(f"Sources: {source_counts}")

    parents, chunks = create_parent_child_data(
        documents
    )

    save_jsonl(
        parents,
        PARENTS_OUTPUT,
    )

    save_jsonl(
        chunks,
        CHUNKS_OUTPUT,
    )

    print(f"Parent contexts: {len(parents)}")
    print(f"Child chunks: {len(chunks)}")
    print(f"Saved parents: {PARENTS_OUTPUT}")
    print(f"Saved children: {CHUNKS_OUTPUT}")


if __name__ == "__main__":
    main()
