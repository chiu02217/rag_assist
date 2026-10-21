"""Canonical document model shared by ingestion connectors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DocumentRecord:
    """Normalized source document used by the RAG ingestion layer."""

    document_id: str
    source: str
    source_type: str
    source_id: str
    source_url: str
    title: str
    content: str

    relative_path: str = ""
    category: str = ""
    doc_type: str = "document"

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        # Keep top-level fields compatible with the current chunker/indexer.
        data.update(self.metadata)
        data["metadata"] = self.metadata

        return data
