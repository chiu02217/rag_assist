"""Validate model citation IDs and format citations from retrieved metadata."""
from __future__ import annotations
import re
from typing import Any
from rag_assist.generation.context_builder import ContextSource

_CITATION_RE = re.compile(r"\[(S\d+)\]")

def validate_citations(answer: str, sources: list[ContextSource]) -> tuple[str, list[str], list[str]]:
    valid = {source.citation_id for source in sources}
    found = _CITATION_RE.findall(answer)
    unknown = sorted({cid for cid in found if cid not in valid})
    clean = _CITATION_RE.sub(lambda m: m.group(0) if m.group(1) in valid else "", answer)
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    clean = re.sub(r" +([,.;:])", r"\1", clean).strip()
    used = []
    for cid in found:
        if cid in valid and cid not in used:
            used.append(cid)
    return clean, used, unknown

def format_source_list(sources: list[ContextSource]) -> list[dict[str, Any]]:
    # URLs come from retrieved metadata, not from the LLM.
    return [{**source.to_dict(), "url": source.source_url} for source in sources]
