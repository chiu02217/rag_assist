"""Build bounded, citation-addressable context from retrieved parents."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

# max_chars_per_source：限制單一來源的最大字元數（預設 4000 字）。

# max_chars：限制總體上下文的最大字元數（預設 12000 字）。

# 當內容超出限制時，它會自動截斷（Truncate），並在結尾補上 \n[Context truncated]，同時將 truncated 標記設為 True。
@dataclass(frozen=True)
class ContextSource:
    citation_id: str
    title: str
    section: str
    source_type: str
    source_id: str
    source_url: str
    relative_path: str
    content: str
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id, "title": self.title,
            "section": self.section, "source_type": self.source_type,
            "source_id": self.source_id, "source_url": self.source_url,
            "relative_path": self.relative_path, "score": self.score,
        }

@dataclass(frozen=True)
class BuiltContext:
    text: str
    sources: list[ContextSource]
    truncated: bool

def build_context(parent_results: list[dict[str, Any]], *,
                  max_chars: int = 12000,
                  max_chars_per_source: int = 4000) -> BuiltContext:
    if max_chars <= 0 or max_chars_per_source <= 0:
        raise ValueError("Context character limits must be positive.")
    blocks, sources = [], []
    chars_used, truncated = 0, False

    for result in parent_results:
        parent = result.get("parent", result)
        raw = str(parent.get("content") or "").strip()
        if not raw:
            continue
        citation_id = f"S{len(sources) + 1}"
        title = str(parent.get("title") or "Untitled source")
        section = str(parent.get("section") or "")
        source_type = str(parent.get("source_type") or "unknown")
        source_id = str(parent.get("source_id") or "")
        source_url = str(parent.get("source_url") or "")
        relative_path = str(parent.get("relative_path") or "")
        header = (f"[{citation_id}]\nSource type: {source_type}\nTitle: {title}\n"
                  f"Section: {section}\nSource ID: {source_id}\nURL: {source_url}\nContent:\n")
        remaining = max_chars - chars_used - (2 if blocks else 0)
        content_budget = min(max_chars_per_source, len(raw), remaining - len(header))
        if content_budget <= 0:
            truncated = True
            break
        content = raw[:content_budget]
        if len(content) < len(raw):
            content = content.rstrip() + "\n[Context truncated]"
            truncated = True
        block = header + content
        if blocks:
            block = "\n\n" + block
        if chars_used + len(block) > max_chars:
            truncated = True
            break
        child = result.get("best_child") or {}
        score = child.get("reranker_score")
        try:
            score = float(score) if score is not None else None
        except (ValueError, TypeError):
            score = None
        sources.append(ContextSource(
            citation_id=citation_id, title=title, section=section,
            source_type=source_type, source_id=source_id, source_url=source_url,
            relative_path=relative_path, content=content, score=score,
        ))
        blocks.append(block)
        chars_used += len(block)
    return BuiltContext("".join(blocks), sources, truncated)
