"""Context assembly, grounded generation, and backend citation mapping."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import os
from rag_assist.generation.citations import format_source_list, validate_citations
from rag_assist.generation.context_builder import build_context
from rag_assist.generation.llm import OpenAICompatibleLLM
from rag_assist.generation.prompt import SYSTEM_PROMPT, make_user_prompt

@dataclass
class AnswerResult:
    query: str
    answer: str
    sources: list[dict[str, Any]]
    used_citations: list[str]
    warnings: list[str]
    context_chars: int

class AnswerService:
    def __init__(self, llm: OpenAICompatibleLLM, max_context_chars: int | None = None):
        self.llm = llm
        self.max_context_chars = max_context_chars or int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))

    def answer(self, query: str, parent_results: list[dict[str, Any]]) -> AnswerResult:
        context = build_context(parent_results, max_chars=self.max_context_chars)
        if not context.sources:
            return AnswerResult(
                query=query,
                answer="目前沒有檢索到可用的來源內容，因此我無法根據知識庫回答這個問題。",
                sources=[], used_citations=[],
                warnings=["No usable parent contexts were retrieved; the LLM was not called."],
                context_chars=0,
            )
        warnings = []
        if context.truncated:
            warnings.append("Retrieved context was truncated to fit the configured character budget.")
        answer = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=make_user_prompt(query, context.text),
        )
        answer, used, unknown = validate_citations(answer, context.sources)
        if unknown:
            warnings.append("Removed unknown citation IDs: " + ", ".join(unknown))
        if not used:
            warnings.append("Model returned no valid [S#] citations; source list is shown, but claim-level citations were not verified.")
        return AnswerResult(
            query=query,
            answer=answer,
            sources=format_source_list(context.sources),
            used_citations=used,
            warnings=warnings,
            context_chars=len(context.text),
        )
