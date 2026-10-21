from rag_assist.generation.answer_service import AnswerService
from rag_assist.generation.citations import validate_citations
from rag_assist.generation.context_builder import ContextSource, build_context

class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.called = False
    def generate(self, *, system_prompt, user_prompt):
        self.called = True
        assert "[S1]" in user_prompt
        return self.response

def _parent(source_type="documentation"):
    return {
        "parent": {
            "title": "About cgroup v2",
            "section": "What is cgroup v2?",
            "source_type": source_type,
            "source_id": "doc-1",
            "source_url": "https://kubernetes.io/docs/concepts/architecture/cgroups/",
            "relative_path": "concepts/architecture/cgroups.md",
            "content": "cgroup v2 provides a unified control system.",
        },
        "best_child": {"reranker_score": 0.9},
        "support_count": 1,
    }

def test_context_building_provides_source_ids():
    context = build_context([_parent()], max_chars=2000)
    assert len(context.sources) == 1
    assert context.sources[0].citation_id == "S1"
    assert "[S1]" in context.text
    assert "unified control system" in context.text

def test_answer_service_calls_llm_and_returns_backend_sources():
    llm = FakeLLM("cgroup v2 uses a unified hierarchy [S1].")
    service = AnswerService(llm, max_context_chars=2000)
    result = service.answer("What is cgroup v2?", [_parent()])
    assert llm.called
    assert result.used_citations == ["S1"]
    assert result.sources[0]["url"].startswith("https://kubernetes.io/")
    assert result.warnings == []

def test_no_context_does_not_call_llm():
    llm = FakeLLM("should not be called")
    result = AnswerService(llm).answer("Question", [])
    assert not llm.called
    assert result.sources == []
    assert "無法" in result.answer

def test_unknown_citation_is_removed():
    source = ContextSource(
        citation_id="S1", title="T", section="", source_type="documentation",
        source_id="d", source_url="https://example.com", relative_path="",
        content="fact",
    )
    answer, used, unknown = validate_citations("Fact [S1] and made up [S99].", [source])
    assert "[S99]" not in answer
    assert used == ["S1"]
    assert unknown == ["S99"]
