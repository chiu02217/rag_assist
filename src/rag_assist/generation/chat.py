"""Interactive single-turn RAG chat."""
from __future__ import annotations
import argparse
from rag_assist.generation.answer_service import AnswerService
from rag_assist.generation.llm import OpenAICompatibleLLM
from rag_assist.retrieval.service import HybridRetriever

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask questions against Kubernetes docs and GitHub Issues."
    )
    parser.add_argument(
        "--source-type",
        choices=("documentation", "github_issue", "all"),
        default="all",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Overrides RAG_LLM_BASE_URL, e.g. http://localhost:11434/v1.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Overrides RAG_LLM_MODEL, e.g. qwen2.5:3b.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    llm = OpenAICompatibleLLM.from_env()
    if args.base_url:
        llm.base_url = args.base_url
    if args.model:
        llm.model = args.model
    answer_service = AnswerService(llm)

    print("=" * 80)
    print("RAG Assist - Grounded Q&A")
    print("=" * 80)
    print(f"Source scope: {args.source_type}")
    print(f"LLM endpoint: {llm.base_url}")
    print(f"LLM model: {llm.model}")
    print("Type `exit` to stop.\n")

    with HybridRetriever(source_type=args.source_type) as retriever:
        while True:
            query = input("You > ").strip()
            if query.lower() in {"exit", "quit"}:
                break
            if not query:
                continue
            try:
                contexts = retriever.retrieve(query)
                result = answer_service.answer(query, contexts)
            except RuntimeError as exc:
                print(f"\nError: {exc}\n")
                continue

            print("\nAssistant >")
            print(result.answer)
            print("\nSources:")
            if not result.sources:
                print("  (No usable sources retrieved.)")
            for source in result.sources:
                print(f"  [{source['citation_id']}] [{source['source_type']}] {source['title']}")
                if source.get("section"):
                    print(f"      Section: {source['section']}")
                if source.get("source_id"):
                    print(f"      Source ID: {source['source_id']}")
                if source.get("url"):
                    print(f"      URL: {source['url']}")
            if result.warnings:
                print("\nWarnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
            print(f"\n[debug] context_chars={result.context_chars}; cited={result.used_citations}")
            print("\n" + "-" * 80 + "\n")

if __name__ == "__main__":
    main()
