import json
from pathlib import Path

from rank_bm25 import BM25Okapi


INPUT_FILE = Path(
    "data/processed/chunks_structured.jsonl"
)

TOP_K = 5


def load_chunks():
    chunks = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            chunks.append(json.loads(line))

    return chunks


def tokenize(text: str):
    return text.lower().split()


def build_bm25(chunks):

    corpus = [
        tokenize(chunk["content"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(corpus)

    return bm25


def search(
    bm25,
    chunks,
    query,
    top_k=5
):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []

    for index in ranked_indices:

        results.append(
            (
                chunks[index],
                scores[index]
            )
        )

    return results


def main():

    print("Loading chunks...")

    chunks = load_chunks()

    print(
        f"Loaded {len(chunks)} chunks."
    )

    print("Building BM25 index...")

    bm25 = build_bm25(chunks)

    print("BM25 ready.")

    while True:

        query = input("\nQuery > ").strip()

        if query.lower() == "exit":
            break

        results = search(
            bm25,
            chunks,
            query,
            TOP_K
        )

        print("\n" + "=" * 80)

        for rank, (
            chunk,
            score
        ) in enumerate(
            results,
            start=1
        ):

            print(
                f"\n[{rank}] "
                f"Score: {score:.4f}"
            )

            print(
                f"Title: {chunk['title']}"
            )

            print(
                f"Section: {chunk['section']}"
            )

            print(
                f"Path: {chunk['relative_path']}"
            )

            print(
                f"\n{chunk['content'][:800]}"
            )

            print("-" * 80)


if __name__ == "__main__":
    main()