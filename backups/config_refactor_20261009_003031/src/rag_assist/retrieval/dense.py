import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "kubernetes_docs"
MODEL_NAME = "BAAI/bge-m3"

TOP_K = 5


def load_model():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        model_kwargs={
            "torch_dtype": torch.float16
        }
    )

    return model


def search(
    client,
    model,
    query: str,
    top_k: int = 5
):

    # 1. Convert query into vector
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # 2. Search Qdrant
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=top_k,
        with_payload=True
    )

    return results.points


def main():

    client = QdrantClient(
        url=QDRANT_URL
    )

    model = load_model()

    collection_info = client.get_collection(
        COLLECTION_NAME
    )

    print(
        f"Collection points: "
        f"{collection_info.points_count}"
    )

    print("\nType your question.")
    print("Type 'exit' to quit.\n")

    while True:

        query = input("Query > ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = search(
            client,
            model,
            query,
            TOP_K
        )

        print("\n" + "=" * 80)

        for i, result in enumerate(
            results,
            start=1
        ):

            payload = result.payload

            print(
                f"\n[{i}] "
                f"Score: {result.score:.4f}"
            )

            print(
                f"Title: "
                f"{payload['title']}"
            )

            print(
                f"Section: "
                f"{payload['section']}"
            )

            print(
                f"Path: "
                f"{payload['relative_path']}"
            )

            print(
                f"\n{payload['content'][:800]}"
            )

            print("-" * 80)


if __name__ == "__main__":
    main()