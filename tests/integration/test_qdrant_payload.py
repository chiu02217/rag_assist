from qdrant_client import QdrantClient


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "kubernetes_docs"


def main():

    client = QdrantClient(
        url=QDRANT_URL
    )

    result = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=3,
        with_payload=True,
        with_vectors=False,
    )

    points = result[0]

    print(
        f"Checked points: {len(points)}"
    )

    for i, point in enumerate(
        points,
        start=1
    ):

        print("\n" + "=" * 80)

        print(
            f"Point {i}"
        )

        print(
            f"Qdrant ID: {point.id}"
        )

        payload = point.payload

        print(
            f"chunk_id: "
            f"{payload.get('chunk_id')}"
        )

        print(
            f"parent_id: "
            f"{payload.get('parent_id')}"
        )

        print(
            f"document_id: "
            f"{payload.get('document_id')}"
        )

        print(
            f"title: "
            f"{payload.get('title')}"
        )

        print(
            f"section: "
            f"{payload.get('section')}"
        )

        print(
            f"category: "
            f"{payload.get('category')}"
        )


if __name__ == "__main__":
    main()