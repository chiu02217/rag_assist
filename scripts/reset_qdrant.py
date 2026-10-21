from qdrant_client import QdrantClient


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "kubernetes_docs"


def main():

    client = QdrantClient(
        url=QDRANT_URL
    )

    collections = client.get_collections()

    exists = any(
        collection.name == COLLECTION_NAME
        for collection in collections.collections
    )

    if exists:

        print(
            f"Deleting collection: "
            f"{COLLECTION_NAME}"
        )

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

        print("Collection deleted.")

    else:

        print(
            f"Collection does not exist: "
            f"{COLLECTION_NAME}"
        )


if __name__ == "__main__":
    main()