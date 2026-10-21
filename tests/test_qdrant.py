from qdrant_client import QdrantClient


def main():
    client = QdrantClient(
        url="http://localhost:6333"
    )

    print("Connected to Qdrant!")

    collections = client.get_collections()

    print(collections)


if __name__ == "__main__":
    main()