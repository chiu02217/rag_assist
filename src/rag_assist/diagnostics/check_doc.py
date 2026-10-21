import json

INPUT_FILE = "data/processed/documents.jsonl"


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):

            document = json.loads(line)

            print("=" * 80)
            print(f"Document {i + 1}")
            print(f"Title: {document['title']}")
            print(f"Path: {document['relative_path']}")
            print()
            print(document["content"][:1000])

            if i == 4:
                break


if __name__ == "__main__":
    main()