from pathlib import Path


DOCS_DIR = Path("data/raw/kubernetes-website/content/en/docs")


def main():
    files = list(DOCS_DIR.rglob("*.md"))

    print(f"Total markdown files: {len(files)}")

    total_chars = 0

    for file in files:
        text = file.read_text(encoding="utf-8", errors="ignore")
        total_chars += len(text)

    print(f"Total characters: {total_chars}")


if __name__ == "__main__":
    main()