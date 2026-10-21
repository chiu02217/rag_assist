from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import re

DOCS_DIR = Path("data/raw/kubernetes-website/content/en/docs")
OUTPUT_FILE = Path("data/processed/documents.jsonl")

@dataclass
class Document:
    document_id: str
    content: str
    source: str
    title: str
    relative_path: str
    category: str
    doc_type: str

def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()

def extract_title(content: str, file_path: Path) -> str:
    front_matter = re.match(r"^---\s*\n(.*?)\n---\s*", content, flags=re.DOTALL)
    if front_matter:
        title_match = re.search(
            r"^title:\s*[\"']?(.*?)[\"']?\s*$",
            front_matter.group(1),
            flags=re.MULTILINE,
        )
        if title_match:
            return title_match.group(1).strip()

    h1_match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()

    return file_path.stem.replace("-", " ").title()

def clean_hugo_shortcodes(content: str) -> str:
    content = re.sub(
        r'\{\{<\s*glossary_tooltip\b[^>]*?\btext="([^"]+)"[^>]*>\}\}',
        r"\1",
        content,
    )
    content = re.sub(r'\{\{<\s*glossary_definition\b[^>]*>\}\}', "", content)
    content = re.sub(r'\{\{<\s*feature-state\b[^>]*>\}\}', "", content)
    content = re.sub(
        r'\{\{<\s*figure\b[^>]*?\bcaption="([^"]+)"[^>]*>\}\}',
        r"\n\1\n",
        content,
    )
    content = re.sub(
        r'\{\{<\s*details\b[^>]*?\bsummary="([^"]+)"[^>]*>\}\}',
        r"\n## \1\n",
        content,
    )
    content = re.sub(r"\{\{<\s*/details\s*>\}\}", "", content)
    content = re.sub(r"\{\{<\s*(note|warning|caution|tip)\s*>\}\}", "", content)
    content = re.sub(r"\{\{<\s*/[^>]+\s*>\}\}", "", content)
    content = re.sub(r"\{\{<.*?>\}\}", "", content, flags=re.DOTALL)
    return content

def clean_markdown(content: str) -> str:
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
    content = clean_hugo_shortcodes(content)
    content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)
    content = content.replace("\r\n", "\n")
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()

def load_documents():
    documents = []
    files = sorted(DOCS_DIR.rglob("*.md"))
    print(f"Found {len(files)} markdown files.")

    for file_path in files:
        try:
            raw_content = file_path.read_text(encoding="utf-8", errors="ignore")
            if not raw_content.strip():
                continue

            relative_path = str(file_path.relative_to(DOCS_DIR))
            clean_content = clean_markdown(raw_content)
            if not clean_content:
                continue

            parts = Path(relative_path).parts
            category = parts[0] if len(parts) >= 2 else "root"
            doc_type = "documentation"

            documents.append(
                Document(
                    document_id=stable_id(relative_path),
                    content=clean_content,
                    source="Kubernetes Documentation",
                    title=extract_title(raw_content, file_path),
                    relative_path=relative_path,
                    category=category,
                    doc_type=doc_type,
                )
            )
        except Exception as exc:
            print(f"Failed to read {file_path}: {exc}")

    return documents

def save_jsonl(documents):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for document in documents:
            f.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")

def main():
    documents = load_documents()
    print(f"Loaded documents: {len(documents)}")
    save_jsonl(documents)
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
