import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict

INPUT_FILE = Path("data/processed/documents.jsonl")
PARENTS_OUTPUT = Path("data/processed/parents.jsonl")
CHUNKS_OUTPUT = Path("data/processed/chunks_structured.jsonl")

# Parent context is larger than the retrieval child chunk.
PARENT_MAX_CHARS = 4000
PARENT_OVERLAP_CHARS = 400

CHILD_CHUNK_SIZE = 1200
CHILD_OVERLAP = 200

@dataclass
class ParentSection:
    parent_id: str
    document_id: str
    parent_index: int
    title: str
    section: str
    source: str
    relative_path: str
    category: str
    doc_type: str
    content: str

@dataclass
class ChildChunk:
    chunk_id: str
    parent_id: str
    document_id: str
    parent_index: int
    chunk_index: int
    title: str
    section: str
    source: str
    relative_path: str
    category: str
    doc_type: str
    content: str

def load_documents():
    documents = []
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))
    return documents

def parse_sections(text: str):
    lines = text.splitlines()
    sections = []
    current_heading = "Introduction"
    current_lines = []

    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append({"heading": current_heading, "content": content})
            current_heading = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    content = "\n".join(current_lines).strip()
    if content:
        sections.append({"heading": current_heading, "content": content})
    return sections

def split_long_text(text: str, max_chars: int, overlap: int):
    if len(text) <= max_chars:
        return [text.strip()]

    paragraphs = re.split(r"\n\s*\n", text)
    results = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        candidate = paragraph if not current else current + "\n\n" + paragraph

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            results.append(current.strip())

        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        start = 0
        while start < len(paragraph):
            end = min(start + max_chars, len(paragraph))
            results.append(paragraph[start:end].strip())
            if end >= len(paragraph):
                current = ""
                break
            start = end - overlap

    if current:
        results.append(current.strip())

    # Keep a bounded overlap between parent windows.
    if overlap <= 0 or len(results) <= 1:
        return results

    overlapped = [results[0]]
    for i in range(1, len(results)):
        prefix = results[i - 1][-overlap:]
        merged = prefix + "\n\n" + results[i]
        overlapped.append(merged[:max_chars + overlap])

    return overlapped

def split_child_chunks(parent_content: str, chunk_size=CHILD_CHUNK_SIZE, overlap=CHILD_OVERLAP):
    if len(parent_content) <= chunk_size:
        return [parent_content.strip()]

    paragraphs = re.split(r"\n\s*\n", parent_content)
    chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        candidate = paragraph if not current else current + "\n\n" + paragraph

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())

        if len(paragraph) <= chunk_size:
            current = paragraph
            continue

        start = 0
        pieces = []
        while start < len(paragraph):
            end = min(start + chunk_size, len(paragraph))
            pieces.append(paragraph[start:end].strip())
            if end >= len(paragraph):
                break
            start = end - overlap

        chunks.extend(pieces[:-1])
        current = pieces[-1]

    if current:
        chunks.append(current.strip())

    return chunks

def make_parent_id(document_id, parent_index):
    return f"{document_id}:parent:{parent_index}"

def make_chunk_id(parent_id, chunk_index):
    return f"{parent_id}:chunk:{chunk_index}"

def create_parent_child_data(documents):
    parents = []
    chunks = []

    for document in documents:
        relative_path = document["relative_path"]

        if Path(relative_path).name.lower() == "_index.md":
            continue

        parent_index = 0

        for section in parse_sections(document["content"]):
            parent_parts = split_long_text(
                section["content"],
                PARENT_MAX_CHARS,
                PARENT_OVERLAP_CHARS,
            )

            for parent_content in parent_parts:
                parent_id = make_parent_id(
                    document["document_id"],
                    parent_index,
                )

                parents.append(
                    ParentSection(
                        parent_id=parent_id,
                        document_id=document["document_id"],
                        parent_index=parent_index,
                        title=document["title"],
                        section=section["heading"],
                        source=document["source"],
                        relative_path=relative_path,
                        category=document["category"],
                        doc_type=document["doc_type"],
                        content=parent_content,
                    )
                )

                for child_index, child_content in enumerate(
                    split_child_chunks(parent_content)
                ):
                    chunks.append(
                        ChildChunk(
                            chunk_id=make_chunk_id(parent_id, child_index),
                            parent_id=parent_id,
                            document_id=document["document_id"],
                            parent_index=parent_index,
                            chunk_index=child_index,
                            title=document["title"],
                            section=section["heading"],
                            source=document["source"],
                            relative_path=relative_path,
                            category=document["category"],
                            doc_type=document["doc_type"],
                            content=child_content,
                        )
                    )

                parent_index += 1

    return parents, chunks

def save_jsonl(records, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

def main():
    documents = load_documents()
    print(f"Loaded documents: {len(documents)}")

    parents, chunks = create_parent_child_data(documents)

    save_jsonl(parents, PARENTS_OUTPUT)
    save_jsonl(chunks, CHUNKS_OUTPUT)

    print(f"Parent contexts: {len(parents)}")
    print(f"Child chunks: {len(chunks)}")
    print(f"Saved parents: {PARENTS_OUTPUT}")
    print(f"Saved children: {CHUNKS_OUTPUT}")

if __name__ == "__main__":
    main()
