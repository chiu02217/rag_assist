import json
import sqlite3
from pathlib import Path

DB_FILE = Path("data/db/rag_metadata.sqlite3")
PARENTS_FILE = Path("data/processed/parents.jsonl")

class ParentStore:
    """
    Persistent parent-context store.

    Local:
        SQLite

    Production migration:
        PostgreSQL can implement the same interface.
    """

    def __init__(self, db_file=DB_FILE):
        self.db_file = Path(db_file)
        self.db_file.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(self.db_file)
        self.connection.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self):
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS parent_sections (
                parent_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                parent_index INTEGER NOT NULL,
                title TEXT NOT NULL,
                section TEXT NOT NULL,
                source TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                category TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_parent_document
            ON parent_sections(document_id)
            """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_parent_category
            ON parent_sections(category)
            """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_parent_doc_type
            ON parent_sections(doc_type)
            """
        )

        self.connection.commit()

    def upsert_parents_from_jsonl(self, parents_file=PARENTS_FILE):
        rows = []
        with Path(parents_file).open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))

        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO parent_sections (
                    parent_id,
                    document_id,
                    parent_index,
                    title,
                    section,
                    source,
                    relative_path,
                    category,
                    doc_type,
                    content
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(parent_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    parent_index = excluded.parent_index,
                    title = excluded.title,
                    section = excluded.section,
                    source = excluded.source,
                    relative_path = excluded.relative_path,
                    category = excluded.category,
                    doc_type = excluded.doc_type,
                    content = excluded.content,
                    updated_at = CURRENT_TIMESTAMP
                """,
                [
                    (
                        row["parent_id"],
                        row["document_id"],
                        row["parent_index"],
                        row["title"],
                        row["section"],
                        row["source"],
                        row["relative_path"],
                        row["category"],
                        row["doc_type"],
                        row["content"],
                    )
                    for row in rows
                ],
            )

        return len(rows)

    def get_parents(self, parent_ids):
        parent_ids = list(dict.fromkeys(parent_ids))
        if not parent_ids:
            return {}

        placeholders = ",".join(["?"] * len(parent_ids))
        cursor = self.connection.execute(
            f"""
            SELECT
                parent_id,
                document_id,
                parent_index,
                title,
                section,
                source,
                relative_path,
                category,
                doc_type,
                content
            FROM parent_sections
            WHERE parent_id IN ({placeholders})
            """,
            parent_ids,
        )

        return {
            row["parent_id"]: dict(row)
            for row in cursor.fetchall()
        }

    def count(self):
        cursor = self.connection.execute(
            "SELECT COUNT(*) AS count FROM parent_sections"
        )
        return cursor.fetchone()["count"]

    def close(self):
        self.connection.close()
