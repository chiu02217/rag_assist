"""Persistent parent-context repository with multi-source metadata."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from rag_assist.config import get_settings


settings = get_settings()

DB_FILE = settings.parent_db_file
PARENTS_FILE = settings.parents_file


class ParentStore:
    """SQLite repository for parent contexts.

    The public interface is intentionally simple so SQLite can later be replaced
    by PostgreSQL without changing the retrieval layer.
    """

    def __init__(self, db_file: Path | str = DB_FILE):
        self.db_file = Path(db_file)
        self.db_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            self.db_file
        )
        self.connection.row_factory = sqlite3.Row

        self._initialize_schema()
        self._migrate_schema()

    def _initialize_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS parent_sections (
                parent_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                parent_index INTEGER NOT NULL,
                title TEXT NOT NULL,
                section TEXT NOT NULL,
                source TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'documentation',
                source_id TEXT NOT NULL DEFAULT '',
                source_url TEXT NOT NULL DEFAULT '',
                relative_path TEXT NOT NULL,
                category TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.connection.commit()

    def _migrate_schema(self) -> None:
        """Add columns when upgrading an existing SQLite database."""
        existing = {
            row["name"]
            for row in self.connection.execute(
                "PRAGMA table_info(parent_sections)"
            )
        }

        migrations = {
            "source_type": (
                "TEXT NOT NULL DEFAULT 'documentation'"
            ),
            "source_id": (
                "TEXT NOT NULL DEFAULT ''"
            ),
            "source_url": (
                "TEXT NOT NULL DEFAULT ''"
            ),
            "metadata_json": (
                "TEXT NOT NULL DEFAULT '{}'"
            ),
        }

        for column, definition in migrations.items():
            if column not in existing:
                self.connection.execute(
                    f"""
                    ALTER TABLE parent_sections
                    ADD COLUMN {column} {definition}
                    """
                )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_parent_source_type
            ON parent_sections(source_type)
            """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_parent_source_id
            ON parent_sections(source_id)
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

    def upsert_parents_from_jsonl(
        self,
        parents_file: Path | str = PARENTS_FILE,
    ) -> int:
        rows = []

        with Path(parents_file).open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                if not line.strip():
                    continue

                row = json.loads(line)

                rows.append(
                    (
                        row["parent_id"],
                        row["document_id"],
                        row["parent_index"],
                        row["title"],
                        row["section"],
                        row["source"],
                        row.get(
                            "source_type",
                            "documentation",
                        ),
                        row.get(
                            "source_id",
                            "",
                        ),
                        row.get(
                            "source_url",
                            "",
                        ),
                        row["relative_path"],
                        row["category"],
                        row["doc_type"],
                        row["content"],
                        json.dumps(
                            row.get(
                                "metadata",
                                {},
                            ),
                            ensure_ascii=False,
                        ),
                    )
                )

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
                    source_type,
                    source_id,
                    source_url,
                    relative_path,
                    category,
                    doc_type,
                    content,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(parent_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    parent_index = excluded.parent_index,
                    title = excluded.title,
                    section = excluded.section,
                    source = excluded.source,
                    source_type = excluded.source_type,
                    source_id = excluded.source_id,
                    source_url = excluded.source_url,
                    relative_path = excluded.relative_path,
                    category = excluded.category,
                    doc_type = excluded.doc_type,
                    content = excluded.content,
                    metadata_json = excluded.metadata_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                rows,
            )

        return len(rows)

    def get_parents(
        self,
        parent_ids: Iterable[str],
    ) -> dict[str, dict[str, Any]]:
        parent_ids = list(dict.fromkeys(parent_ids))

        if not parent_ids:
            return {}

        placeholders = ",".join(
            ["?"] * len(parent_ids)
        )

        cursor = self.connection.execute(
            f"""
            SELECT
                parent_id,
                document_id,
                parent_index,
                title,
                section,
                source,
                source_type,
                source_id,
                source_url,
                relative_path,
                category,
                doc_type,
                content,
                metadata_json
            FROM parent_sections
            WHERE parent_id IN ({placeholders})
            """,
            parent_ids,
        )

        results = {}

        for row in cursor.fetchall():
            item = dict(row)

            try:
                item["metadata"] = json.loads(
                    item.pop(
                        "metadata_json",
                        "{}",
                    )
                )
            except json.JSONDecodeError:
                item["metadata"] = {}

            results[item["parent_id"]] = item

        return results

    def count(self) -> int:
        cursor = self.connection.execute(
            "SELECT COUNT(*) AS count FROM parent_sections"
        )
        return int(
            cursor.fetchone()["count"]
        )

    def count_by_source_type(
        self,
        source_type: str,
    ) -> int:
        cursor = self.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM parent_sections
            WHERE source_type = ?
            """,
            (source_type,),
        )
        return int(
            cursor.fetchone()["count"]
        )

    def close(self) -> None:
        self.connection.close()
