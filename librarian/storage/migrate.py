"""
v0.14 schema migration.

This is the canonical creator of the v0.14 librarian-owned schema. It runs on
top of the base tables created by ``Database._init_schema`` and the legacy
multi-modal migration, and brings the database up to the v0.14 shape:

* ``documents`` gains ``document_id`` (deterministic hash), ``document_source_uri``
  and ``source_created_at``.
* ``chunks`` gains ``chunk_id`` (deterministic hash), ``document_size``,
  ``source_created_at``, ``deleted_at``, ``deletion_reason``,
  ``document_source_uri`` and ``chunk_source_uri``.
* ``chunk_embeddings`` gains a ``model_version`` auxiliary column.
* a ``sync_state`` table tracks connector cursors.
* a ``source_file_state`` table holds one indexed mtime row per file so the
  file-mode ingest path advances its cursor in O(1) per file.

The on-disk v0.13 schema is *not* migrated in place: v0.14 detects it on startup
and asks the user to rebuild (see the detect-and-rebuild flow). This module only
brings a fresh or already-v0.14 database to the current shape, additively.

Note on naming: the ``chunk_embeddings`` virtual table keeps a column literally
named ``chunk_id`` that holds the integer ``chunks.id`` surrogate (used as the
vec0 / FTS5 rowid). The *public* deterministic identity lives in the new
``chunks.chunk_id`` TEXT column. The two never collide in SQL because they are in
different tables.
"""

import logging
import sqlite3

logger = logging.getLogger(__name__)

# New flat columns, keyed by table.
_DOCUMENT_COLUMNS: dict[str, str] = {
    "document_id": "TEXT",
    "document_source_uri": "TEXT",
    "source_created_at": "TEXT",
}

_CHUNK_COLUMNS: dict[str, str] = {
    "chunk_id": "TEXT",
    "document_size": "INTEGER",
    "source_created_at": "TEXT",
    "deleted_at": "TEXT",
    "deletion_reason": "TEXT",
    "document_source_uri": "TEXT",
    "chunk_source_uri": "TEXT",
    # Per-chunk type-specific metadata as JSON. Holds the VLM vision pipeline's
    # ``processing_status`` (and caption/transcription) so failed image chunks
    # can be found and retried by ``libr reprocess``.
    "modality_data": "JSON",
}


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cursor.fetchall()}


def _add_missing_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = _existing_columns(conn, table)
    for name, col_type in columns.items():
        if name not in existing:
            logger.info("v0.14 migrate: adding %s.%s", table, name)
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")


def _embeddings_has_model_version(conn: sqlite3.Connection) -> bool:
    """Probe whether ``chunk_embeddings`` already has a ``model_version`` column."""
    try:
        conn.execute("SELECT model_version FROM chunk_embeddings LIMIT 0")
    except sqlite3.OperationalError:
        return False
    else:
        return True


def _vec_dimension(conn: sqlite3.Connection) -> int:
    """Read the embedding dimension declared on the existing chunk_embeddings table."""
    import re

    row = conn.execute("SELECT sql FROM sqlite_master WHERE name='chunk_embeddings'").fetchone()
    if row and row[0]:
        match = re.search(r"float\[(\d+)\]", row[0])
        if match:
            return int(match.group(1))
    # Fall back to the configured provider dimension.
    from librarian.storage.database import get_effective_embedding_dimension

    return get_effective_embedding_dimension()


def _ensure_embeddings_model_version(conn: sqlite3.Connection) -> None:
    """Ensure ``chunk_embeddings`` carries a ``model_version`` auxiliary column.

    sqlite-vec vec0 tables cannot be ``ALTER``-ed, so when the column is missing
    we recreate the table. This is only safe when the table is empty; a populated
    v0.13 table is handled by the detect-and-rebuild flow (which wipes first), so
    by the time this runs the table is empty.
    """
    if _embeddings_has_model_version(conn):
        return

    count = conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
    if count:
        # Populated v0.13 table: cannot ALTER a vec0 table, and recreating would
        # drop data. This is the detect-and-rebuild boundary -- fail loudly so the
        # user runs `libr index --rebuild` rather than half-migrating.
        from librarian.storage.schema_version import (
            REBUILD_MESSAGE,
            SchemaRebuildRequired,
        )

        raise SchemaRebuildRequired(REBUILD_MESSAGE)

    dim = _vec_dimension(conn)
    logger.info("v0.14 migrate: recreating chunk_embeddings with model_version (dim=%d)", dim)
    conn.execute("DROP TABLE IF EXISTS chunk_embeddings")
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE chunk_embeddings USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding float[{dim}] distance_metric=cosine,
            +model_version TEXT
        )
        """
    )


def _create_sync_state(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sync_state (
            source_key TEXT PRIMARY KEY,
            cursor JSON,
            status TEXT DEFAULT 'idle',
            last_success_at TEXT,
            last_attempt_at TEXT,
            last_error TEXT,
            documents_seen INTEGER DEFAULT 0,
            chunks_written INTEGER DEFAULT 0,
            config_version INTEGER DEFAULT 0
        )
        """
    )


def _create_source_file_state(conn: sqlite3.Connection) -> None:
    """Per-file mtime cursor, one indexed row per (source, path)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS source_file_state (
            source_key TEXT NOT NULL,
            path TEXT NOT NULL,
            mtime REAL,
            PRIMARY KEY (source_key, path)
        )
        """
    )


def migrate(conn: sqlite3.Connection) -> None:
    """Bring ``conn``'s database to the v0.14 schema (idempotent).

    Args:
        conn: An open SQLite connection with sqlite-vec loaded.
    """
    logger.info("Running v0.14 schema migration")
    _add_missing_columns(conn, "documents", _DOCUMENT_COLUMNS)
    _add_missing_columns(conn, "chunks", _CHUNK_COLUMNS)

    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_document_id ON documents(document_id)"
    )
    # Supports list_documents' ``ORDER BY updated_at DESC, id DESC`` so a
    # paginated read is an index-ordered scan rather than a full scan + sort.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_updated_at_id "
        "ON documents(updated_at DESC, id DESC)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_deleted_at ON chunks(deleted_at)")

    _ensure_embeddings_model_version(conn)
    _create_sync_state(conn)
    _create_source_file_state(conn)
    conn.commit()
    logger.info("v0.14 schema migration complete")
