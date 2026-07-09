"""
SQLiteStorage -- the default concrete implementation of the v0.14 Storage bundle.

Wraps the existing :class:`~librarian.storage.database.Database`,
:class:`~librarian.storage.vector_store.VectorStore` and
:class:`~librarian.storage.fts_store.FTSStore` (which provide the read-side
protocols) and adds:

* :meth:`migrate` -- run the v0.14 schema migration.
* :meth:`transaction` -- atomic content + cursor writes on one connection.
* sync-state persistence (the ``StateStore`` protocol).
* :meth:`write_upsert` / :meth:`soft_delete_document` -- the write paths the
  orchestrator drives.

All writes go through raw SQL on the transaction connection rather than through
``Database``'s per-call mutators, because those commit eagerly and would break
the "content + cursor advance in one transaction" guarantee.
"""

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

from librarian.storage._common import iso as _iso
from librarian.storage._common import json_default as _json_default
from librarian.storage._common import modality_table as _modality_table
from librarian.storage.database import (
    Database,
    deserialize_embedding,
    get_database,
    serialize_embedding,
)
from librarian.storage.fts_store import FTSStore
from librarian.storage.migrate import migrate as migrate_schema
from librarian.storage.protocols import SyncState
from librarian.storage.vector_store import VectorStore
from librarian.storage.write_models import PreparedDocument

logger = logging.getLogger(__name__)

__all__ = ["SQLiteStorage", "get_storage"]


class SQLiteStorage:
    """SQLite-backed implementation of the v0.14 ``Storage`` bundle."""

    def __init__(self, database: Database | None = None) -> None:
        self._db = database or get_database()
        self.metadata = self._db
        self.vectors = VectorStore(self._db)
        self.fts = FTSStore(self._db)
        # The StateStore protocol is satisfied by this object itself.
        self.state = self

    @property
    def database(self) -> Database:
        return self._db

    def migrate(self) -> None:
        """Run the v0.14 schema migration against the underlying database."""
        conn = self._db._get_connection()
        migrate_schema(conn)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield the thread-local connection as a single atomic transaction.

        Commits on clean exit, rolls back on exception. Content writes and
        cursor advances issued within the same ``with`` block commit or roll
        back together.
        """
        conn = self._db._get_connection()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()

    # =========================================================================
    # StateStore
    # =========================================================================

    def get_sync_state(self, source_key: str) -> SyncState | None:
        conn = self._db._get_connection()
        row = conn.execute(
            "SELECT * FROM sync_state WHERE source_key = ?", (source_key,)
        ).fetchone()
        if row is None:
            return None
        return SyncState(
            source_key=row["source_key"],
            cursor=json.loads(row["cursor"]) if row["cursor"] else {},
            status=row["status"],
            last_success_at=row["last_success_at"],
            last_attempt_at=row["last_attempt_at"],
            last_error=row["last_error"],
            documents_seen=row["documents_seen"] or 0,
            chunks_written=row["chunks_written"] or 0,
            config_version=row["config_version"] or 0,
        )

    def put_sync_state(self, state: SyncState, conn: sqlite3.Connection | None = None) -> None:
        """Upsert a sync-state row.

        When ``conn`` is provided the write joins the caller's transaction (so a
        cursor advance commits atomically with the content it describes).
        """
        own = conn is None
        conn = conn or self._db._get_connection()
        conn.execute(
            """
            INSERT INTO sync_state (
                source_key, cursor, status, last_success_at, last_attempt_at,
                last_error, documents_seen, chunks_written, config_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_key) DO UPDATE SET
                cursor = excluded.cursor,
                status = excluded.status,
                last_success_at = excluded.last_success_at,
                last_attempt_at = excluded.last_attempt_at,
                last_error = excluded.last_error,
                documents_seen = excluded.documents_seen,
                chunks_written = excluded.chunks_written,
                config_version = excluded.config_version
            """,
            (
                state.source_key,
                json.dumps(state.cursor),
                state.status,
                _iso(state.last_success_at)
                if isinstance(state.last_success_at, datetime)
                else state.last_success_at,
                _iso(state.last_attempt_at)
                if isinstance(state.last_attempt_at, datetime)
                else state.last_attempt_at,
                state.last_error,
                state.documents_seen,
                state.chunks_written,
                state.config_version,
            ),
        )
        if own:
            conn.commit()

    def get_file_mtime(self, source_key: str, path: str) -> float | None:
        """Return the last-recorded mtime for a single file, or ``None``."""
        conn = self._db._get_connection()
        row = conn.execute(
            "SELECT mtime FROM source_file_state WHERE source_key = ? AND path = ?",
            (source_key, path),
        ).fetchone()
        return row["mtime"] if row is not None else None

    def set_file_mtime(
        self,
        source_key: str,
        path: str,
        mtime: float,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Upsert one file's mtime as a single indexed row (O(1), no full rewrite).

        When ``conn`` is provided the write joins the caller's transaction so the
        per-file cursor advances atomically with the content it describes.
        """
        own = conn is None
        conn = conn or self._db._get_connection()
        conn.execute(
            """
            INSERT INTO source_file_state (source_key, path, mtime)
            VALUES (?, ?, ?)
            ON CONFLICT(source_key, path) DO UPDATE SET mtime = excluded.mtime
            """,
            (source_key, path, mtime),
        )
        if own:
            conn.commit()

    # =========================================================================
    # Write paths
    # =========================================================================

    def existing_text_chunks(self, document_id: str) -> dict[str, tuple[str, list[float] | None]]:
        """Return the document's live chunks as ``chunk_id -> (content, text_embedding)``.

        Only non-deleted chunks are included; the embedding is the stored
        TEXT-modality vector, or ``None`` when the chunk has no text embedding.
        CODE/VISION embeddings are not returned.
        """
        conn = self._db._get_connection()
        rows = conn.execute(
            """
            SELECT c.chunk_id, c.content, ce.embedding AS embedding
            FROM chunks c
            LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.id
            WHERE c.document_id = (SELECT id FROM documents WHERE document_id = ?)
              AND c.deleted_at IS NULL
              AND c.chunk_id IS NOT NULL
            """,
            (document_id,),
        ).fetchall()
        result: dict[str, tuple[str, list[float] | None]] = {}
        for row in rows:
            embedding = deserialize_embedding(row["embedding"]) if row["embedding"] else None
            result[row["chunk_id"]] = (row["content"], embedding)
        return result

    def write_upsert(self, conn: sqlite3.Connection, prepared: PreparedDocument) -> None:
        """Create-or-replace a document and all its chunks within ``conn``'s txn."""
        doc_pk = self._upsert_document_row(conn, prepared)
        self._replace_chunks(conn, doc_pk, prepared)

    def _upsert_document_row(self, conn: sqlite3.Connection, prepared: PreparedDocument) -> int:
        metadata_json = (
            json.dumps(prepared.metadata, default=_json_default) if prepared.metadata else None
        )
        row = conn.execute(
            "SELECT id FROM documents WHERE document_id = ? OR path = ?",
            (prepared.document_id, prepared.path),
        ).fetchone()
        if row is not None:
            doc_pk = row["id"]
            conn.execute(
                """
                UPDATE documents
                SET document_id = ?, path = ?, title = ?, content = ?, metadata = ?,
                    file_mtime = ?, asset_type = ?, document_source_uri = ?,
                    source_created_at = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    prepared.document_id,
                    prepared.path,
                    prepared.title,
                    prepared.content,
                    metadata_json,
                    prepared.file_mtime,
                    prepared.asset_type.value,
                    prepared.document_source_uri,
                    _iso(prepared.source_created_at),
                    doc_pk,
                ),
            )
            self._delete_chunks_for_document(conn, doc_pk)
            return int(doc_pk)

        cursor = conn.execute(
            """
            INSERT INTO documents (
                document_id, path, title, content, metadata, file_mtime,
                asset_type, document_source_uri, source_created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prepared.document_id,
                prepared.path,
                prepared.title,
                prepared.content,
                metadata_json,
                prepared.file_mtime,
                prepared.asset_type.value,
                prepared.document_source_uri,
                _iso(prepared.source_created_at),
            ),
        )
        return int(cursor.lastrowid)  # type: ignore[arg-type]

    def _delete_chunks_for_document(self, conn: sqlite3.Connection, doc_pk: int) -> None:
        for table in ("chunk_embeddings", "vec_chunks_code", "vec_chunks_vision"):
            conn.execute(
                f"DELETE FROM {table} WHERE chunk_id IN "  # noqa: S608 - table literal
                "(SELECT id FROM chunks WHERE document_id = ?)",
                (doc_pk,),
            )
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_pk,))

    def _replace_chunks(
        self, conn: sqlite3.Connection, doc_pk: int, prepared: PreparedDocument
    ) -> None:
        for chunk in prepared.chunks:
            cursor = conn.execute(
                """
                INSERT INTO chunks (
                    document_id, chunk_id, content, heading_path, chunk_index,
                    start_char, end_char, asset_type, modality, document_size,
                    source_created_at, document_source_uri, chunk_source_uri
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_pk,
                    chunk.chunk_id,
                    chunk.content,
                    chunk.heading_path,
                    chunk.chunk_index,
                    chunk.start_char,
                    chunk.end_char,
                    chunk.asset_type.value,
                    chunk.modality.value,
                    prepared.document_size,
                    _iso(prepared.source_created_at),
                    prepared.document_source_uri,
                    chunk.chunk_source_uri,
                ),
            )
            chunk_pk = cursor.lastrowid
            if chunk.embedding:
                table = _modality_table(chunk.modality)
                if table == "chunk_embeddings":
                    conn.execute(
                        "INSERT INTO chunk_embeddings (chunk_id, embedding, model_version) "
                        "VALUES (?, ?, ?)",
                        (chunk_pk, serialize_embedding(chunk.embedding), chunk.model_version),
                    )
                else:
                    conn.execute(
                        f"INSERT INTO {table} (chunk_id, embedding) VALUES (?, ?)",  # noqa: S608
                        (chunk_pk, serialize_embedding(chunk.embedding)),
                    )

    def soft_delete_document(
        self, conn: sqlite3.Connection, document_id: str, reason: str | None
    ) -> int:
        """Tombstone all chunks of a document. Returns the number tombstoned."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            """
            UPDATE chunks
            SET deleted_at = ?, deletion_reason = ?
            WHERE document_id = (SELECT id FROM documents WHERE document_id = ?)
              AND deleted_at IS NULL
            """,
            (now, reason, document_id),
        )
        return cursor.rowcount

    def delete_document_by_path(self, path: str) -> bool:
        """Hard-delete a document and its chunks/embeddings by path.

        The admin removal path (``remove_from_library``): drops the document row
        plus every chunk, embedding, and FTS entry. FTS rows are cleaned by the
        ``AFTER DELETE ON chunks`` trigger; vector rows by the explicit deletes
        in :meth:`_delete_chunks_for_document`.
        """
        conn = self._db._get_connection()
        row = conn.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()
        if row is None:
            return False
        doc_pk = int(row["id"])
        self._delete_chunks_for_document(conn, doc_pk)
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_pk,))
        conn.commit()
        return True


_storage_instance: SQLiteStorage | None = None


def get_storage() -> SQLiteStorage:
    """Get a process-wide :class:`SQLiteStorage` (migrated to v0.14)."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = SQLiteStorage()
        _storage_instance.migrate()
    return _storage_instance
