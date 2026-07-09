"""
PostgresStorage -- the pgvector-backed implementation of the v0.14 Storage bundle.

A drop-in peer of :class:`librarian.storage.sqlite_storage.SQLiteStorage`: it
satisfies the same ``Storage`` protocol (``metadata`` / ``vectors`` / ``fts`` /
``state`` capability stores plus ``migrate`` / ``transaction`` / write paths) so
the orchestrator and retrieval layer run against it unchanged.

The crucial guarantee carries over verbatim: :meth:`write_upsert` and
:meth:`put_sync_state` issued inside one :meth:`transaction` block commit or
roll back together, so a mid-stream crash never leaves orphan ``chunks`` rows
or a cursor pointing past uncommitted content.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

import librarian.config as config
from librarian.config import POSTGRES_DSN, POSTGRES_SCHEMA
from librarian.storage._common import iso as _iso
from librarian.storage._common import json_default as _json_default
from librarian.storage._common import modality_table as _modality_table
from librarian.storage.postgres.database import (
    PostgresDatabase,
    parse_vector,
    vector_literal,
)
from librarian.storage.postgres.fts_store import PgFTSStore
from librarian.storage.postgres.migrate import migrate as migrate_schema
from librarian.storage.postgres.vector_store import PgVectorStore
from librarian.storage.protocols import SyncState
from librarian.storage.write_models import PreparedChunk, PreparedDocument

logger = logging.getLogger(__name__)

__all__ = ["PostgresStorage", "get_postgres_storage"]


class PostgresStorage:
    """Postgres-backed implementation of the v0.14 ``Storage`` bundle."""

    def __init__(
        self,
        database: PostgresDatabase | None = None,
        dsn: str | None = None,
        schema: str | None = None,
    ) -> None:
        self._db = database or PostgresDatabase(
            dsn=dsn or POSTGRES_DSN, schema=schema or POSTGRES_SCHEMA
        )
        self.metadata = self._db
        self.vectors = PgVectorStore(self._db)
        self.fts = PgFTSStore(self._db)
        # The StateStore protocol is satisfied by this object itself.
        self.state = self

    @property
    def database(self) -> PostgresDatabase:
        return self._db

    def migrate(self) -> None:
        """Create/upgrade the v0.14 Postgres schema (idempotent)."""
        with self._db._connection() as conn:
            migrate_schema(conn, schema=self._db.schema)

    def transaction(self) -> Any:
        """Run a single atomic transaction on one pooled connection.

        Delegates to :meth:`PostgresDatabase.transaction`: content writes and
        the reads/cursor advances issued inside the ``with`` block all join the
        one checked-out connection's transaction, committing or rolling back
        together.
        """
        return self._db.transaction()

    # =========================================================================
    # StateStore
    # =========================================================================

    def get_sync_state(self, source_key: str) -> SyncState | None:
        # Bare read on an autocommit connection: standalone it commits nothing;
        # nested inside a ``transaction()`` block ``_connection()`` hands back
        # that transaction's connection so the read joins it without committing
        # it (the atomicity guarantee).
        with self._db._connection() as conn:
            row = conn.execute(
                "SELECT * FROM sync_state WHERE source_key = %s", (source_key,)
            ).fetchone()
        if row is None:
            return None
        cursor = row["cursor"]
        if isinstance(cursor, str):
            cursor = json.loads(cursor) if cursor else {}
        elif cursor is None:
            cursor = {}
        return SyncState(
            source_key=row["source_key"],
            cursor=cursor,
            status=row["status"],
            last_success_at=row["last_success_at"],
            last_attempt_at=row["last_attempt_at"],
            last_error=row["last_error"],
            documents_seen=row["documents_seen"] or 0,
            chunks_written=row["chunks_written"] or 0,
            config_version=row["config_version"] or 0,
        )

    def put_sync_state(self, state: SyncState, conn: Any = None) -> None:
        """Upsert a sync-state row, joining ``conn``'s transaction when given.

        With no ``conn`` the write runs on a pooled autocommit connection (so it
        commits on its own); with ``conn`` it joins the caller's transaction.
        """
        if conn is not None:
            self._exec_put_sync_state(conn, state)
        else:
            with self._db._connection() as own_conn:
                self._exec_put_sync_state(own_conn, state)

    def _exec_put_sync_state(self, conn: Any, state: SyncState) -> None:
        conn.execute(
            """
            INSERT INTO sync_state (
                source_key, cursor, status, last_success_at, last_attempt_at,
                last_error, documents_seen, chunks_written, config_version
            ) VALUES (%s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s)
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

    def get_file_mtime(self, source_key: str, path: str) -> float | None:
        """Return the last-recorded mtime for a single file, or ``None``.

        Bare read (see :meth:`get_sync_state`): safe to call inside a write
        ``transaction()`` without committing it.
        """
        with self._db._connection() as conn:
            row = conn.execute(
                "SELECT mtime FROM source_file_state WHERE source_key = %s AND path = %s",
                (source_key, path),
            ).fetchone()
        return float(row["mtime"]) if row is not None else None

    def set_file_mtime(
        self,
        source_key: str,
        path: str,
        mtime: float,
        conn: Any = None,
    ) -> None:
        """Upsert one file's mtime, joining ``conn``'s transaction when given."""
        if conn is not None:
            self._exec_set_file_mtime(conn, source_key, path, mtime)
        else:
            with self._db._connection() as own_conn:
                self._exec_set_file_mtime(own_conn, source_key, path, mtime)

    def _exec_set_file_mtime(self, conn: Any, source_key: str, path: str, mtime: float) -> None:
        conn.execute(
            """
            INSERT INTO source_file_state (source_key, path, mtime)
            VALUES (%s, %s, %s)
            ON CONFLICT(source_key, path) DO UPDATE SET
                mtime = excluded.mtime,
                updated_at = now()
            """,
            (source_key, path, mtime),
        )

    # =========================================================================
    # Write paths
    # =========================================================================

    def existing_text_chunks(self, document_id: str) -> dict[str, tuple[str, list[float] | None]]:
        """Return the document's live chunks as ``chunk_id -> (content, text_embedding)``.

        Mirrors SQLiteStorage so the orchestrator can reuse unchanged text
        embeddings before replacing a document's chunks.

        This is a bare read on the autocommit connection, so it does not hold a
        snapshot open across the (potentially slow/network) embed call that
        follows in the orchestrator -- no idle-in-transaction, nothing to block
        VACUUM.
        """
        with self._db._connection() as conn:
            rows = conn.execute(
                """
                SELECT c.chunk_id, c.content, ce.embedding AS embedding
                FROM chunks c
                LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.id
                WHERE c.document_id = (SELECT id FROM documents WHERE document_id = %s)
                  AND c.deleted_at IS NULL
                  AND c.chunk_id IS NOT NULL
                """,
                (document_id,),
            ).fetchall()
        return {row["chunk_id"]: (row["content"], parse_vector(row["embedding"])) for row in rows}

    def write_upsert(self, conn: Any, prepared: PreparedDocument) -> None:
        """Create-or-replace a document and all its chunks within ``conn``'s txn."""
        doc_pk = self._upsert_document_row(conn, prepared)
        self._replace_chunks(conn, doc_pk, prepared)

    def _upsert_document_row(self, conn: Any, prepared: PreparedDocument) -> int:
        metadata_json = (
            json.dumps(prepared.metadata, default=_json_default) if prepared.metadata else None
        )
        row = conn.execute(
            "SELECT id FROM documents WHERE document_id = %s OR path = %s",
            (prepared.document_id, prepared.path),
        ).fetchone()
        if row is not None:
            doc_pk = row["id"]
            conn.execute(
                """
                UPDATE documents
                SET document_id = %s, path = %s, title = %s, content = %s,
                    metadata = %s::jsonb, file_mtime = %s, asset_type = %s,
                    document_source_uri = %s, source_created_at = %s,
                    updated_at = now()
                WHERE id = %s
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
            # Chunk embeddings cascade from the chunks FK.
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_pk,))
            return int(doc_pk)

        row = conn.execute(
            """
            INSERT INTO documents (
                document_id, path, title, content, metadata, file_mtime,
                asset_type, document_source_uri, source_created_at
            ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
            RETURNING id
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
        ).fetchone()
        return int(row["id"])

    # Multi-row INSERTs are bounded by Postgres's 65535 bind-parameter ceiling.
    # The chunk insert binds 13 params/row, so cap each batch well under that;
    # huge documents are written in a handful of round-trips instead of 2N.
    _CHUNK_INSERT_BATCH = 1000

    def _replace_chunks(self, conn: Any, doc_pk: int, prepared: PreparedDocument) -> None:
        """Insert all of a document's chunks + embeddings in bulk.

        Per batch: one multi-row ``INSERT ... RETURNING`` for the chunks, then
        one bulk insert per embedding table (down from the old 2N round-trips).
        The chunk insert returns ``(id, chunk_index)`` so the new primary keys
        can be matched back to each prepared chunk by its document-unique
        ``chunk_index`` -- never relying on RETURNING preserving VALUES order,
        which Postgres does not guarantee. Batched so a document with thousands
        of chunks stays under the bind-parameter ceiling.
        """
        chunks = prepared.chunks
        for start in range(0, len(chunks), self._CHUNK_INSERT_BATCH):
            self._insert_chunk_batch(
                conn, doc_pk, prepared, chunks[start : start + self._CHUNK_INSERT_BATCH]
            )

    def _insert_chunk_batch(
        self, conn: Any, doc_pk: int, prepared: PreparedDocument, batch: list[PreparedChunk]
    ) -> None:
        if not batch:
            return

        chunk_columns = (
            "document_id, chunk_id, content, heading_path, chunk_index, "
            "start_char, end_char, asset_type, modality, document_size, "
            "source_created_at, document_source_uri, chunk_source_uri"
        )
        source_created_at = _iso(prepared.source_created_at)
        values_sql: list[str] = []
        params: list[Any] = []
        for chunk in batch:
            values_sql.append("(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
            params.extend((
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
                source_created_at,
                prepared.document_source_uri,
                chunk.chunk_source_uri,
            ))
        rows = conn.execute(
            f"INSERT INTO chunks ({chunk_columns}) VALUES {', '.join(values_sql)} "  # noqa: S608 - fixed column list + %s placeholders
            "RETURNING id, chunk_index",
            params,
        ).fetchall()
        pk_by_index = {row["chunk_index"]: row["id"] for row in rows}
        # Re-pairing embeddings to PKs by chunk_index only works if chunk_index is
        # unique within the batch; a preparer emitting duplicates would silently
        # mispair embeddings. Fail loudly instead (no UNIQUE constraint enforces
        # this at the schema level yet).
        if len(pk_by_index) != len(batch):
            raise ValueError(
                "Duplicate chunk_index within one document; chunk_index must be "
                "unique per document for embedding re-pairing to be correct."
            )

        # Group embedding rows by their destination table, then one bulk insert
        # each. chunk_embeddings carries model_version; the code/vision tables do
        # not, so they are batched separately.
        text_rows: list[tuple[int, str, str | None]] = []
        other_rows: dict[str, list[tuple[int, str]]] = {}
        for chunk in batch:
            if not chunk.embedding:
                continue
            chunk_pk = pk_by_index[chunk.chunk_index]
            literal = vector_literal(chunk.embedding)
            table = _modality_table(chunk.modality)
            if table == "chunk_embeddings":
                text_rows.append((chunk_pk, literal, chunk.model_version))
            else:
                other_rows.setdefault(table, []).append((chunk_pk, literal))

        if text_rows:
            placeholders = ", ".join("(%s, %s::vector, %s)" for _ in text_rows)
            params = [field for row in text_rows for field in row]
            cols = "chunk_id, embedding, model_version"
            sql = f"INSERT INTO chunk_embeddings ({cols}) VALUES {placeholders}"  # noqa: S608 - placeholders are %s tuples only
            conn.execute(sql, params)
        for table, table_rows in other_rows.items():
            placeholders = ", ".join("(%s, %s::vector)" for _ in table_rows)
            params = [field for row in table_rows for field in row]
            sql = f"INSERT INTO {table} (chunk_id, embedding) VALUES {placeholders}"  # noqa: S608 - table is a fixed internal literal
            conn.execute(sql, params)

    def soft_delete_document(self, conn: Any, document_id: str, reason: str | None) -> int:
        """Tombstone all chunks of a document. Returns the number tombstoned."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            """
            UPDATE chunks
            SET deleted_at = %s, deletion_reason = %s
            WHERE document_id = (SELECT id FROM documents WHERE document_id = %s)
              AND deleted_at IS NULL
            """,
            (now, reason, document_id),
        )
        return int(cursor.rowcount)

    def delete_document_by_path(self, path: str) -> bool:
        """Hard-delete a document and its chunks/embeddings by path.

        The admin removal path (``remove_from_library``). Chunks, embeddings and
        the FTS ``content_tsv`` all hang off the document via ``ON DELETE
        CASCADE`` (the tsvector is a generated column on ``chunks``), so a single
        ``DELETE`` on ``documents`` removes everything. Runs on the autocommit
        connection, so it commits on its own.
        """
        with self._db._connection() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE path = %s", (path,))
            return int(cursor.rowcount) > 0


_storage_instances: dict[tuple[str, str], PostgresStorage] = {}


def get_postgres_storage() -> PostgresStorage:
    """Get a process-wide :class:`PostgresStorage` (migrated to v0.14).

    The instance is cached on the active ``(POSTGRES_DSN, POSTGRES_SCHEMA)`` pair
    rather than as a single global. Reading config dynamically means a test that
    isolates substrates per-schema -- or a deployment that rotates its DSN -- gets
    a fresh, correctly-targeted instance instead of silently reusing the first
    schema/DSN bound at startup.
    """
    dsn = config.POSTGRES_DSN
    schema = config.POSTGRES_SCHEMA
    key = (dsn or "", schema)
    instance = _storage_instances.get(key)
    if instance is None:
        instance = PostgresStorage(dsn=dsn, schema=schema)
        instance.migrate()
        _storage_instances[key] = instance
    return instance
