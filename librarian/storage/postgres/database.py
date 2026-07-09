"""
PostgresDatabase -- connection management and read-side metadata access.

Mirrors the read surface of :class:`librarian.storage.database.Database` (the
SQLite metadata store) so the retrieval and MCP layers are substrate-agnostic.
Connections come from a bounded :class:`psycopg_pool.ConnectionPool` (capped at
``POSTGRES_POOL_MAX_SIZE``): a threaded MCP/HTTP server checks connections out
of the pool for the duration of one operation rather than opening one per
thread, so request fan-out can't drift toward the server's ``max_connections``.
Every pooled connection runs in ``autocommit`` mode and pins ``search_path`` to
the configured schema; explicit multi-statement atomicity is opened on demand
via :meth:`transaction` (a real ``conn.transaction()`` block). Autocommit means
a bare read never leaves an idle-in-transaction snapshot open, and never
prematurely commits an in-flight write transaction it happens to be nested
inside.

During a :meth:`transaction` the checked-out connection is parked on a
thread-local so the reads/writes issued inside the ``with`` block all join that
one transaction (the atomicity guarantee) instead of borrowing a second,
independent connection from the pool.

Connections also set ``connect_timeout`` (unreachable host fails fast),
``statement_timeout`` (pathological query is bounded) and
``idle_in_transaction_session_timeout`` (a stuck transaction can't pin a
connection forever) -- all configurable via ``POSTGRES_*`` settings.

Embeddings are exchanged as pgvector text literals (``[1,2,3]``) and cast in
SQL (``%s::vector``). This keeps the backend working with only ``psycopg``
installed and sidesteps the extension-OID lookup ordering that
``pgvector.psycopg.register_vector`` requires on a freshly created database.
"""

import json
import logging
import math
import threading
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any

from librarian.config import (
    POSTGRES_CONNECT_TIMEOUT,
    POSTGRES_DSN,
    POSTGRES_IDLE_TX_TIMEOUT_MS,
    POSTGRES_POOL_MAX_SIZE,
    POSTGRES_POOL_MIN_SIZE,
    POSTGRES_POOL_TIMEOUT,
    POSTGRES_SCHEMA,
    POSTGRES_STATEMENT_TIMEOUT_MS,
)
from librarian.storage._common import list_documents_query
from librarian.types import AssetType, Document

if TYPE_CHECKING:
    from librarian.storage.protocols import ChunkContext

logger = logging.getLogger(__name__)

__all__ = ["PostgresDatabase", "vector_literal"]


def _require_psycopg_pool() -> Any:
    """Import psycopg_pool, raising a clear error if the ``postgres`` extra is absent."""
    try:
        import psycopg_pool
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "PostgresStorage requires the 'postgres' extra. Install it with:\n"
            "    uv pip install -e '.[postgres]'\n"
            "or set STORAGE_BACKEND=sqlite to use the default backend."
        ) from e
    return psycopg_pool


def vector_literal(embedding: list[float]) -> str:
    """Render an embedding as a pgvector text literal (``[0.1,0.2,...]``).

    Raises ``ValueError`` on non-finite components: pgvector rejects ``inf`` /
    ``nan`` at insert time, so we fail loudly and uniformly here rather than
    diverging from sqlite-vec (which would store the raw bits silently).
    """
    parts: list[str] = []
    for x in embedding:
        value = float(x)
        if not math.isfinite(value):
            raise ValueError(
                "Embedding contains a non-finite value (inf/nan); pgvector "
                "rejects these. Check the embedding model output."
            )
        parts.append(repr(value))
    return "[" + ",".join(parts) + "]"


def parse_vector(value: Any) -> list[float] | None:
    """Parse a pgvector value (text literal or list) back into floats."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    text = str(value).strip().strip("[]")
    if not text:
        return None
    return [float(x) for x in text.split(",")]


def _as_metadata(value: Any) -> dict[str, Any]:
    """Normalize a JSONB column (psycopg returns dict) into a plain dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    parsed = json.loads(value)
    return dict(parsed) if isinstance(parsed, dict) else {}


class PostgresDatabase:
    """Postgres connection manager + read-side metadata store."""

    def __init__(self, dsn: str | None = None, schema: str | None = None) -> None:
        self.dsn = dsn or POSTGRES_DSN
        if not self.dsn:
            raise ValueError(
                "Postgres backend selected but no connection string configured. "
                "Set POSTGRES_DSN (or DATABASE_URL)."
            )
        self.schema = schema or POSTGRES_SCHEMA
        # The bounded pool is created lazily on first use (so constructing a
        # PostgresDatabase never dials out). ``_local.txn_conn`` holds the
        # connection checked out for the duration of an open transaction(), so
        # reads issued inside that block join the same transaction instead of
        # borrowing a second connection from the pool.
        self._pool: Any = None
        # Guards lazy pool creation: without it, threads racing the first
        # checkout (the exact MCP/HTTP fan-out this pool exists for) could each
        # build a ConnectionPool and leak all but the last.
        self._pool_lock = threading.Lock()
        self._local = threading.local()

    # =========================================================================
    # Connection management
    # =========================================================================

    def _configure_connection(self, conn: Any) -> None:
        """Pin search_path + timeouts on a freshly created pooled connection.

        Run once per physical connection (psycopg_pool's ``configure`` hook), not
        per checkout: these are session-level settings that persist for the
        connection's lifetime.
        """
        from psycopg import sql

        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.schema))
            )
            # Bound runaway queries and stuck transactions. ``0`` disables either.
            cur.execute(
                sql.SQL("SET statement_timeout = {}").format(
                    sql.Literal(POSTGRES_STATEMENT_TIMEOUT_MS)
                )
            )
            cur.execute(
                sql.SQL("SET idle_in_transaction_session_timeout = {}").format(
                    sql.Literal(POSTGRES_IDLE_TX_TIMEOUT_MS)
                )
            )

    def _get_pool(self) -> Any:
        """Get or lazily create the bounded connection pool (thread-safe)."""
        pool = self._pool
        if pool is not None and not pool.closed:
            return pool

        with self._pool_lock:
            # Double-check inside the lock: another thread may have built it
            # while we waited.
            if self._pool is not None and not self._pool.closed:
                return self._pool

            psycopg_pool = _require_psycopg_pool()
            from psycopg.rows import dict_row

            connect_kwargs: dict[str, Any] = {"autocommit": True, "row_factory": dict_row}
            if POSTGRES_CONNECT_TIMEOUT > 0:
                connect_kwargs["connect_timeout"] = POSTGRES_CONNECT_TIMEOUT

            # Note: a bad DSN/credential no longer fails on the first bare read
            # with the driver's OperationalError -- a checkout now waits up to
            # POSTGRES_POOL_TIMEOUT and raises psycopg_pool.PoolTimeout, with the
            # underlying connection error in the pool's worker logs.
            self._pool = psycopg_pool.ConnectionPool(
                self.dsn,
                min_size=POSTGRES_POOL_MIN_SIZE,
                max_size=POSTGRES_POOL_MAX_SIZE,
                timeout=POSTGRES_POOL_TIMEOUT,
                kwargs=connect_kwargs,
                configure=self._configure_connection,
                open=True,
            )
            return self._pool

    @contextmanager
    def _connection(self) -> Generator[Any, None, None]:
        """Yield a connection for a (read-side) statement group.

        Inside an open :meth:`transaction` this yields that transaction's
        connection (so the read joins it without committing it). Otherwise it
        borrows a connection from the pool for the duration of the ``with`` block
        and returns it afterward. The connection is in ``autocommit`` mode, so
        individual reads commit immediately and never leave an idle transaction
        open.
        """
        txn_conn = getattr(self._local, "txn_conn", None)
        if txn_conn is not None:
            # Join the in-flight transaction; its owner (transaction()) returns
            # the connection to the pool on commit/rollback.
            yield txn_conn
            return

        with self._get_pool().connection() as conn:
            yield conn

    @contextmanager
    def transaction(self) -> Generator[Any, None, None]:
        """Check out one pooled connection and run a single atomic transaction.

        Opens a real ``conn.transaction()`` block on the autocommit connection:
        commits on clean exit, rolls back on exception. The connection is parked
        on a thread-local for the duration so reads/writes issued inside the
        block (e.g. ``get_sync_state``, ``write_upsert``) all run in this one
        transaction and commit or roll back together. The pool reclaims the
        connection when the block exits.

        Not re-entrant: a nested ``transaction()`` would check out a second
        connection (self-deadlock risk, guaranteed ``PoolTimeout`` at
        ``max_size=1``) and, worse, its ``finally`` would unpark the connection
        so the outer block's remaining writes silently commit independently --
        breaking the content+cursor atomicity guarantee with no error. We raise
        instead. (No current caller nests; this keeps a future one honest.)
        """
        if getattr(self._local, "txn_conn", None) is not None:
            raise RuntimeError(
                "PostgresDatabase.transaction() is not re-entrant: a transaction "
                "is already open on this thread. Pass the existing connection to "
                "the write methods instead of opening a nested transaction."
            )
        with self._get_pool().connection() as conn:
            self._local.txn_conn = conn
            try:
                with conn.transaction():
                    yield conn
            finally:
                self._local.txn_conn = None

    def close(self) -> None:
        """Close the connection pool, if one was opened."""
        if self._pool is not None and not self._pool.closed:
            self._pool.close()
        self._pool = None

    # =========================================================================
    # Document reads (MetadataStore protocol)
    # =========================================================================

    def _row_to_document(self, row: dict[str, Any]) -> Document:
        asset_type = AssetType(row["asset_type"]) if row.get("asset_type") else AssetType.TEXT
        return Document(
            id=row["id"],
            path=row["path"],
            title=row["title"],
            content=row["content"],
            metadata=_as_metadata(row.get("metadata")),
            asset_type=asset_type,
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            file_mtime=row.get("file_mtime"),
        )

    def get_document_by_path(self, path: str) -> Document | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE path = %s", (path,)).fetchone()
            return self._row_to_document(row) if row else None

    def get_document_by_id(self, doc_id: int) -> Document | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = %s", (doc_id,)).fetchone()
            return self._row_to_document(row) if row else None

    def list_documents(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Document]:
        """List documents newest-first, optionally windowed and paginated.

        Mirrors :meth:`librarian.storage.database.Database.list_documents` via
        the shared :func:`~librarian.storage._common.list_documents_query`
        builder: the same ``updated_at DESC, id DESC`` order and ``limit`` /
        ``offset`` bounds, so a paginated read returns the same window on either
        substrate. Postgres binds datetimes directly (no isoformat).
        """
        sql, params = list_documents_query(
            placeholder="%s",
            unbounded_limit="ALL",
            format_date=lambda d: d,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )
        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_document(row) for row in rows]

    def get_document_ids_in_timerange(self, start_date: datetime, end_date: datetime) -> list[int]:
        start_ts = start_date.timestamp()
        end_ts = end_date.timestamp()
        with self._connection() as conn:
            # Two index-friendly OR'd predicates rather than a CASE in the WHERE
            # (which is non-sargable and forces a full scan). Postgres can plan
            # each branch against an index on file_mtime / updated_at.
            rows = conn.execute(
                """
                SELECT id FROM documents
                WHERE (file_mtime IS NOT NULL AND file_mtime >= %s AND file_mtime < %s)
                   OR (file_mtime IS NULL AND updated_at >= %s AND updated_at < %s)
                """,
                (start_ts, end_ts, start_date, end_date),
            ).fetchall()
            return [row["id"] for row in rows]

    def get_chunk_public_fields(self, chunk_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Fetch v0.14 public chunk fields keyed by internal ``chunks.id``."""
        if not chunk_ids:
            return {}
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, chunk_id, chunk_index, document_size,
                       source_created_at, chunk_source_uri, deleted_at,
                       modality_data
                FROM chunks WHERE id = ANY(%s)
                """,
                (list(chunk_ids),),
            ).fetchall()
        return {
            row["id"]: {
                "chunk_id": row["chunk_id"],
                "chunk_index": row["chunk_index"],
                "document_size": row["document_size"],
                "source_created_at": row["source_created_at"],
                "chunk_source_uri": row["chunk_source_uri"],
                "deleted_at": row["deleted_at"],
                # psycopg adapts a ``jsonb`` column straight to a dict (or None).
                "modality_data": row["modality_data"],
            }
            for row in rows
        }

    def get_chunk_context(
        self,
        chunk_id: str,
        before: int = 2,
        after: int = 2,
        include_deleted: bool = False,
    ) -> "list[ChunkContext]":
        """Postgres parity for :meth:`Database.get_chunk_context`.

        Same window semantics as the SQLite backend: locate the anchor by public
        ``chunk_id`` (TEXT), then return the chunks in
        ``[anchor - before, anchor + after]`` excluding the anchor, in source
        order. Soft-deleted neighbors are included only when ``include_deleted``
        is set.
        """
        from librarian.storage._common import chunk_context_from_row, deleted_filter

        anchor = self._find_anchor_chunk(chunk_id)
        if anchor is None:
            return []
        doc_pk, anchor_index = anchor

        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    c.id AS internal_id,
                    c.chunk_id AS chunk_id,
                    c.document_id AS document_id,
                    c.content AS content,
                    c.heading_path AS heading_path,
                    c.chunk_index AS chunk_index,
                    c.asset_type AS asset_type,
                    c.chunk_source_uri AS chunk_source_uri,
                    c.deleted_at AS deleted_at,
                    d.path AS document_path
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.document_id = %s
                    AND c.chunk_index BETWEEN %s AND %s
                    AND c.chunk_index != %s
                    {deleted_filter(include_deleted)}
                ORDER BY c.chunk_index ASC
                """,  # noqa: S608 - deleted_filter returns a fixed internal literal
                (doc_pk, anchor_index - before, anchor_index + after, anchor_index),
            ).fetchall()

        return [chunk_context_from_row(row) for row in rows]

    def _find_anchor_chunk(self, chunk_id: str) -> tuple[int, int] | None:
        """Resolve ``chunk_id`` to its ``(document_id, chunk_index)`` or ``None``.

        Matches only the deterministic public ``chunk_id`` (TEXT). Postgres is a
        v0.14-only substrate, so legacy NULL-``chunk_id`` rows can't exist by
        construction -- there is no integer-surrogate fallback to diverge from
        SQLite over.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT document_id, chunk_index FROM chunks WHERE chunk_id = %s",
                (chunk_id,),
            ).fetchone()
        if row is None:
            return None
        return row["document_id"], row["chunk_index"]

    def chunk_exists(self, chunk_id: str) -> bool:
        """True if a chunk with this public ``chunk_id`` exists (deleted or not)."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE chunk_id = %s LIMIT 1", (chunk_id,)
            ).fetchone()
        return row is not None

    def get_stats(self) -> dict[str, Any]:
        with self._connection() as conn:
            doc_count = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            chunk_count = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
            embedding_count = conn.execute("SELECT COUNT(*) AS n FROM chunk_embeddings").fetchone()[
                "n"
            ]
            return {
                "document_count": doc_count,
                "chunk_count": chunk_count,
                "embedding_count": embedding_count,
                "database_path": self._sanitized_dsn(),
            }

    def _sanitized_dsn(self) -> str:
        """A display-safe DSN (password redacted) for stats/logging.

        Handles every form psycopg accepts -- ``postgres://`` URLs, libpq keyword
        DSNs (``host=... password=...``) and no-scheme forms -- by parsing rather
        than string-splitting. If parsing is uncertain we redact the whole value
        so a password can never leak through (e.g. via ``libr stats``).
        """
        dsn = self.dsn or ""
        if not dsn:
            return dsn
        try:
            from psycopg.conninfo import conninfo_to_dict, make_conninfo

            params = conninfo_to_dict(dsn)
            if params.get("password"):
                params["password"] = "***"  # noqa: S105 - redaction, not a credential
            return str(make_conninfo(**params))
        except Exception:
            return "***"
