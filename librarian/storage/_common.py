"""
Dialect-free storage helpers shared by every concrete backend.

These have zero substrate content -- they are pure value transforms used
identically by :mod:`librarian.storage.sqlite_storage` and
:mod:`librarian.storage.postgres`. Keeping them here stops the byte-for-byte
twins from drifting. (The genuinely dialect-specific SQL -- ``write_upsert`` /
``transaction`` / ``put_sync_state`` -- intentionally stays duplicated per
backend, since its body diverges.)
"""

from collections.abc import Callable
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from librarian.types import AssetType, EmbeddingModality

if TYPE_CHECKING:
    from librarian.storage.protocols import ChunkContext

__all__ = [
    "chunk_context_from_row",
    "deleted_filter",
    "iso",
    "json_default",
    "list_documents_query",
    "modality_table",
]


def list_documents_query(
    placeholder: str,
    unbounded_limit: str,
    format_date: Callable[[datetime], Any],
    start_date: datetime | None,
    end_date: datetime | None,
    limit: int | None,
    offset: int,
) -> tuple[str, list[Any]]:
    """Build the shared ``list_documents`` SQL + params for either substrate.

    The one place the pagination query lives, so the two backends can't drift on
    the ordering (``updated_at DESC, id DESC``) or the limit/offset semantics
    that the ``MetadataStore`` protocol makes a cross-substrate contract. The
    dialect differences are passed in: ``placeholder`` (``?`` vs ``%s``),
    ``unbounded_limit`` (the ``LIMIT`` token meaning "no cap" -- ``-1`` on
    SQLite, ``ALL`` on Postgres, used so ``offset`` works without a ``limit``),
    and ``format_date`` (how a datetime is bound). ``placeholder`` and
    ``unbounded_limit`` are fixed internal tokens, never user input.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if start_date:
        clauses.append(f"updated_at >= {placeholder}")
        params.append(format_date(start_date))
    if end_date:
        clauses.append(f"updated_at < {placeholder}")
        params.append(format_date(end_date))
    sql = "SELECT * FROM documents"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY updated_at DESC, id DESC"
    if limit is not None:
        sql += f" LIMIT {placeholder} OFFSET {placeholder}"
        params.extend((limit, offset))
    elif offset:
        sql += f" LIMIT {unbounded_limit} OFFSET {placeholder}"
        params.append(offset)
    return sql, params


def modality_table(modality: EmbeddingModality) -> str:
    """Map an embedding modality to its physical embedding table name."""
    if modality == EmbeddingModality.CODE:
        return "vec_chunks_code"
    if modality == EmbeddingModality.VISION:
        return "vec_chunks_vision"
    return "chunk_embeddings"


def deleted_filter(include_deleted: bool, alias: str = "c") -> str:
    """Return the ``deleted_at`` predicate fragment for a search/window query.

    The single place that encodes soft-delete filtering for every store on both
    substrates: ``""`` when soft-deleted rows should be included, otherwise
    ``AND <alias>.deleted_at IS NULL``. Interpolated into SQL (never a bound
    parameter), so callers keep their ``# noqa: S608`` -- but there is now one
    definition to change when soft-delete semantics evolve, not seven.
    """
    return "" if include_deleted else f"AND {alias}.deleted_at IS NULL"


def chunk_context_from_row(row: Any) -> "ChunkContext":
    """Map a ``get_chunk_context`` result row to a :class:`ChunkContext`.

    A pure value transform shared verbatim by both backends' ``get_chunk_context``
    (the SQL that produces ``row`` diverges per dialect; this mapping does not).
    Works for both ``sqlite3.Row`` and psycopg dict rows -- both support
    ``row["key"]`` access.
    """
    from librarian.storage.protocols import ChunkContext

    return ChunkContext(
        chunk_id=row["chunk_id"],
        internal_id=row["internal_id"],
        document_id=row["document_id"],
        document_path=row["document_path"],
        content=row["content"],
        heading_path=row["heading_path"],
        chunk_index=row["chunk_index"],
        asset_type=row["asset_type"] or AssetType.TEXT.value,
        chunk_source_uri=row["chunk_source_uri"],
        deleted_at=row["deleted_at"],
    )


def iso(value: datetime | None) -> str | None:
    """Render a datetime as an ISO-8601 string, passing ``None`` through."""
    return value.isoformat() if value is not None else None


def json_default(value: Any) -> str:
    """JSON fallback for date/datetime values (e.g. from YAML frontmatter)."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)
