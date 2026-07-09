"""Read-only library health diagnostics.

This module reports deterministic retrieval risk signals for an existing
library index. It does not evaluate retrieval accuracy, run searches, create
schema, migrate databases, or write to storage.
"""

from __future__ import annotations

# ruff: noqa: S608
# Health diagnostics must inspect fixed internal schema tables across backends.
# User values are always passed as SQL parameters; dynamic SQL below is limited
# to known table names, known column names, and internally-built WHERE fragments.
import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Protocol
from urllib.request import pathname2url

import sqlite_vec

import librarian.config as config

SMALL_CHUNK_CHARS = 50
LARGE_CHUNK_CHARS = 4000
VERY_SHORT_DOCUMENT_CHARS = 100
SHORT_PDF_MIN_CHARS = 100
SHORT_PDF_CHARS_PER_PAGE = 50
FILE_CHECK_LIMIT = 1000
ISSUE_SAMPLE_LIMIT = 20


class Severity(str, Enum):
    """Health issue severity."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueCode(str, Enum):
    """Stable health issue codes."""

    EMPTY_INDEX = "empty_index"
    SCHEMA_UNAVAILABLE = "schema_unavailable"
    DOCUMENTS_WITH_ZERO_CHUNKS = "documents_with_zero_chunks"
    MISSING_EMBEDDINGS = "missing_embeddings"
    FTS_MISMATCH = "fts_mismatch"
    EMPTY_CHUNKS = "empty_chunks"
    SMALL_CHUNKS = "small_chunks"
    LARGE_CHUNKS = "large_chunks"
    DUPLICATE_CHUNKS = "duplicate_chunks"
    SHORT_DOCUMENT = "short_document"
    SHORT_PDF_TEXT = "short_pdf_text"
    IMAGE_WITHOUT_OCR = "image_without_ocr"
    CODE_WITHOUT_SYMBOLS = "code_without_symbols"
    INDEXED_FILE_MISSING = "indexed_file_missing"
    INDEXED_FILE_CHANGED = "indexed_file_changed"
    SOURCE_PATH_MISSING = "source_path_missing"
    SYNC_STATE_ERROR = "sync_state_error"


SEVERITY_ORDER = {
    Severity.HIGH: 0,
    Severity.MEDIUM: 1,
    Severity.LOW: 2,
}


@dataclass
class HealthIssue:
    """One deterministic library health issue."""

    severity: Severity
    code: IssueCode
    message: str
    path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        data: dict[str, Any] = {
            "severity": self.severity.value,
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }
        if self.path is not None:
            data["path"] = self.path
        return data


@dataclass
class HealthReport:
    """Complete library health report."""

    generated_at: str
    backend: str
    database_path: str
    document_count: int
    chunk_count: int
    embedding_count: int
    fts_count: int | None
    embedding_coverage: float
    fts_coverage: float | None
    document_asset_counts: dict[str, int]
    chunk_asset_counts: dict[str, int]
    chunk_modality_counts: dict[str, int]
    embedding_counts: dict[str, int]
    issue_counts: dict[str, int]
    issues: list[HealthIssue]
    source: str | None = None
    checked_files: int = 0
    skipped_file_checks: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "generated_at": self.generated_at,
            "backend": self.backend,
            "database_path": self.database_path,
            "source": self.source,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "embedding_count": self.embedding_count,
            "fts_count": self.fts_count,
            "embedding_coverage": self.embedding_coverage,
            "fts_coverage": self.fts_coverage,
            "document_asset_counts": self.document_asset_counts,
            "chunk_asset_counts": self.chunk_asset_counts,
            "chunk_modality_counts": self.chunk_modality_counts,
            "embedding_counts": self.embedding_counts,
            "issue_counts": self.issue_counts,
            "checked_files": self.checked_files,
            "skipped_file_checks": self.skipped_file_checks,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class HealthSourceFilter:
    """A source path filter resolved by the CLI."""

    name: str | None
    path: str
    is_file: bool = False

    @property
    def label(self) -> str:
        """Human-readable source label."""
        return self.name or self.path

    @property
    def directory_prefix(self) -> str:
        """Prefix used to avoid sibling path over-matching."""
        return self.path.rstrip(os.sep) + os.sep

    def contains(self, candidate: str) -> bool:
        """Return True when ``candidate`` belongs to this source."""
        if self.is_file:
            return candidate == self.path
        return candidate == self.path or candidate.startswith(self.directory_prefix)


@dataclass
class DocumentHealthRow:
    """Document fields needed for health checks."""

    id: int
    path: str
    asset_type: str
    content: str
    content_length: int
    metadata: dict[str, Any]
    file_mtime: float | None = None
    document_source_uri: str | None = None


@dataclass
class PathCount:
    """Count with optional example paths."""

    count: int = 0
    paths: list[str] = field(default_factory=list)


@dataclass
class DuplicateChunkSummary:
    """Exact duplicate chunk summary."""

    group_count: int = 0
    duplicate_chunk_count: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SyncStateHealthRow:
    """Stored connector sync state fields relevant to health."""

    source_key: str
    status: str | None
    last_error: str | None
    documents_seen: int
    chunks_written: int


@dataclass
class InspectorData:
    """Normalized aggregate data from one storage backend."""

    backend: str
    database_path: str
    document_count: int = 0
    chunk_count: int = 0
    fts_count: int | None = None
    document_asset_counts: dict[str, int] = field(default_factory=dict)
    chunk_asset_counts: dict[str, int] = field(default_factory=dict)
    chunk_modality_counts: dict[str, int] = field(default_factory=dict)
    embedding_counts: dict[str, int] = field(default_factory=dict)
    documents: list[DocumentHealthRow] = field(default_factory=list)
    source_file_mtimes: dict[str, float] = field(default_factory=dict)
    sync_states: list[SyncStateHealthRow] = field(default_factory=list)
    zero_chunk_documents: PathCount = field(default_factory=PathCount)
    missing_embedding_chunks: PathCount = field(default_factory=PathCount)
    empty_chunks: PathCount = field(default_factory=PathCount)
    small_chunks: PathCount = field(default_factory=PathCount)
    large_chunks: PathCount = field(default_factory=PathCount)
    duplicate_chunks: DuplicateChunkSummary = field(default_factory=DuplicateChunkSummary)
    unavailable_reason: str | None = None

    @property
    def embedding_count(self) -> int:
        """Total active embeddings counted across physical embedding tables."""
        return sum(self.embedding_counts.values())


class HealthInspector(Protocol):
    """Backend-specific read-only aggregate inspector."""

    def collect(self) -> InspectorData:
        """Collect normalized aggregate data."""


def _parse_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _sample_paths(rows: list[Any], key: str = "path") -> list[str]:
    return [str(row[key]) for row in rows[:ISSUE_SAMPLE_LIMIT]]


class SQLiteHealthInspector:
    """Read-only SQLite health aggregate inspector."""

    _EMBEDDING_TABLES: ClassVar[dict[str, str]] = {
        "text": "chunk_embeddings",
        "code": "vec_chunks_code",
        "vision": "vec_chunks_vision",
    }

    def __init__(self, db_path: str, source_filter: HealthSourceFilter | None = None) -> None:
        self.db_path = db_path
        self.source_filter = source_filter
        self._tables: set[str] = set()
        self._columns: dict[str, set[str]] = {}

    def collect(self) -> InspectorData:
        data = InspectorData(
            backend="sqlite",
            database_path=self.db_path,
            embedding_counts=dict.fromkeys(self._EMBEDDING_TABLES, 0),
        )
        path = Path(self.db_path)
        if not path.exists():
            data.unavailable_reason = f"Database does not exist: {self.db_path}"
            return data

        try:
            conn = self._connect_readonly(path)
        except sqlite3.Error as exc:
            data.unavailable_reason = f"Could not open database read-only: {exc}"
            return data

        try:
            self._load_schema(conn)
            if "documents" not in self._tables:
                data.unavailable_reason = "Database has no documents table"
                return data
            data.document_count = self._count_documents(conn)
            data.document_asset_counts = self._document_asset_counts(conn)
            data.documents = self._documents(conn)

            if "chunks" in self._tables:
                data.chunk_count = self._count_active_chunks(conn)
                data.chunk_asset_counts = self._chunk_group_counts(conn, "asset_type", "text")
                data.chunk_modality_counts = self._chunk_group_counts(conn, "modality", "text")
                data.zero_chunk_documents = self._zero_chunk_documents(conn)
                data.missing_embedding_chunks = self._missing_embedding_chunks(conn)
                data.empty_chunks = self._chunk_size_issue(conn, "=", 0)
                data.small_chunks = self._chunk_size_issue(conn, "<", SMALL_CHUNK_CHARS, minimum=1)
                data.large_chunks = self._chunk_size_issue(conn, ">", LARGE_CHUNK_CHARS)
                data.duplicate_chunks = self._duplicate_chunks(conn)

            data.embedding_counts = {
                modality: self._embedding_count(conn, table)
                for modality, table in self._EMBEDDING_TABLES.items()
            }
            data.fts_count = self._fts_count(conn)
            data.source_file_mtimes = self._source_file_mtimes(conn)
            data.sync_states = self._sync_states(conn)
            return data
        finally:
            conn.close()

    def _connect_readonly(self, path: Path) -> sqlite3.Connection:
        uri = f"file:{pathname2url(str(path))}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def _load_schema(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
        self._tables = {str(row["name"]) for row in rows}
        self._columns = {}
        for table in self._tables:
            try:
                cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            except sqlite3.Error:
                self._columns[table] = set()
            else:
                self._columns[table] = {str(row["name"]) for row in cols}

    def _has_column(self, table: str, column: str) -> bool:
        return column in self._columns.get(table, set())

    def _document_conditions(self, alias: str = "d") -> tuple[list[str], list[Any]]:
        if self.source_filter is None:
            return [], []
        if self.source_filter.is_file:
            return [f"{alias}.path = ?"], [self.source_filter.path]
        return [f"({alias}.path = ? OR {alias}.path LIKE ?)"], [
            self.source_filter.path,
            f"{self.source_filter.directory_prefix}%",
        ]

    def _active_chunk_conditions(self) -> tuple[list[str], list[Any]]:
        conditions: list[str] = []
        params: list[Any] = []
        if self._has_column("chunks", "deleted_at"):
            conditions.append("c.deleted_at IS NULL")
        doc_conditions, doc_params = self._document_conditions("d")
        conditions.extend(doc_conditions)
        params.extend(doc_params)
        return conditions, params

    def _where(self, conditions: list[str]) -> str:
        return " WHERE " + " AND ".join(conditions) if conditions else ""

    def _count_documents(self, conn: sqlite3.Connection) -> int:
        conditions, params = self._document_conditions("d")
        sql = "SELECT COUNT(*) AS n FROM documents d" + self._where(conditions)
        return int(conn.execute(sql, params).fetchone()["n"])

    def _count_active_chunks(self, conn: sqlite3.Connection) -> int:
        conditions, params = self._active_chunk_conditions()
        sql = (
            "SELECT COUNT(*) AS n FROM chunks c "
            "JOIN documents d ON d.id = c.document_id" + self._where(conditions)
        )
        return int(conn.execute(sql, params).fetchone()["n"])

    def _document_asset_counts(self, conn: sqlite3.Connection) -> dict[str, int]:
        conditions, params = self._document_conditions("d")
        column = "d.asset_type" if self._has_column("documents", "asset_type") else "'text'"
        sql = (
            f"SELECT COALESCE({column}, 'text') AS key, COUNT(*) AS n FROM documents d"
            + self._where(conditions)
            + " GROUP BY key ORDER BY key"
        )
        return {str(row["key"]): int(row["n"]) for row in conn.execute(sql, params).fetchall()}

    def _chunk_group_counts(
        self, conn: sqlite3.Connection, column: str, default: str
    ) -> dict[str, int]:
        conditions, params = self._active_chunk_conditions()
        expr = f"c.{column}" if self._has_column("chunks", column) else f"'{default}'"
        sql = (
            f"SELECT COALESCE({expr}, ?) AS key, COUNT(*) AS n FROM chunks c "
            "JOIN documents d ON d.id = c.document_id"
            + self._where(conditions)
            + " GROUP BY key ORDER BY key"
        )
        return {
            str(row["key"]): int(row["n"])
            for row in conn.execute(sql, [default, *params]).fetchall()
        }

    def _documents(self, conn: sqlite3.Connection) -> list[DocumentHealthRow]:
        conditions, params = self._document_conditions("d")
        asset_expr = "d.asset_type" if self._has_column("documents", "asset_type") else "'text'"
        uri_expr = (
            "d.document_source_uri"
            if self._has_column("documents", "document_source_uri")
            else "NULL"
        )
        mtime_expr = "d.file_mtime" if self._has_column("documents", "file_mtime") else "NULL"
        metadata_expr = "d.metadata" if self._has_column("documents", "metadata") else "NULL"
        sql = (
            "SELECT d.id, d.path, "
            f"COALESCE({asset_expr}, 'text') AS asset_type, "
            "COALESCE(d.content, '') AS content, "
            "LENGTH(COALESCE(d.content, '')) AS content_length, "
            f"{metadata_expr} AS metadata, {mtime_expr} AS file_mtime, "
            f"{uri_expr} AS document_source_uri "
            "FROM documents d" + self._where(conditions) + " ORDER BY d.path"
        )
        rows = conn.execute(sql, params).fetchall()
        return [
            DocumentHealthRow(
                id=int(row["id"]),
                path=str(row["path"]),
                asset_type=str(row["asset_type"] or "text"),
                content=str(row["content"] or ""),
                content_length=int(row["content_length"] or 0),
                metadata=_parse_metadata(row["metadata"]),
                file_mtime=float(row["file_mtime"]) if row["file_mtime"] is not None else None,
                document_source_uri=str(row["document_source_uri"])
                if row["document_source_uri"]
                else None,
            )
            for row in rows
        ]

    def _embedding_count(self, conn: sqlite3.Connection, table: str) -> int:
        if table not in self._tables or "chunks" not in self._tables:
            return 0
        conditions, params = self._active_chunk_conditions()
        sql = (
            f"SELECT COUNT(*) AS n FROM {table} e "
            "JOIN chunks c ON c.id = e.chunk_id "
            "JOIN documents d ON d.id = c.document_id" + self._where(conditions)
        )
        return int(conn.execute(sql, params).fetchone()["n"])

    def _fts_count(self, conn: sqlite3.Connection) -> int | None:
        if "chunks_fts" not in self._tables or "chunks" not in self._tables:
            return None
        conditions, params = self._active_chunk_conditions()
        sql = (
            "SELECT COUNT(*) AS n FROM chunks_fts f "
            "JOIN chunks c ON c.id = f.rowid "
            "JOIN documents d ON d.id = c.document_id" + self._where(conditions)
        )
        return int(conn.execute(sql, params).fetchone()["n"])

    def _zero_chunk_documents(self, conn: sqlite3.Connection) -> PathCount:
        if "chunks" not in self._tables:
            return PathCount()
        doc_conditions, params = self._document_conditions("d")
        active_join = "c.document_id = d.id"
        if self._has_column("chunks", "deleted_at"):
            active_join += " AND c.deleted_at IS NULL"
        conditions = [*doc_conditions, "c.id IS NULL"]
        sql = (
            "SELECT d.path FROM documents d "
            f"LEFT JOIN chunks c ON {active_join}" + self._where(conditions) + " ORDER BY d.path"
        )
        rows = conn.execute(sql, params).fetchall()
        return PathCount(count=len(rows), paths=_sample_paths(rows))

    def _missing_embedding_chunks(self, conn: sqlite3.Connection) -> PathCount:
        if "chunks" not in self._tables:
            return PathCount()
        joins = []
        null_checks = []
        for alias, table in (
            ("te", "chunk_embeddings"),
            ("ce", "vec_chunks_code"),
            ("ve", "vec_chunks_vision"),
        ):
            if table in self._tables:
                joins.append(f"LEFT JOIN {table} {alias} ON {alias}.chunk_id = c.id")
                null_checks.append(f"{alias}.chunk_id IS NULL")
        if not null_checks:
            return PathCount(count=self._count_active_chunks(conn), paths=[])
        conditions, params = self._active_chunk_conditions()
        conditions.extend(null_checks)
        sql = (
            "SELECT d.path FROM chunks c "
            "JOIN documents d ON d.id = c.document_id "
            + " ".join(joins)
            + self._where(conditions)
            + " ORDER BY d.path, c.chunk_index"
        )
        rows = conn.execute(sql, params).fetchall()
        return PathCount(count=len(rows), paths=_sample_paths(rows))

    def _chunk_size_issue(
        self,
        conn: sqlite3.Connection,
        operator: str,
        threshold: int,
        *,
        minimum: int | None = None,
    ) -> PathCount:
        conditions, params = self._active_chunk_conditions()
        length_expr = "LENGTH(TRIM(COALESCE(c.content, '')))"
        if minimum is not None:
            conditions.append(f"{length_expr} >= ?")
            params.append(minimum)
        conditions.append(f"{length_expr} {operator} ?")
        params.append(threshold)
        sql = (
            "SELECT d.path FROM chunks c "
            "JOIN documents d ON d.id = c.document_id"
            + self._where(conditions)
            + " ORDER BY d.path, c.chunk_index"
        )
        rows = conn.execute(sql, params).fetchall()
        return PathCount(count=len(rows), paths=_sample_paths(rows))

    def _duplicate_chunks(self, conn: sqlite3.Connection) -> DuplicateChunkSummary:
        conditions, params = self._active_chunk_conditions()
        conditions.append("LENGTH(TRIM(COALESCE(c.content, ''))) > 0")
        where = self._where(conditions)
        base = (
            "FROM chunks c JOIN documents d ON d.id = c.document_id"
            + where
            + " GROUP BY c.content HAVING COUNT(*) > 1"
        )
        # The grouped subqueries are composed only from fixed internal SQL
        # fragments plus parameter markers from _active_chunk_conditions.
        group_count = int(
            conn.execute(f"SELECT COUNT(*) AS n FROM (SELECT c.content {base})", params).fetchone()[
                "n"
            ]
        )
        duplicate_chunk_count = int(
            conn.execute(
                f"SELECT COALESCE(SUM(n), 0) AS n FROM (SELECT COUNT(*) AS n {base})",
                params,
            ).fetchone()["n"]
        )
        rows = conn.execute(
            f"SELECT c.content, COUNT(*) AS n {base} ORDER BY n DESC LIMIT ?",
            [*params, 5],
        ).fetchall()
        examples = [{"count": int(row["n"]), "preview": str(row["content"])[:120]} for row in rows]
        return DuplicateChunkSummary(
            group_count=group_count,
            duplicate_chunk_count=duplicate_chunk_count,
            examples=examples,
        )

    def _source_file_mtimes(self, conn: sqlite3.Connection) -> dict[str, float]:
        if "source_file_state" not in self._tables:
            return {}
        conditions = ["source_key = ?"]
        params: list[Any] = ["local_file"]
        if self.source_filter is not None:
            if self.source_filter.is_file:
                conditions.append("path = ?")
                params.append(self.source_filter.path)
            else:
                conditions.append("(path = ? OR path LIKE ?)")
                params.extend([self.source_filter.path, f"{self.source_filter.directory_prefix}%"])
        rows = conn.execute(
            "SELECT path, mtime FROM source_file_state" + self._where(conditions),
            params,
        ).fetchall()
        return {str(row["path"]): float(row["mtime"]) for row in rows if row["mtime"] is not None}

    def _sync_states(self, conn: sqlite3.Connection) -> list[SyncStateHealthRow]:
        if "sync_state" not in self._tables:
            return []
        rows = conn.execute(
            """
            SELECT source_key, status, last_error, documents_seen, chunks_written
            FROM sync_state
            ORDER BY source_key
            """
        ).fetchall()
        return [
            SyncStateHealthRow(
                source_key=str(row["source_key"]),
                status=str(row["status"]) if row["status"] is not None else None,
                last_error=str(row["last_error"]) if row["last_error"] else None,
                documents_seen=int(row["documents_seen"] or 0),
                chunks_written=int(row["chunks_written"] or 0),
            )
            for row in rows
        ]


class PostgresHealthInspector:
    """Postgres health aggregate inspector.

    The public storage bundle does not expose table-level aggregate diagnostics,
    so this backend-specific inspector keeps substrate SQL isolated here.
    """

    _EMBEDDING_TABLES: ClassVar[dict[str, str]] = {
        "text": "chunk_embeddings",
        "code": "vec_chunks_code",
        "vision": "vec_chunks_vision",
    }

    def __init__(self, source_filter: HealthSourceFilter | None = None) -> None:
        self.source_filter = source_filter

    def collect(self) -> InspectorData:
        from librarian.storage import get_read_storage

        storage = get_read_storage()
        database = getattr(storage, "database", None)
        data = InspectorData(
            backend="postgres",
            database_path=getattr(database, "_sanitized_dsn", lambda: "postgres")(),
            embedding_counts=dict.fromkeys(self._EMBEDDING_TABLES, 0),
        )
        if database is None or not hasattr(database, "_get_connection"):
            data.unavailable_reason = "Postgres storage connection is unavailable"
            return data

        conn = database._get_connection()
        data.document_count = self._count_documents(conn)
        data.chunk_count = self._count_active_chunks(conn)
        data.document_asset_counts = self._document_asset_counts(conn)
        data.chunk_asset_counts = self._chunk_group_counts(conn, "asset_type", "text")
        data.chunk_modality_counts = self._chunk_group_counts(conn, "modality", "text")
        data.embedding_counts = {
            modality: self._embedding_count(conn, table)
            for modality, table in self._EMBEDDING_TABLES.items()
        }
        data.fts_count = data.chunk_count
        data.documents = self._documents(conn)
        data.zero_chunk_documents = self._zero_chunk_documents(conn)
        data.missing_embedding_chunks = self._missing_embedding_chunks(conn)
        data.empty_chunks = self._chunk_size_issue(conn, "=", 0)
        data.small_chunks = self._chunk_size_issue(conn, "<", SMALL_CHUNK_CHARS, minimum=1)
        data.large_chunks = self._chunk_size_issue(conn, ">", LARGE_CHUNK_CHARS)
        data.duplicate_chunks = self._duplicate_chunks(conn)
        data.source_file_mtimes = self._source_file_mtimes(conn)
        data.sync_states = self._sync_states(conn)
        return data

    def _document_conditions(self, alias: str = "d") -> tuple[list[str], list[Any]]:
        if self.source_filter is None:
            return [], []
        if self.source_filter.is_file:
            return [f"{alias}.path = %s"], [self.source_filter.path]
        return [f"({alias}.path = %s OR {alias}.path LIKE %s)"], [
            self.source_filter.path,
            f"{self.source_filter.directory_prefix}%",
        ]

    def _active_chunk_conditions(self) -> tuple[list[str], list[Any]]:
        conditions = ["c.deleted_at IS NULL"]
        params: list[Any] = []
        doc_conditions, doc_params = self._document_conditions("d")
        conditions.extend(doc_conditions)
        params.extend(doc_params)
        return conditions, params

    def _where(self, conditions: list[str]) -> str:
        return " WHERE " + " AND ".join(conditions) if conditions else ""

    def _count_documents(self, conn: Any) -> int:
        conditions, params = self._document_conditions("d")
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM documents d" + self._where(conditions),
            params,
        ).fetchone()
        return int(row["n"])

    def _count_active_chunks(self, conn: Any) -> int:
        conditions, params = self._active_chunk_conditions()
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM chunks c JOIN documents d ON d.id = c.document_id"
            + self._where(conditions),
            params,
        ).fetchone()
        return int(row["n"])

    def _document_asset_counts(self, conn: Any) -> dict[str, int]:
        conditions, params = self._document_conditions("d")
        rows = conn.execute(
            "SELECT COALESCE(d.asset_type, 'text') AS key, COUNT(*) AS n FROM documents d"
            + self._where(conditions)
            + " GROUP BY key ORDER BY key",
            params,
        ).fetchall()
        return {str(row["key"]): int(row["n"]) for row in rows}

    def _chunk_group_counts(self, conn: Any, column: str, default: str) -> dict[str, int]:
        conditions, params = self._active_chunk_conditions()
        rows = conn.execute(
            f"SELECT COALESCE(c.{column}, %s) AS key, COUNT(*) AS n FROM chunks c "
            "JOIN documents d ON d.id = c.document_id"
            + self._where(conditions)
            + " GROUP BY key ORDER BY key",
            [default, *params],
        ).fetchall()
        return {str(row["key"]): int(row["n"]) for row in rows}

    def _documents(self, conn: Any) -> list[DocumentHealthRow]:
        conditions, params = self._document_conditions("d")
        rows = conn.execute(
            """
            SELECT d.id, d.path, COALESCE(d.asset_type, 'text') AS asset_type,
                   COALESCE(d.content, '') AS content,
                   LENGTH(COALESCE(d.content, '')) AS content_length,
                   d.metadata, d.file_mtime, d.document_source_uri
            FROM documents d
            """
            + self._where(conditions)
            + " ORDER BY d.path",
            params,
        ).fetchall()
        return [
            DocumentHealthRow(
                id=int(row["id"]),
                path=str(row["path"]),
                asset_type=str(row["asset_type"] or "text"),
                content=str(row["content"] or ""),
                content_length=int(row["content_length"] or 0),
                metadata=_parse_metadata(row["metadata"]),
                file_mtime=float(row["file_mtime"]) if row["file_mtime"] is not None else None,
                document_source_uri=str(row["document_source_uri"])
                if row["document_source_uri"]
                else None,
            )
            for row in rows
        ]

    def _embedding_count(self, conn: Any, table: str) -> int:
        conditions, params = self._active_chunk_conditions()
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM {table} e "
            "JOIN chunks c ON c.id = e.chunk_id "
            "JOIN documents d ON d.id = c.document_id" + self._where(conditions),
            params,
        ).fetchone()
        return int(row["n"])

    def _zero_chunk_documents(self, conn: Any) -> PathCount:
        doc_conditions, params = self._document_conditions("d")
        conditions = [*doc_conditions, "c.id IS NULL"]
        rows = conn.execute(
            "SELECT d.path FROM documents d "
            "LEFT JOIN chunks c ON c.document_id = d.id AND c.deleted_at IS NULL"
            + self._where(conditions)
            + " ORDER BY d.path",
            params,
        ).fetchall()
        return PathCount(count=len(rows), paths=_sample_paths(rows))

    def _missing_embedding_chunks(self, conn: Any) -> PathCount:
        conditions, params = self._active_chunk_conditions()
        conditions.extend([
            "te.chunk_id IS NULL",
            "ce.chunk_id IS NULL",
            "ve.chunk_id IS NULL",
        ])
        rows = conn.execute(
            """
            SELECT d.path FROM chunks c
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN chunk_embeddings te ON te.chunk_id = c.id
            LEFT JOIN vec_chunks_code ce ON ce.chunk_id = c.id
            LEFT JOIN vec_chunks_vision ve ON ve.chunk_id = c.id
            """
            + self._where(conditions)
            + " ORDER BY d.path, c.chunk_index",
            params,
        ).fetchall()
        return PathCount(count=len(rows), paths=_sample_paths(rows))

    def _chunk_size_issue(
        self, conn: Any, operator: str, threshold: int, *, minimum: int | None = None
    ) -> PathCount:
        conditions, params = self._active_chunk_conditions()
        length_expr = "LENGTH(TRIM(COALESCE(c.content, '')))"
        if minimum is not None:
            conditions.append(f"{length_expr} >= %s")
            params.append(minimum)
        conditions.append(f"{length_expr} {operator} %s")
        params.append(threshold)
        rows = conn.execute(
            "SELECT d.path FROM chunks c JOIN documents d ON d.id = c.document_id"
            + self._where(conditions)
            + " ORDER BY d.path, c.chunk_index",
            params,
        ).fetchall()
        return PathCount(count=len(rows), paths=_sample_paths(rows))

    def _duplicate_chunks(self, conn: Any) -> DuplicateChunkSummary:
        conditions, params = self._active_chunk_conditions()
        conditions.append("LENGTH(TRIM(COALESCE(c.content, ''))) > 0")
        where = self._where(conditions)
        base = (
            "FROM chunks c JOIN documents d ON d.id = c.document_id"
            + where
            + " GROUP BY c.content HAVING COUNT(*) > 1"
        )
        group_count = int(
            conn.execute(
                f"SELECT COUNT(*) AS n FROM (SELECT c.content {base}) x", params
            ).fetchone()["n"]
        )
        duplicate_chunk_count = int(
            conn.execute(
                f"SELECT COALESCE(SUM(n), 0) AS n FROM (SELECT COUNT(*) AS n {base}) x",
                params,
            ).fetchone()["n"]
        )
        rows = conn.execute(
            f"SELECT c.content, COUNT(*) AS n {base} ORDER BY n DESC LIMIT %s",
            [*params, 5],
        ).fetchall()
        return DuplicateChunkSummary(
            group_count=group_count,
            duplicate_chunk_count=duplicate_chunk_count,
            examples=[
                {"count": int(row["n"]), "preview": str(row["content"])[:120]} for row in rows
            ],
        )

    def _source_file_mtimes(self, conn: Any) -> dict[str, float]:
        conditions = ["source_key = %s"]
        params: list[Any] = ["local_file"]
        if self.source_filter is not None:
            if self.source_filter.is_file:
                conditions.append("path = %s")
                params.append(self.source_filter.path)
            else:
                conditions.append("(path = %s OR path LIKE %s)")
                params.extend([self.source_filter.path, f"{self.source_filter.directory_prefix}%"])
        rows = conn.execute(
            "SELECT path, mtime FROM source_file_state" + self._where(conditions),
            params,
        ).fetchall()
        return {str(row["path"]): float(row["mtime"]) for row in rows if row["mtime"] is not None}

    def _sync_states(self, conn: Any) -> list[SyncStateHealthRow]:
        rows = conn.execute(
            """
            SELECT source_key, status, last_error, documents_seen, chunks_written
            FROM sync_state
            ORDER BY source_key
            """
        ).fetchall()
        return [
            SyncStateHealthRow(
                source_key=str(row["source_key"]),
                status=str(row["status"]) if row["status"] is not None else None,
                last_error=str(row["last_error"]) if row["last_error"] else None,
                documents_seen=int(row["documents_seen"] or 0),
                chunks_written=int(row["chunks_written"] or 0),
            )
            for row in rows
        ]


def normalize_source_filter(source: dict[str, Any] | None) -> HealthSourceFilter | None:
    """Convert a CLI source dict into a normalized source filter."""
    if source is None:
        return None
    raw_path = str(source["path"])
    path = str(Path(raw_path).expanduser().resolve(strict=False))
    is_file = bool(source.get("is_file", Path(path).is_file()))
    return HealthSourceFilter(name=source.get("name"), path=path, is_file=is_file)


def path_source_filter(source: str) -> HealthSourceFilter:
    """Build a source filter directly from a path-like CLI argument."""
    path = Path(source).expanduser().resolve(strict=False)
    return HealthSourceFilter(name=None, path=str(path), is_file=path.is_file())


def _is_local_source(source: dict[str, Any]) -> bool:
    return source.get("type", "local") == "local" and "path" in source


def _local_source_paths(sources: list[dict[str, Any]]) -> list[HealthSourceFilter]:
    filters: list[HealthSourceFilter] = []
    for source in sources:
        if not _is_local_source(source):
            continue
        normalized = normalize_source_filter(source)
        if normalized is not None:
            filters.append(normalized)
    return filters


def _is_local_document(
    document: DocumentHealthRow,
    source_mtimes: dict[str, float],
    local_sources: list[HealthSourceFilter],
) -> bool:
    if document.path in source_mtimes:
        return True
    if document.document_source_uri and document.document_source_uri.startswith("file://"):
        return True
    return any(source.contains(document.path) for source in local_sources)


def _file_mtime_for_document(
    document: DocumentHealthRow, source_mtimes: dict[str, float]
) -> float | None:
    return source_mtimes.get(document.path, document.file_mtime)


def _document_has_ocr_text(document: DocumentHealthRow) -> bool:
    """Return True when image OCR text is present in metadata or searchable content."""
    if str(document.metadata.get("ocr_text", "")).strip():
        return True
    marker = "Text extracted from image:"
    if marker not in document.content:
        return False
    return bool(document.content.split(marker, 1)[1].strip())


def _issue_counts(issues: list[HealthIssue]) -> dict[str, int]:
    counts = {severity.value: 0 for severity in Severity}
    for issue in issues:
        counts[issue.severity.value] += 1
    return counts


def _sorted_issues(issues: list[HealthIssue]) -> list[HealthIssue]:
    return sorted(issues, key=lambda issue: (SEVERITY_ORDER[issue.severity], issue.code.value))


class HealthScanner:
    """Build read-only health reports for an indexed library."""

    def __init__(
        self,
        *,
        database_path: str | None = None,
        storage_backend: str | None = None,
        source_filter: HealthSourceFilter | None = None,
        sources: list[dict[str, Any]] | None = None,
        file_check_limit: int = FILE_CHECK_LIMIT,
    ) -> None:
        self.database_path = database_path or config.DATABASE_PATH
        self.storage_backend = storage_backend or config.STORAGE_BACKEND
        self.source_filter = source_filter
        self.sources = sources or []
        self.file_check_limit = file_check_limit

    def scan(self) -> HealthReport:
        """Scan the current index and return a health report."""
        data = self._inspector().collect()
        issues: list[HealthIssue] = []

        if data.unavailable_reason:
            issues.append(
                HealthIssue(
                    severity=Severity.HIGH,
                    code=IssueCode.SCHEMA_UNAVAILABLE,
                    message=data.unavailable_reason,
                    details={"backend": data.backend},
                )
            )

        if data.document_count == 0:
            issues.append(
                HealthIssue(
                    severity=Severity.HIGH,
                    code=IssueCode.EMPTY_INDEX,
                    message="No indexed documents were found.",
                )
            )
        elif data.chunk_count == 0:
            issues.append(
                HealthIssue(
                    severity=Severity.HIGH,
                    code=IssueCode.EMPTY_INDEX,
                    message="Indexed documents exist, but no active chunks were found.",
                )
            )

        self._add_count_issue(
            issues,
            data.zero_chunk_documents,
            Severity.HIGH,
            IssueCode.DOCUMENTS_WITH_ZERO_CHUNKS,
            "documents have no active chunks.",
        )
        self._add_count_issue(
            issues,
            data.missing_embedding_chunks,
            Severity.HIGH,
            IssueCode.MISSING_EMBEDDINGS,
            "active chunks are missing embeddings.",
        )
        if data.fts_count is not None and data.fts_count != data.chunk_count:
            issues.append(
                HealthIssue(
                    severity=Severity.MEDIUM,
                    code=IssueCode.FTS_MISMATCH,
                    message=(
                        "Keyword index row count does not match active chunk count "
                        f"({data.fts_count} keyword rows vs {data.chunk_count} active chunks)."
                    ),
                    details={"fts_count": data.fts_count, "chunk_count": data.chunk_count},
                )
            )
        self._add_count_issue(
            issues,
            data.empty_chunks,
            Severity.MEDIUM,
            IssueCode.EMPTY_CHUNKS,
            "active chunks are empty.",
        )
        self._add_count_issue(
            issues,
            data.small_chunks,
            Severity.LOW,
            IssueCode.SMALL_CHUNKS,
            f"active chunks are under {SMALL_CHUNK_CHARS} characters.",
        )
        self._add_count_issue(
            issues,
            data.large_chunks,
            Severity.MEDIUM,
            IssueCode.LARGE_CHUNKS,
            f"active chunks are over {LARGE_CHUNK_CHARS} characters.",
        )
        if data.duplicate_chunks.group_count:
            issues.append(
                HealthIssue(
                    severity=Severity.LOW,
                    code=IssueCode.DUPLICATE_CHUNKS,
                    message=(
                        f"Found {data.duplicate_chunks.group_count} duplicate exact chunk "
                        "text groups."
                    ),
                    details={
                        "duplicate_chunk_count": data.duplicate_chunks.duplicate_chunk_count,
                        "examples": data.duplicate_chunks.examples,
                    },
                )
            )

        self._add_document_content_issues(issues, data.documents)
        checked_files, skipped_file_checks = self._add_filesystem_issues(issues, data)
        self._add_source_issues(issues)
        self._add_sync_state_issues(issues, data.sync_states)

        embedding_coverage = data.embedding_count / data.chunk_count if data.chunk_count else 0.0
        fts_coverage = (
            data.fts_count / data.chunk_count
            if data.fts_count is not None and data.chunk_count
            else None
        )
        sorted_issues = _sorted_issues(issues)
        return HealthReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            backend=data.backend,
            database_path=data.database_path,
            source=self.source_filter.label if self.source_filter else None,
            document_count=data.document_count,
            chunk_count=data.chunk_count,
            embedding_count=data.embedding_count,
            fts_count=data.fts_count,
            embedding_coverage=embedding_coverage,
            fts_coverage=fts_coverage,
            document_asset_counts=data.document_asset_counts,
            chunk_asset_counts=data.chunk_asset_counts,
            chunk_modality_counts=data.chunk_modality_counts,
            embedding_counts=data.embedding_counts,
            issue_counts=_issue_counts(sorted_issues),
            issues=sorted_issues,
            checked_files=checked_files,
            skipped_file_checks=skipped_file_checks,
        )

    def _inspector(self) -> HealthInspector:
        if self.storage_backend == "postgres":
            return PostgresHealthInspector(source_filter=self.source_filter)
        return SQLiteHealthInspector(self.database_path, source_filter=self.source_filter)

    def _add_count_issue(
        self,
        issues: list[HealthIssue],
        summary: PathCount,
        severity: Severity,
        code: IssueCode,
        suffix: str,
    ) -> None:
        if not summary.count:
            return
        issues.append(
            HealthIssue(
                severity=severity,
                code=code,
                message=f"{summary.count} {suffix}",
                path=summary.paths[0] if len(summary.paths) == 1 else None,
                details={"paths": summary.paths},
            )
        )

    def _add_document_content_issues(
        self, issues: list[HealthIssue], documents: list[DocumentHealthRow]
    ) -> None:
        short_docs = [
            doc.path
            for doc in documents
            if doc.asset_type not in {"image", "pdf"}
            and 0 < doc.content_length < VERY_SHORT_DOCUMENT_CHARS
        ]
        if short_docs:
            issues.append(
                HealthIssue(
                    severity=Severity.LOW,
                    code=IssueCode.SHORT_DOCUMENT,
                    message=(
                        f"{len(short_docs)} documents have under "
                        f"{VERY_SHORT_DOCUMENT_CHARS} extracted characters."
                    ),
                    details={"paths": short_docs[:ISSUE_SAMPLE_LIMIT]},
                )
            )

        short_pdfs: list[str] = []
        image_without_ocr: list[str] = []
        code_without_symbols: list[str] = []
        for doc in documents:
            if doc.asset_type == "pdf":
                page_count = int(doc.metadata.get("page_count") or 1)
                min_chars = max(SHORT_PDF_MIN_CHARS, page_count * SHORT_PDF_CHARS_PER_PAGE)
                if doc.content_length < min_chars:
                    short_pdfs.append(doc.path)
            elif doc.asset_type == "image":
                if not _document_has_ocr_text(doc):
                    image_without_ocr.append(doc.path)
            elif doc.asset_type == "code":
                symbols = doc.metadata.get("symbols")
                if isinstance(symbols, list) and not symbols:
                    code_without_symbols.append(doc.path)

        if short_pdfs:
            issues.append(
                HealthIssue(
                    severity=Severity.MEDIUM,
                    code=IssueCode.SHORT_PDF_TEXT,
                    message=f"{len(short_pdfs)} PDFs have very little extracted text.",
                    details={"paths": short_pdfs[:ISSUE_SAMPLE_LIMIT]},
                )
            )
        if image_without_ocr:
            issues.append(
                HealthIssue(
                    severity=Severity.LOW,
                    code=IssueCode.IMAGE_WITHOUT_OCR,
                    message=f"{len(image_without_ocr)} images do not have OCR text in metadata.",
                    details={"paths": image_without_ocr[:ISSUE_SAMPLE_LIMIT]},
                )
            )
        if code_without_symbols:
            issues.append(
                HealthIssue(
                    severity=Severity.MEDIUM,
                    code=IssueCode.CODE_WITHOUT_SYMBOLS,
                    message=f"{len(code_without_symbols)} code files have no detected symbols.",
                    details={"paths": code_without_symbols[:ISSUE_SAMPLE_LIMIT]},
                )
            )

    def _add_filesystem_issues(
        self, issues: list[HealthIssue], data: InspectorData
    ) -> tuple[int, int]:
        local_sources = _local_source_paths(self.sources)
        if self.source_filter is not None:
            local_sources.append(self.source_filter)

        local_docs = [
            doc
            for doc in data.documents
            if _is_local_document(doc, data.source_file_mtimes, local_sources)
        ]
        skipped = max(0, len(local_docs) - self.file_check_limit)
        checked = 0
        missing: list[str] = []
        changed: list[str] = []
        for doc in local_docs[: self.file_check_limit]:
            checked += 1
            path = Path(doc.path)
            try:
                stat = path.stat()
            except OSError:
                missing.append(doc.path)
                continue
            indexed_mtime = _file_mtime_for_document(doc, data.source_file_mtimes)
            if indexed_mtime is not None and abs(stat.st_mtime - indexed_mtime) > 1e-6:
                changed.append(doc.path)

        if missing:
            issues.append(
                HealthIssue(
                    severity=Severity.MEDIUM,
                    code=IssueCode.INDEXED_FILE_MISSING,
                    message=f"{len(missing)} indexed local files are missing on disk.",
                    details={"paths": missing[:ISSUE_SAMPLE_LIMIT]},
                )
            )
        if changed:
            issues.append(
                HealthIssue(
                    severity=Severity.MEDIUM,
                    code=IssueCode.INDEXED_FILE_CHANGED,
                    message=f"{len(changed)} indexed local files changed since indexing.",
                    details={"paths": changed[:ISSUE_SAMPLE_LIMIT]},
                )
            )
        return checked, skipped

    def _add_source_issues(self, issues: list[HealthIssue]) -> None:
        for source in self.sources:
            if not _is_local_source(source):
                continue
            normalized = normalize_source_filter(source)
            if normalized is None:
                continue
            if self.source_filter is not None and normalized.path != self.source_filter.path:
                continue
            if not Path(normalized.path).exists():
                issues.append(
                    HealthIssue(
                        severity=Severity.LOW,
                        code=IssueCode.SOURCE_PATH_MISSING,
                        message=f"Registered source path is missing: {normalized.label}",
                        path=normalized.path,
                        details={"source": normalized.label},
                    )
                )

    def _add_sync_state_issues(
        self, issues: list[HealthIssue], sync_states: list[SyncStateHealthRow]
    ) -> None:
        for state in sync_states:
            if state.last_error:
                issues.append(
                    HealthIssue(
                        severity=Severity.MEDIUM,
                        code=IssueCode.SYNC_STATE_ERROR,
                        message=f"Source '{state.source_key}' has a recorded sync error.",
                        details={
                            "source_key": state.source_key,
                            "status": state.status,
                            "last_error": state.last_error,
                            "documents_seen": state.documents_seen,
                            "chunks_written": state.chunks_written,
                        },
                    )
                )


def scan_library_health(
    *,
    database_path: str | None = None,
    storage_backend: str | None = None,
    source_filter: HealthSourceFilter | None = None,
    sources: list[dict[str, Any]] | None = None,
    file_check_limit: int = FILE_CHECK_LIMIT,
) -> HealthReport:
    """Run a read-only library health scan."""
    return HealthScanner(
        database_path=database_path,
        storage_backend=storage_backend,
        source_filter=source_filter,
        sources=sources,
        file_check_limit=file_check_limit,
    ).scan()
