"""
Storage protocol bundle for the v0.14 single forward path.

A storage backend is described by four capability protocols that together form
the ``Storage`` bundle:

* :class:`MetadataStore` -- document metadata read access.
* :class:`VectorStore` -- vector similarity search.
* :class:`FTSStore` -- full-text (keyword) search.
* :class:`StateStore` -- connector sync cursors (the ``sync_state`` table).

The concrete :class:`~librarian.storage.sqlite_storage.SQLiteStorage` implements
the bundle. A future ``PostgresStorage`` (DEV-472) implements the same protocols
so callers are substrate-agnostic. The existing ``Database``, ``VectorStore`` and
``FTSStore`` classes satisfy the read-side protocols where applicable.

These are structural ``typing.Protocol`` definitions, so any object with the
right methods satisfies them -- no explicit subclassing required.
"""

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from librarian.storage.fts_store import FTSSearchResult
    from librarian.storage.vector_store import VectorSearchResult
    from librarian.storage.write_models import PreparedDocument
    from librarian.types import Document, EmbeddingModality

__all__ = [
    "ChunkContext",
    "FTSStore",
    "MetadataStore",
    "StateStore",
    "Storage",
    "SyncState",
    "TxnHandle",
    "VectorStore",
]

# An opaque handle to an open storage transaction. Concrete backends bind it to
# whatever their ``transaction()`` yields (a ``sqlite3.Connection`` today, an
# async session for a future Postgres backend). Callers never inspect it; they
# only pass it back into the write methods so a content write and its cursor
# advance land in the same transaction.
TxnHandle = Any


@dataclass
class SyncState:
    """One row of the ``sync_state`` table: a connector source's cursor + stats.

    ``source_key`` identifies a single sync stream (typically the connector
    name, optionally namespaced per source). ``cursor`` is the opaque
    JSON-serializable checkpoint the connector understands.
    """

    source_key: str
    cursor: dict[str, Any] = field(default_factory=dict)
    status: str = "idle"
    last_success_at: datetime | None = None
    last_attempt_at: datetime | None = None
    last_error: str | None = None
    documents_seen: int = 0
    chunks_written: int = 0
    config_version: int = 0


@dataclass
class ChunkContext:
    """One chunk in an :meth:`MetadataStore.get_chunk_context` window.

    Carries both identity forms: the deterministic public ``chunk_id`` (TEXT,
    what search returns) and the internal ``internal_id`` surrogate
    (``chunks.id``). ``document_id`` is the internal ``documents.id``.
    """

    chunk_id: str | None
    internal_id: int
    document_id: int
    document_path: str
    content: str
    heading_path: str | None
    chunk_index: int
    asset_type: str
    chunk_source_uri: str | None
    deleted_at: str | None = None


@runtime_checkable
class MetadataStore(Protocol):
    """Read access to document metadata."""

    def get_document_by_path(self, path: str) -> "Document | None": ...

    def get_document_by_id(self, doc_id: int) -> "Document | None": ...

    def list_documents(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> "list[Document]": ...

    def get_document_ids_in_timerange(
        self, start_date: datetime, end_date: datetime
    ) -> list[int]: ...

    def get_chunk_public_fields(self, chunk_ids: list[int]) -> dict[int, dict[str, Any]]: ...

    def get_chunk_context(
        self,
        chunk_id: str,
        before: int = 2,
        after: int = 2,
        include_deleted: bool = False,
    ) -> list[ChunkContext]: ...

    def chunk_exists(self, chunk_id: str) -> bool: ...

    def get_stats(self) -> dict[str, Any]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Vector similarity search.

    Beyond the primary :meth:`search`, the retrieval layer relies on
    modality-scoped search and single-chunk embedding lookup (for cross-modal
    queries and MMR diversification), so both backends provide them.
    """

    def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.0,
        include_deleted: bool = False,
    ) -> "list[VectorSearchResult]": ...

    def search_by_modality(
        self,
        query_embedding: list[float],
        modality: "EmbeddingModality",
        limit: int = 10,
        min_similarity: float = 0.0,
        include_deleted: bool = False,
    ) -> "list[VectorSearchResult]": ...

    def get_embedding(
        self,
        chunk_id: int,
        modality: "EmbeddingModality | None" = None,
    ) -> list[float] | None: ...


@runtime_checkable
class FTSStore(Protocol):
    """Full-text (keyword) search."""

    def search(
        self,
        query: str,
        limit: int = 10,
        snippet_length: int = 64,
        include_deleted: bool = False,
    ) -> "list[FTSSearchResult]": ...


@runtime_checkable
class StateStore(Protocol):
    """Connector sync-cursor persistence (the ``sync_state`` table)."""

    def get_sync_state(self, source_key: str) -> SyncState | None: ...

    def put_sync_state(self, state: SyncState, conn: "TxnHandle | None" = None) -> None: ...

    def get_file_mtime(self, source_key: str, path: str) -> float | None: ...

    def set_file_mtime(
        self, source_key: str, path: str, mtime: float, conn: "TxnHandle | None" = None
    ) -> None: ...


@runtime_checkable
class Storage(Protocol):
    """The full storage bundle a connector + orchestrator runs against.

    Bundles the four capability protocols and adds:

    * :meth:`migrate` -- create/upgrade librarian-owned tables.
    * :meth:`transaction` -- a context manager yielding a handle on which
      content writes and cursor advances happen atomically.
    * :meth:`write_upsert` / :meth:`soft_delete_document` -- the document write
      paths the orchestrator drives inside that transaction.

    Sync-cursor persistence lives on the composed :attr:`state` store rather
    than being duplicated here, so callers reach it via ``storage.state``.
    """

    # Read-only properties (rather than bare attributes) so a concrete backend
    # whose stores are subtypes of these protocols still satisfies the bundle:
    # mutable protocol attributes are invariant, properties are covariant.
    @property
    def metadata(self) -> MetadataStore: ...

    @property
    def vectors(self) -> VectorStore: ...

    @property
    def fts(self) -> FTSStore: ...

    @property
    def state(self) -> StateStore: ...

    def migrate(self) -> None: ...

    def transaction(self) -> AbstractContextManager["TxnHandle"]: ...

    def existing_text_chunks(
        self, document_id: str
    ) -> "dict[str, tuple[str, list[float] | None]]": ...

    def write_upsert(self, conn: "TxnHandle", prepared: "PreparedDocument") -> None: ...

    def soft_delete_document(
        self, conn: "TxnHandle", document_id: str, reason: str | None
    ) -> int: ...
