"""
Substrate-parity contract suite for the v0.14 ``Storage`` bundle.

The same assertions run against every backend via the parameterized ``backend``
fixture: today SQLite (always) and Postgres (when a test database is reachable
through ``TEST_POSTGRES_DSN``; otherwise skipped). This is the executable form
of the DEV-472 acceptance criteria -- the storage abstraction is only
"genuinely pluggable" if one suite passes unchanged on two substrates.

Each backend is wrapped in a small ``StorageHarness`` so the tests can make
backend-neutral row-count assertions with identical SQL. Postgres runs are
isolated in a throwaway schema per test so commits/rollbacks are exercised for
real (no transaction-wrapping that would mask the atomicity guarantees).
"""

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from librarian import ids
from librarian.storage.write_models import PreparedChunk, PreparedDocument
from librarian.types import AssetType, EmbeddingModality

from .conftest import FakeEmbedder

CONNECTOR = "contract"
SOURCE_TYPE = "doc"
EMBED_DIM = 384  # matches conftest's EMBEDDING_DIMENSION / FakeEmbedder


@dataclass
class StorageHarness:
    """A storage bundle plus backend-neutral inspection helpers."""

    storage: Any
    backend: str

    def count(self, table: str, where: str | None = None) -> int:
        sql = f"SELECT COUNT(*) AS n FROM {table}"  # noqa: S608 - fixed table literals
        if where:
            sql += f" WHERE {where}"
        with self.storage.database._connection() as conn:
            return conn.execute(sql).fetchone()["n"]

    def chunk_ids(self) -> list[str]:
        with self.storage.database._connection() as conn:
            rows = conn.execute("SELECT chunk_id FROM chunks ORDER BY id").fetchall()
        return [row["chunk_id"] for row in rows]

    def internal_chunk_ids(self) -> list[int]:
        """Internal ``chunks.id`` PKs (what the public-field/embedding reads key on)."""
        with self.storage.database._connection() as conn:
            rows = conn.execute("SELECT id FROM chunks ORDER BY id").fetchall()
        return [row["id"] for row in rows]


# =============================================================================
# Fixtures
# =============================================================================


def _make_sqlite(tmp_path: Path) -> StorageHarness:
    from librarian.storage.database import Database
    from librarian.storage.sqlite_storage import SQLiteStorage

    storage = SQLiteStorage(database=Database(str(tmp_path / "contract.db")))
    storage.migrate()
    return StorageHarness(storage=storage, backend="sqlite")


@contextmanager
def _make_postgres(dsn: str) -> Iterator[StorageHarness]:
    """A migrated PostgresStorage in a throwaway schema, dropped on exit.

    The single home for the per-test schema setup/teardown, shared by the
    parameterized ``backend`` fixture and the Postgres-only standalone tests.
    """
    from librarian.storage.postgres import PostgresStorage

    schema = f"librarian_test_{uuid.uuid4().hex}"
    storage = PostgresStorage(dsn=dsn, schema=schema)
    storage.migrate()
    try:
        yield StorageHarness(storage=storage, backend="postgres")
    finally:
        from psycopg import sql

        try:
            with storage.database._connection() as conn:
                conn.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
                )
        finally:
            storage.database.close()


@pytest.fixture(params=["sqlite", "postgres"])
def backend(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StorageHarness]:
    if request.param == "sqlite":
        yield _make_sqlite(tmp_path)
        return

    dsn = os.getenv("TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("TEST_POSTGRES_DSN not set; skipping Postgres parity run")
    try:
        import psycopg  # noqa: F401
    except ImportError:
        pytest.skip("psycopg not installed; skipping Postgres parity run")
    with _make_postgres(dsn) as harness:
        yield harness


# =============================================================================
# Helpers
# =============================================================================


def _prepare(native_id: str, text: str, embedder: FakeEmbedder) -> PreparedDocument:
    """Build a single-chunk PreparedDocument with deterministic ids."""
    chunk_native = f"{native_id}#0"
    embedding = embedder.embed_documents([text])[0]
    chunk = PreparedChunk(
        chunk_id=ids.chunk_id(CONNECTOR, SOURCE_TYPE, chunk_native),
        content=text,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        asset_type=AssetType.TEXT,
        modality=EmbeddingModality.TEXT,
        embedding=embedding,
        model_version=embedder.model_name,
    )
    return PreparedDocument(
        document_id=ids.document_id(CONNECTOR, SOURCE_TYPE, native_id),
        path=native_id,
        title=native_id,
        content=text,
        metadata={"source": "contract"},
        asset_type=AssetType.TEXT,
        document_size=len(text),
        chunks=[chunk],
    )


def _prepare_multi(native_id: str, texts: list[str], embedder: FakeEmbedder) -> PreparedDocument:
    """Build a multi-chunk PreparedDocument with deterministic ids + source URIs.

    Used by the ``expand_context`` window tests and the batched-insert test,
    which both need several ordered chunks in one document.
    """
    content = "\n\n".join(texts)
    doc_uri = f"file:///{native_id}"
    chunks = [
        PreparedChunk(
            chunk_id=ids.chunk_id(CONNECTOR, SOURCE_TYPE, f"{native_id}#{i}"),
            content=text,
            chunk_index=i,
            start_char=0,
            end_char=len(text),
            chunk_source_uri=f"{doc_uri}#chunk={i}",
            asset_type=AssetType.TEXT,
            modality=EmbeddingModality.TEXT,
            embedding=embedder.embed_documents([text])[0],
            model_version=embedder.model_name,
        )
        for i, text in enumerate(texts)
    ]
    return PreparedDocument(
        document_id=ids.document_id(CONNECTOR, SOURCE_TYPE, native_id),
        path=native_id,
        title=native_id,
        content=content,
        metadata={"source": "contract"},
        asset_type=AssetType.TEXT,
        document_source_uri=doc_uri,
        document_size=len(content),
        chunks=chunks,
    )


def _write(harness: StorageHarness, prepared: PreparedDocument) -> None:
    with harness.storage.transaction() as conn:
        harness.storage.write_upsert(conn, prepared)


# =============================================================================
# Contract tests (run against every backend)
# =============================================================================


def test_write_upsert_persists_document_and_chunks(backend: StorageHarness) -> None:
    embedder = FakeEmbedder()
    _write(backend, _prepare("m1", "hello world", embedder))

    assert backend.count("documents") == 1
    assert backend.count("chunks") == 1
    assert backend.count("chunk_embeddings") == 1

    doc = backend.storage.metadata.get_document_by_path("m1")
    assert doc is not None
    assert doc.content == "hello world"
    assert doc.metadata.get("source") == "contract"


def test_write_upsert_persists_all_chunks_of_multi_chunk_document(
    backend: StorageHarness,
) -> None:
    """A multi-chunk document writes every chunk + embedding (batched on Postgres).

    Locks in the batched-insert path: all N chunks land with their embeddings,
    each searchable, with the deterministic ids preserved -- the same invariants
    a per-chunk loop guaranteed, now under a single multi-row INSERT.
    """
    embedder = FakeEmbedder()
    texts = ["alpha one", "beta two", "gamma three", "delta four", "epsilon five"]
    prepared = _prepare_multi("multi", texts, embedder)

    _write(backend, prepared)

    assert backend.count("documents") == 1
    assert backend.count("chunks") == len(texts)
    assert backend.count("chunk_embeddings") == len(texts)
    # Every chunk's deterministic id round-trips.
    assert set(backend.chunk_ids()) == {c.chunk_id for c in prepared.chunks}

    # Each chunk's own embedding is its own nearest neighbor (embeddings paired
    # to the right chunk, not shuffled by the batch insert).
    for chunk, text in zip(prepared.chunks, texts, strict=True):
        results = backend.storage.vectors.search(
            embedder.embed_documents([text])[0], limit=1, min_similarity=-1.0
        )
        assert results
        assert results[0].content == chunk.content


def test_reingest_is_idempotent(backend: StorageHarness) -> None:
    embedder = FakeEmbedder()
    prepared = _prepare("m1", "hello world", embedder)

    _write(backend, prepared)
    ids_first = set(backend.chunk_ids())
    _write(backend, _prepare("m1", "hello world", embedder))

    assert backend.count("documents") == 1
    assert backend.count("chunks") == 1  # replaced, not duplicated
    assert backend.count("chunk_embeddings") == 1
    assert set(backend.chunk_ids()) == ids_first  # same deterministic ids


def test_existing_text_chunks_returns_live_text_embeddings(backend: StorageHarness) -> None:
    embedder = FakeEmbedder()
    prepared = _prepare("m1", "hello world", embedder)
    _write(backend, prepared)

    existing = backend.storage.existing_text_chunks(prepared.document_id)

    assert set(existing) == {prepared.chunks[0].chunk_id}
    content, embedding = existing[prepared.chunks[0].chunk_id]
    assert content == "hello world"
    assert embedding == pytest.approx(prepared.chunks[0].embedding)


def test_write_and_cursor_advance_commit_together(backend: StorageHarness) -> None:
    from librarian.storage.protocols import SyncState

    embedder = FakeEmbedder()
    prepared = _prepare("m1", "hello world", embedder)

    with backend.storage.transaction() as conn:
        backend.storage.write_upsert(conn, prepared)
        backend.storage.put_sync_state(
            SyncState(source_key="contract", cursor={"i": 1}, status="ok"),
            conn=conn,
        )

    assert backend.count("chunks") == 1
    state = backend.storage.get_sync_state("contract")
    assert state is not None
    assert state.cursor == {"i": 1}


def test_crash_rolls_back_content_and_cursor(backend: StorageHarness) -> None:
    """The transactional acceptance test: a crash after the content write but
    before the commit must roll *both* the content and the cursor advance back."""
    from librarian.storage.protocols import SyncState

    embedder = FakeEmbedder()
    prepared = _prepare("m1", "hello world", embedder)

    with pytest.raises(RuntimeError, match="boom"), backend.storage.transaction() as conn:
        backend.storage.write_upsert(conn, prepared)
        backend.storage.put_sync_state(SyncState(source_key="contract", cursor={"i": 1}), conn=conn)
        raise RuntimeError("boom")  # simulated crash before commit

    # Nothing committed: no orphan rows, no orphan cursor.
    assert backend.count("documents") == 0
    assert backend.count("chunks") == 0
    assert backend.count("chunk_embeddings") == 0
    assert backend.storage.get_sync_state("contract") is None


def test_list_documents_pagination(backend: StorageHarness) -> None:
    """list_documents bounds the result set with limit/offset and a stable order.

    The deliberate ``MetadataStore`` protocol change: both backends order
    newest-first with an ``id`` tiebreak, so paging is deterministic across the
    substrate even when ``updated_at`` ties at coarse resolution.
    """
    embedder = FakeEmbedder()
    for i in range(5):
        _write(backend, _prepare(f"p{i}", f"document number {i}", embedder))

    meta = backend.storage.metadata

    # Full list (no bound) returns all five.
    everything = meta.list_documents()
    assert len(everything) == 5
    full_order = [d.path for d in everything]

    # limit caps the count; the prefix matches the unbounded order.
    first_two = meta.list_documents(limit=2)
    assert [d.path for d in first_two] == full_order[:2]

    # offset skips from the same stable order; limit+offset partition the list.
    next_two = meta.list_documents(limit=2, offset=2)
    assert [d.path for d in next_two] == full_order[2:4]

    # offset past the end yields nothing.
    assert meta.list_documents(limit=2, offset=10) == []

    # offset without a limit is honored (not silently ignored) on both backends.
    assert [d.path for d in meta.list_documents(offset=2)] == full_order[2:]


def test_delete_document_by_path_removes_document_and_chunks(backend: StorageHarness) -> None:
    """The admin hard-delete is backend-agnostic: it drops the document, its
    chunks and embeddings, and reports whether anything was removed."""
    embedder = FakeEmbedder()
    _write(backend, _prepare("keep", "kept document", embedder))
    _write(backend, _prepare("drop", "doomed document", embedder))

    assert backend.storage.delete_document_by_path("drop") is True

    assert backend.count("documents") == 1
    assert backend.count("chunks") == 1
    assert backend.count("chunk_embeddings") == 1
    assert backend.storage.metadata.get_document_by_path("drop") is None
    assert backend.storage.metadata.get_document_by_path("keep") is not None

    # Deleting a path that isn't present is a no-op that reports False.
    assert backend.storage.delete_document_by_path("drop") is False


def test_soft_delete_tombstones_without_removing_rows(backend: StorageHarness) -> None:
    embedder = FakeEmbedder()
    prepared = _prepare("m1", "hello world", embedder)
    _write(backend, prepared)

    with backend.storage.transaction() as conn:
        tombstoned = backend.storage.soft_delete_document(conn, prepared.document_id, "gone")

    assert tombstoned == 1
    assert backend.count("chunks") == 1  # row still present
    assert backend.count("chunks", "deleted_at IS NOT NULL") == 1
    assert backend.count("chunks", "deletion_reason = 'gone'") == 1


def test_soft_delete_hides_document_from_search(backend: StorageHarness) -> None:
    """A tombstoned document must disappear from BOTH vector and FTS search.

    soft_delete only sets deleted_at (the embedding/FTS rows survive), so each
    search store must filter ``deleted_at IS NULL``. This is the assertion that
    catches a missing filter on either backend.
    """
    embedder = FakeEmbedder()
    prepared = _prepare("m1", "the quick brown fox", embedder)
    _write(backend, prepared)

    query = embedder.embed_documents(["the quick brown fox"])[0]
    # Present before deletion.
    assert backend.storage.vectors.search(query, limit=5, min_similarity=-1.0)
    assert backend.storage.fts.search("quick fox", limit=5)

    with backend.storage.transaction() as conn:
        backend.storage.soft_delete_document(conn, prepared.document_id, "gone")

    # Gone from both search paths after the tombstone.
    assert backend.storage.vectors.search(query, limit=5, min_similarity=-1.0) == []
    assert backend.storage.fts.search("quick fox", limit=5) == []


def test_sync_state_roundtrip(backend: StorageHarness) -> None:
    from librarian.storage.protocols import SyncState

    assert backend.storage.get_sync_state("nope") is None

    backend.storage.put_sync_state(
        SyncState(
            source_key="contract",
            cursor={"mtimes": {"/a": 1.5}},
            status="ok",
            documents_seen=3,
            chunks_written=7,
            config_version=2,
        )
    )
    state = backend.storage.get_sync_state("contract")
    assert state is not None
    assert state.cursor == {"mtimes": {"/a": 1.5}}
    assert state.documents_seen == 3
    assert state.chunks_written == 7
    assert state.config_version == 2

    # Upsert overwrites in place (no duplicate row).
    backend.storage.put_sync_state(SyncState(source_key="contract", cursor={"i": 9}, status="ok"))
    assert backend.count("sync_state") == 1
    assert backend.storage.get_sync_state("contract").cursor == {"i": 9}  # type: ignore[union-attr]


def test_file_mtime_roundtrip(backend: StorageHarness) -> None:
    path = "contract/doc.md"

    assert backend.storage.state.get_file_mtime("contract", path) is None

    backend.storage.state.set_file_mtime("contract", path, 123.456)
    assert backend.storage.state.get_file_mtime("contract", path) == pytest.approx(123.456)

    with backend.storage.transaction() as conn:
        backend.storage.state.set_file_mtime("contract", path, 789.0, conn=conn)

    assert backend.storage.state.get_file_mtime("contract", path) == pytest.approx(789.0)


def test_vector_search_finds_written_chunk(backend: StorageHarness) -> None:
    embedder = FakeEmbedder()
    _write(backend, _prepare("m1", "the quick brown fox", embedder))
    _write(backend, _prepare("m2", "lorem ipsum dolor", embedder))

    # Query with m1's own (deterministic) document embedding so the nearest
    # neighbor is unambiguous across substrates -- FakeEmbedder vectors are
    # otherwise uncorrelated random noise.
    query = embedder.embed_documents(["the quick brown fox"])[0]
    results = backend.storage.vectors.search(query, limit=5, min_similarity=-1.0)

    assert results
    assert results[0].document_path == "m1"
    # similarity = 1 - distance is in a sane cosine range, best match first.
    assert all(-1.01 <= (1.0 - r.distance) <= 1.01 for r in results)
    assert (1.0 - results[0].distance) == pytest.approx(1.0, abs=1e-3)


def test_fts_search_finds_written_chunk(backend: StorageHarness) -> None:
    embedder = FakeEmbedder()
    _write(backend, _prepare("m1", "the quick brown fox", embedder))
    _write(backend, _prepare("m2", "lorem ipsum dolor", embedder))

    results = backend.storage.fts.search("quick fox", limit=5)

    assert results
    assert any(r.document_path == "m1" for r in results)


def test_migrate_is_idempotent(backend: StorageHarness) -> None:
    embedder = FakeEmbedder()
    _write(backend, _prepare("m1", "hello world", embedder))

    # Re-running migrate on a populated database is a no-op (no data loss).
    backend.storage.migrate()
    backend.storage.migrate()

    assert backend.count("documents") == 1
    assert backend.count("chunks") == 1
    assert backend.count("chunk_embeddings") == 1


def test_stats_reports_counts(backend: StorageHarness) -> None:
    embedder = FakeEmbedder()
    _write(backend, _prepare("m1", "hello world", embedder))

    stats = backend.storage.metadata.get_stats()
    assert stats["document_count"] == 1
    assert stats["chunk_count"] == 1
    assert stats["embedding_count"] == 1


def test_read_protocol_parity(backend: StorageHarness) -> None:
    """Exercise the read surface most prone to substrate-specific SQL drift.

    These methods carry the timezone/CASE, modality-table, embedding-text-parse
    and id-lookup divergences, so they're where parity is most likely to break
    silently if only one backend is tested.
    """
    embedder = FakeEmbedder()
    p1 = _prepare("m1", "the quick brown fox", embedder)
    p2 = _prepare("m2", "lorem ipsum dolor", embedder)
    _write(backend, p1)
    _write(backend, p2)

    meta = backend.storage.metadata

    # get_document_by_id round-trips with get_document_by_path.
    doc = meta.get_document_by_path("m1")
    assert doc is not None
    by_id = meta.get_document_by_id(doc.id)
    assert by_id is not None
    assert by_id.path == "m1"
    assert meta.get_document_by_id(10_000_000) is None

    # list_documents with a window spanning "now" returns both documents.
    now = datetime.now(timezone.utc)
    start, end = now - timedelta(days=1), now + timedelta(days=1)
    docs = meta.list_documents(start_date=start, end_date=end)
    assert {d.path for d in docs} == {"m1", "m2"}

    # get_document_ids_in_timerange (updated_at branch -- no file_mtime set).
    in_range = meta.get_document_ids_in_timerange(start, end)
    ids2 = meta.get_document_by_path("m2")
    assert ids2 is not None
    assert set(in_range) == {doc.id, ids2.id}

    # get_chunk_public_fields keyed by internal chunks.id.
    internal_ids = backend.internal_chunk_ids()
    assert len(internal_ids) == 2
    public = meta.get_chunk_public_fields(internal_ids)
    assert set(public) == set(internal_ids)
    assert all(fields["chunk_index"] == 0 for fields in public.values())
    assert meta.get_chunk_public_fields([]) == {}

    # search_by_modality(TEXT) finds the matching chunk.
    query = embedder.embed_documents(["the quick brown fox"])[0]
    modality_hits = backend.storage.vectors.search_by_modality(
        query, EmbeddingModality.TEXT, limit=5, min_similarity=-1.0
    )
    assert any(r.document_path == "m1" for r in modality_hits)

    # get_embedding returns the stored TEXT vector for a chunk (embedding::text
    # parse path on Postgres, blob deserialize on SQLite).
    m1_chunk_internal_id = internal_ids[0]
    stored = backend.storage.vectors.get_embedding(m1_chunk_internal_id)
    assert stored is not None
    assert len(stored) == EMBED_DIM
    assert meta.get_chunk_public_fields([m1_chunk_internal_id])  # sanity: id exists


def test_postgres_connection_pool_is_bounded_and_concurrent() -> None:
    """The Postgres backend serves reads from a bounded, thread-shared pool.

    Asserts two things the thread-local design couldn't give us: (1) the pool
    has a hard ``max_size`` ceiling, so concurrent request fan-out can't drift
    toward the server's ``max_connections``; (2) many threads can read through
    one shared pool concurrently and each get correct results (connections are
    checked out per-operation, not pinned per-thread for the process lifetime).
    """
    dsn = os.getenv("TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("TEST_POSTGRES_DSN not set")

    import threading

    import psycopg  # noqa: F401

    with _make_postgres(dsn) as harness:
        storage = harness.storage
        embedder = FakeEmbedder()
        with storage.transaction() as conn:
            storage.write_upsert(conn, _prepare("m1", "hello world", embedder))

        pool = storage.database._get_pool()
        assert pool.max_size <= max(1, int(os.getenv("POSTGRES_POOL_MAX_SIZE", "10")))

        # Fan a read out across more threads than the pool's max_size: every
        # thread must still get the right answer, with checkouts queued behind
        # the ceiling rather than each opening its own connection.
        results: list[int] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def _read() -> None:
            try:
                stats = storage.metadata.get_stats()
                with lock:
                    results.append(stats["document_count"])
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=_read) for _ in range(pool.max_size * 3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert results == [1] * len(threads)


def test_postgres_transaction_is_not_reentrant() -> None:
    """A nested ``transaction()`` raises rather than silently checking out a
    second pooled connection and breaking the atomicity guarantee."""
    dsn = os.getenv("TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("TEST_POSTGRES_DSN not set")

    import psycopg  # noqa: F401

    with _make_postgres(dsn) as harness:
        storage = harness.storage
        # Opening a second transaction while the first is still active raises on
        # entering the inner context; pytest.raises (wrapping it) catches that,
        # then the outer transaction exits cleanly.
        with (
            storage.transaction(),
            pytest.raises(RuntimeError, match="not re-entrant"),
            storage.transaction(),
        ):
            pass
        # After the guard fires and the outer block exits cleanly, the parked
        # connection is released and new transactions work again.
        embedder = FakeEmbedder()
        with storage.transaction() as conn:
            storage.write_upsert(conn, _prepare("m1", "hello world", embedder))
        assert harness.count("documents") == 1


# =============================================================================
# Slice 3: schema confirmation, model_version, expand_context, soft-delete opt-in
# =============================================================================

# Every v0.14 column the schema must carry, by table. Selecting them with
# ``LIMIT 0`` is a backend-neutral existence probe: it raises on a missing
# column on either substrate but reads no rows.
_V014_COLUMNS: dict[str, list[str]] = {
    "documents": ["document_id", "document_source_uri", "source_created_at"],
    "chunks": [
        "chunk_id",
        "chunk_index",
        "document_size",
        "source_created_at",
        "deleted_at",
        "deletion_reason",
        "document_source_uri",
        "chunk_source_uri",
    ],
    "chunk_embeddings": ["model_version"],
}


def test_v014_schema_columns_present(backend: StorageHarness) -> None:
    """All v0.14 columns exist on both substrates (acceptance criterion #1)."""
    with backend.storage.database._connection() as conn:
        for table, columns in _V014_COLUMNS.items():
            select_list = ", ".join(columns)
            # Raises on either substrate if any column is missing; reads no rows.
            conn.execute(f"SELECT {select_list} FROM {table} LIMIT 0")  # noqa: S608


def test_model_version_recorded_on_every_embedding(backend: StorageHarness) -> None:
    """Every ``chunk_embeddings`` row carries a non-null model_version."""
    embedder = FakeEmbedder()
    _write(backend, _prepare("m1", "hello world", embedder))
    _write(backend, _prepare_multi("m2", ["alpha", "beta", "gamma"], embedder))

    total = backend.count("chunk_embeddings")
    assert total == 4
    with_version = backend.count("chunk_embeddings", "model_version IS NOT NULL")
    assert with_version == total
    with backend.storage.database._connection() as conn:
        rows = conn.execute("SELECT DISTINCT model_version FROM chunk_embeddings").fetchall()
    assert {row["model_version"] for row in rows} == {embedder.model_name}


def test_get_chunk_context_returns_neighbors_in_source_order(backend: StorageHarness) -> None:
    """expand_context window: before=2/after=2 around a middle chunk -> 4 in order."""
    embedder = FakeEmbedder()
    prepared = _prepare_multi("doc", ["c0", "c1", "c2", "c3", "c4"], embedder)
    _write(backend, prepared)

    anchor = prepared.chunks[2].chunk_id  # chunk_index == 2
    neighbors = backend.storage.metadata.get_chunk_context(anchor, before=2, after=2)

    assert [n.chunk_index for n in neighbors] == [0, 1, 3, 4]  # anchor (2) excluded
    assert [n.content for n in neighbors] == ["c0", "c1", "c3", "c4"]
    # Source URIs and ids are preserved on the returned neighbors.
    assert neighbors[0].chunk_id == prepared.chunks[0].chunk_id
    assert neighbors[-1].chunk_source_uri == prepared.chunks[4].chunk_source_uri


def test_get_chunk_context_clips_at_document_boundary(backend: StorageHarness) -> None:
    embedder = FakeEmbedder()
    prepared = _prepare_multi("doc", ["c0", "c1", "c2"], embedder)
    _write(backend, prepared)

    # Anchor at the first chunk: no preceding neighbors, two following.
    head = backend.storage.metadata.get_chunk_context(
        prepared.chunks[0].chunk_id, before=2, after=2
    )
    assert [n.chunk_index for n in head] == [1, 2]

    # A single-chunk document has no neighbors at all.
    solo = _prepare("solo", "only chunk", embedder)
    _write(backend, solo)
    assert backend.storage.metadata.get_chunk_context(solo.chunks[0].chunk_id) == []


def test_get_chunk_context_unknown_chunk_returns_empty(backend: StorageHarness) -> None:
    assert backend.storage.metadata.get_chunk_context("does-not-exist") == []


def test_get_chunk_context_excludes_deleted_unless_opted_in(backend: StorageHarness) -> None:
    embedder = FakeEmbedder()
    prepared = _prepare_multi("doc", ["c0", "c1", "c2", "c3", "c4"], embedder)
    _write(backend, prepared)

    # Tombstone the whole document, then expand around a (now-deleted) anchor.
    with backend.storage.transaction() as conn:
        backend.storage.soft_delete_document(conn, prepared.document_id, "gone")

    anchor = prepared.chunks[2].chunk_id
    # Default: deleted neighbors filtered out even though the anchor still resolves.
    assert backend.storage.metadata.get_chunk_context(anchor, before=2, after=2) == []
    # Opt-in: the tombstoned neighbors come back, still in source order, each
    # carrying the deleted_at timestamp so callers can tell them from live rows.
    included = backend.storage.metadata.get_chunk_context(
        anchor, before=2, after=2, include_deleted=True
    )
    assert [n.chunk_index for n in included] == [0, 1, 3, 4]
    assert all(n.deleted_at for n in included)


def test_chunk_exists(backend: StorageHarness) -> None:
    """chunk_exists resolves live and tombstoned chunks, rejects unknown ids."""
    embedder = FakeEmbedder()
    prepared = _prepare("m1", "hello world", embedder)
    _write(backend, prepared)
    anchor = prepared.chunks[0].chunk_id

    assert backend.storage.metadata.chunk_exists(anchor) is True
    assert backend.storage.metadata.chunk_exists("nope") is False

    # A tombstoned chunk still "exists" (the row survives a soft delete).
    with backend.storage.transaction() as conn:
        backend.storage.soft_delete_document(conn, prepared.document_id, "gone")
    assert backend.storage.metadata.chunk_exists(anchor) is True


def test_search_include_deleted_opt_in(backend: StorageHarness) -> None:
    """Soft-deleted chunks are excluded by default and returned on opt-in.

    Covers both read paths (vector + FTS) on both substrates.
    """
    embedder = FakeEmbedder()
    prepared = _prepare("m1", "the quick brown fox", embedder)
    _write(backend, prepared)
    query = embedder.embed_documents(["the quick brown fox"])[0]

    with backend.storage.transaction() as conn:
        backend.storage.soft_delete_document(conn, prepared.document_id, "gone")

    # Default: hidden from both modalities.
    assert backend.storage.vectors.search(query, limit=5, min_similarity=-1.0) == []
    assert backend.storage.fts.search("quick fox", limit=5) == []

    # Opt-in: both modalities surface the tombstoned chunk again.
    assert backend.storage.vectors.search(query, limit=5, min_similarity=-1.0, include_deleted=True)
    assert backend.storage.fts.search("quick fox", limit=5, include_deleted=True)


def test_postgres_param_not_silently_skipped() -> None:
    """When TEST_POSTGRES_DSN is set, a Postgres run must actually happen.

    Guards against a misconfigured DSN (or a missing ``psycopg``) turning the
    parameterized Postgres run into a green-but-skipped no-op: with the DSN set,
    psycopg must import and the schema must migrate, or this fails loudly.
    """
    dsn = os.getenv("TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("TEST_POSTGRES_DSN not set")

    import psycopg  # noqa: F401  -- ImportError here should FAIL, not skip

    with _make_postgres(dsn) as harness:
        assert harness.storage.metadata.get_stats()["document_count"] == 0
