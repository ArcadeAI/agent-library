"""
Postgres v0.14 schema migration.

Creates the librarian-owned schema on Postgres, mirroring the SQLite v0.14
shape (see :mod:`librarian.storage.migrate`). Differences are substrate-native,
not semantic:

* embeddings live in ``vector(dim)`` columns (pgvector) instead of sqlite-vec
  virtual tables;
* full-text search uses a ``tsvector`` generated column + GIN index instead of
  an FTS5 external-content table with sync triggers (the generated column keeps
  the index in lock-step with ``chunks.content`` automatically);
* ``cursor``/``metadata`` are ``JSONB``.

Every table/index is created with ``IF NOT EXISTS``, so ``migrate()`` is
idempotent on a fresh database: re-running it is a no-op. (There are no
``ALTER TABLE ... ADD COLUMN`` steps yet, so an older table that predates a new
column would not be upgraded in place -- add explicit ALTERs here when the shape
next changes.)
"""

import logging
import re
from typing import Any

from librarian.config import (
    CODE_EMBEDDING_DIMENSION,
    POSTGRES_FTS_LANGUAGE,
    VISION_EMBEDDING_DIMENSION,
    get_effective_embedding_dimension,
)
from librarian.storage.schema_version import SchemaRebuildRequired

logger = logging.getLogger(__name__)

# A regconfig embedded in DDL (the generated column) can't be a bound parameter,
# so restrict it to a plain identifier to keep it injection-safe.
_REGCONFIG_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _fts_language() -> str:
    lang = POSTGRES_FTS_LANGUAGE
    if not _REGCONFIG_RE.match(lang):
        raise ValueError(
            f"Invalid POSTGRES_FTS_LANGUAGE {lang!r}; expected a text-search "
            "config identifier such as 'english' or 'simple'."
        )
    return lang


def _assert_vector_dim(cur: Any, schema: str, table: str, expected: int) -> None:
    """Fail loudly if an existing ``embedding`` column's dimension drifted.

    ``vector(dim)`` is fixed at create time and ``CREATE TABLE IF NOT EXISTS`` is
    a no-op on re-run, so a later change to ``EMBEDDING_DIMENSION`` would
    otherwise surface only as an opaque ``expected N dimensions, not M`` on the
    next insert. pgvector stores the declared dimension directly in
    ``atttypmod``, so we read it back and raise a clear rebuild-required error on
    mismatch (mirroring the SQLite path's ``SchemaRebuildRequired``).
    """
    row = cur.execute(
        """
        SELECT a.atttypmod AS dim
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE n.nspname = %s AND c.relname = %s AND a.attname = 'embedding'
          AND NOT a.attisdropped
        """,
        (schema, table),
    ).fetchone()
    if row is None:
        return
    actual = row["dim"] if isinstance(row, dict) else row[0]
    # atttypmod is -1 for an unspecified typmod; only compare when declared.
    if actual is not None and actual > 0 and actual != expected:
        raise SchemaRebuildRequired(
            f"Postgres table {schema}.{table}.embedding was created with "
            f"vector({actual}) but the configured embedding dimension is now "
            f"{expected}. pgvector columns can't be resized in place -- back up "
            f"and rebuild the index (drop {schema}.{table} or the schema and "
            f"re-ingest under the new dimension)."
        )


def migrate(conn: Any, schema: str = "public") -> None:
    """Bring ``conn``'s database to the v0.14 Postgres schema (idempotent).

    Args:
        conn: An open psycopg connection.
        schema: Schema to create the librarian tables in.
    """
    from psycopg import sql

    text_dim = get_effective_embedding_dimension()
    code_dim = CODE_EMBEDDING_DIMENSION
    vision_dim = VISION_EMBEDDING_DIMENSION
    fts_lang = _fts_language()

    logger.info("Running v0.14 Postgres schema migration (schema=%s)", schema)

    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
        # Pin pgvector to ``public`` (always on the search_path) rather than the
        # current schema. Installing it into a per-deployment / per-test schema
        # would make the ``vector`` type vanish when that schema is dropped, even
        # though ``CREATE EXTENSION IF NOT EXISTS`` still considers it present.
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
        cur.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id BIGSERIAL PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT now(),
                updated_at TIMESTAMPTZ DEFAULT now(),
                file_mtime DOUBLE PRECISION,
                asset_type TEXT,
                document_id TEXT,
                document_source_uri TEXT,
                source_created_at TEXT
            )
            """
        )
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_document_id ON documents(document_id)"
        )
        # Supports list_documents' ``ORDER BY updated_at DESC, id DESC`` so a
        # paginated read is an index-ordered scan of ``limit + offset`` rows
        # rather than a full scan + sort of the (wide, content-bearing) table.
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_updated_at_id "
            "ON documents(updated_at DESC, id DESC)"
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id BIGSERIAL PRIMARY KEY,
                document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                heading_path TEXT,
                chunk_index INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                asset_type TEXT,
                modality TEXT,
                chunk_id TEXT,
                document_size INTEGER,
                source_created_at TEXT,
                deleted_at TEXT,
                deletion_reason TEXT,
                document_source_uri TEXT,
                chunk_source_uri TEXT,
                content_tsv tsvector GENERATED ALWAYS AS
                    (to_tsvector('{fts_lang}', content)) STORED
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_deleted_at ON chunks(deleted_at)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv ON chunks USING GIN(content_tsv)"
        )

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                embedding vector({text_dim}),
                model_version TEXT
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS vec_chunks_code (
                chunk_id BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                embedding vector({code_dim})
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS vec_chunks_vision (
                chunk_id BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                embedding vector({vision_dim})
            )
            """
        )

        # If a table already existed with a different declared dimension, fail
        # loudly now rather than on the next opaque %s::vector insert.
        _assert_vector_dim(cur, schema, "chunk_embeddings", text_dim)
        _assert_vector_dim(cur, schema, "vec_chunks_code", code_dim)
        _assert_vector_dim(cur, schema, "vec_chunks_vision", vision_dim)

        # ANN indexes for the cosine ``<=>`` searches. Without these, every
        # ORDER BY embedding <=> ... is an exact, O(rows) sequential scan -- the
        # exact cliff this backend exists to avoid at scale. HNSW trades a slower
        # build + approximate recall for sublinear query time; ``vector_cosine_ops``
        # matches the ``<=>`` operator used by PgVectorStore.
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_hnsw "
            "ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_vec_chunks_code_hnsw "
            "ON vec_chunks_code USING hnsw (embedding vector_cosine_ops)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_vec_chunks_vision_hnsw "
            "ON vec_chunks_vision USING hnsw (embedding vector_cosine_ops)"
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_state (
                source_key TEXT PRIMARY KEY,
                cursor JSONB,
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
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS source_file_state (
                source_key TEXT NOT NULL,
                path TEXT NOT NULL,
                mtime DOUBLE PRECISION NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT now(),
                PRIMARY KEY (source_key, path)
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_source_file_state_source "
            "ON source_file_state(source_key)"
        )

    conn.commit()
    logger.info("v0.14 Postgres schema migration complete")
