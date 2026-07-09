"""
Database schema migrations for multi-modal support.

Handles incremental schema changes to support text, code, image, and PDF assets
with separate embedding tables per modality.
"""

import logging
import sqlite3

from librarian.config import (
    CODE_EMBEDDING_DIMENSION,
    VISION_EMBEDDING_DIMENSION,
)

logger = logging.getLogger(__name__)


def get_schema_version(conn: sqlite3.Connection) -> int:
    """
    Get the current schema version.

    Args:
        conn: SQLite database connection.

    Returns:
        Current schema version (0 if no version table exists).
    """
    try:
        cursor = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
        row = cursor.fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0


def set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """
    Set the schema version.

    Args:
        conn: SQLite database connection.
        version: Version number to set.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))


def migrate_to_v1_multimodal(conn: sqlite3.Connection) -> None:
    """
    Migration v1: Add multi-modal columns to existing schema.

    Changes:
    - Add asset_type column to documents table
    - Add modality_data column to documents table (JSON)
    - Add asset_type column to chunks table
    - Add modality column to chunks table
    - Add indexes for asset_type and modality
    - Create separate vector tables for code and vision embeddings
    """
    logger.info("Running migration v1: multi-modal support")

    # Check existing columns
    cursor = conn.execute("PRAGMA table_info(documents)")
    doc_columns = {row[1] for row in cursor.fetchall()}

    cursor = conn.execute("PRAGMA table_info(chunks)")
    chunk_columns = {row[1] for row in cursor.fetchall()}

    # Add columns to documents table
    if "asset_type" not in doc_columns:
        logger.info("Adding asset_type column to documents table")
        conn.execute("ALTER TABLE documents ADD COLUMN asset_type TEXT DEFAULT 'text'")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_asset_type ON documents(asset_type)")

    if "modality_data" not in doc_columns:
        logger.info("Adding modality_data column to documents table")
        conn.execute("ALTER TABLE documents ADD COLUMN modality_data JSON")

    # Add columns to chunks table
    if "asset_type" not in chunk_columns:
        logger.info("Adding asset_type column to chunks table")
        conn.execute("ALTER TABLE chunks ADD COLUMN asset_type TEXT DEFAULT 'text'")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_asset_type ON chunks(asset_type)")

    if "modality" not in chunk_columns:
        logger.info("Adding modality column to chunks table")
        conn.execute("ALTER TABLE chunks ADD COLUMN modality TEXT DEFAULT 'text'")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_modality ON chunks(modality)")

    # Create modality-specific vector tables
    # Text embeddings use the existing chunk_embeddings table
    # Code embeddings get their own table
    logger.info("Creating vec_chunks_code table")
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks_code USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding float[{CODE_EMBEDDING_DIMENSION}] distance_metric=cosine
        )
    """)

    # Vision embeddings get their own table
    logger.info("Creating vec_chunks_vision table")
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks_vision USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding float[{VISION_EMBEDDING_DIMENSION}] distance_metric=cosine
        )
    """)

    logger.info("Migration v1 completed successfully")


def migrate_to_v2_chunk_modality_data(conn: sqlite3.Connection) -> None:
    """
    Migration v2: Add the per-chunk ``modality_data`` JSON column.

    The vision pipeline (#53) records a chunk's ``processing_status`` (and VLM
    caption / transcription) here so failed images can be found and retried by
    ``libr reprocess``. This runs from ``Database._init_schema`` on *every* open
    -- read path included -- so a populated v0.14.0/0.14.1 index gains the column
    before the first ``search_library`` selects it (the v0.14 ``migrate.py`` add
    only runs on the write path).
    """
    cursor = conn.execute("PRAGMA table_info(chunks)")
    chunk_columns = {row[1] for row in cursor.fetchall()}
    if "modality_data" not in chunk_columns:
        logger.info("Adding modality_data column to chunks table")
        conn.execute("ALTER TABLE chunks ADD COLUMN modality_data JSON")


def run_migrations(conn: sqlite3.Connection) -> None:
    """
    Run all pending database migrations.

    Args:
        conn: SQLite database connection.
    """
    current_version = get_schema_version(conn)
    logger.info(f"Current schema version: {current_version}")

    migrations = [
        (1, migrate_to_v1_multimodal),
        (2, migrate_to_v2_chunk_modality_data),
    ]

    for version, migration_func in migrations:
        if current_version < version:
            logger.info(f"Applying migration to version {version}")
            migration_func(conn)
            set_schema_version(conn, version)
            conn.commit()
            logger.info(f"Migration to version {version} completed")
        else:
            logger.debug(f"Skipping migration to version {version} (already applied)")

    final_version = get_schema_version(conn)
    logger.info(f"Schema is now at version {final_version}")
