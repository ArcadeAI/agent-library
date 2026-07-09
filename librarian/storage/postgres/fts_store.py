"""
Full-text (keyword) search on Postgres.

Mirrors :class:`librarian.storage.fts_store.FTSStore` (SQLite FTS5/BM25) and
returns the same :class:`FTSSearchResult` dataclass. Postgres ranks with
``ts_rank_cd`` (higher = better); the retrieval layer normalizes on the
absolute value, so the orientation difference from BM25 (where more-negative =
better) is invisible downstream. ``websearch_to_tsquery`` parses the raw user
query, giving forgiving plain-language matching without FTS5-specific escaping.
"""

import logging

from librarian.config import POSTGRES_FTS_LANGUAGE
from librarian.storage._common import deleted_filter
from librarian.storage.fts_store import FTSSearchResult
from librarian.storage.postgres.database import PostgresDatabase

logger = logging.getLogger(__name__)

__all__ = ["PgFTSStore"]


class PgFTSStore:
    """Postgres ``tsvector`` full-text search implementing the read-side protocol."""

    def __init__(self, database: PostgresDatabase) -> None:
        self.db = database

    def search(
        self,
        query: str,
        limit: int = 10,
        snippet_length: int = 64,
        include_deleted: bool = False,
    ) -> list[FTSSearchResult]:
        if not query.strip():
            return []

        max_words = max(5, int(snippet_length))
        headline_opts = (
            f"StartSel=<mark>, StopSel=</mark>, MaxFragments=1, MaxWords={max_words}, MinWords=1"
        )
        # The regconfig is passed as a bound parameter (cast to ::regconfig), so it
        # can't be a SQL-injection vector even though it's configurable. It must
        # match the language baked into the generated ``content_tsv`` column at
        # migration time (see migrate.py / POSTGRES_FTS_LANGUAGE).
        lang = POSTGRES_FTS_LANGUAGE
        with self.db._connection() as conn:
            rows = conn.execute(
                f"""
                WITH q AS (SELECT websearch_to_tsquery(%s::regconfig, %s) AS query)
                SELECT
                    c.id AS chunk_id,
                    ts_rank_cd(c.content_tsv, q.query) AS rank,
                    c.content AS content,
                    c.document_id AS document_id,
                    c.heading_path AS heading_path,
                    d.path AS document_path,
                    d.asset_type AS asset_type,
                    ts_headline(%s::regconfig, c.content, q.query, '{headline_opts}') AS snippet
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                CROSS JOIN q
                WHERE c.content_tsv @@ q.query
                  {deleted_filter(include_deleted)}
                ORDER BY rank DESC
                LIMIT %s
                """,  # noqa: S608 - headline_opts/deleted_filter are fixed internal literals
                (lang, query, lang, limit),
            ).fetchall()

        return [
            FTSSearchResult(
                chunk_id=row["chunk_id"],
                rank=float(row["rank"]),
                content=row["content"],
                document_id=row["document_id"],
                document_path=row["document_path"],
                heading_path=row["heading_path"],
                snippet=row["snippet"],
                asset_type=row["asset_type"],
            )
            for row in rows
        ]
