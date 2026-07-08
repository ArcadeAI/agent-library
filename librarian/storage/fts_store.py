"""
Full-text search operations using SQLite FTS5.

This module provides full-text search functionality
using SQLite's FTS5 extension.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from librarian.storage._common import deleted_filter

if TYPE_CHECKING:
    from librarian.storage.database import Database

logger = logging.getLogger(__name__)


@dataclass
class FTSSearchResult:
    """Result from a full-text search."""

    chunk_id: int
    rank: float
    content: str
    document_id: int
    document_path: str
    heading_path: str | None
    snippet: str | None = None
    asset_type: str | None = None


class FTSStore:
    """
    Full-text search operations using SQLite FTS5.

    Provides methods for keyword-based full-text search with BM25 ranking.
    """

    def __init__(self, database: "Database") -> None:
        """
        Initialize the FTS store.

        Args:
            database: The database instance to use.
        """
        self.db = database

    def search(
        self,
        query: str,
        limit: int = 10,
        snippet_length: int = 64,
        include_deleted: bool = False,
    ) -> list[FTSSearchResult]:
        """
        Search for chunks matching the query using full-text search.

        Args:
            query: The search query (supports FTS5 query syntax).
            limit: Maximum number of results to return.
            snippet_length: Length of snippet to return.
            include_deleted: When True, soft-deleted chunks are included; by
                default they are filtered out.

        Returns:
            List of search results ordered by relevance (best match first).
        """
        if not query.strip():
            return []

        # Escape special FTS5 characters in the query for simple searches
        # Users can still use FTS5 syntax by quoting terms
        safe_query = self._prepare_query(query)

        with self.db._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    c.id as chunk_id,
                    bm25(chunks_fts) as rank,
                    c.content,
                    c.document_id,
                    c.heading_path,
                    d.path as document_path,
                    d.asset_type,
                    snippet(chunks_fts, 0, '<mark>', '</mark>', '...', ?) as snippet
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.rowid = c.id
                JOIN documents d ON c.document_id = d.id
                WHERE chunks_fts MATCH ?
                    {deleted_filter(include_deleted)}
                ORDER BY rank
                LIMIT ?
                """,  # noqa: S608 - deleted_filter returns a fixed internal literal
                (snippet_length, safe_query, limit),
            ).fetchall()

            return [
                FTSSearchResult(
                    chunk_id=row["chunk_id"],
                    rank=row["rank"],
                    content=row["content"],
                    document_id=row["document_id"],
                    document_path=row["document_path"],
                    heading_path=row["heading_path"],
                    snippet=row["snippet"],
                    asset_type=row["asset_type"],
                )
                for row in rows
            ]

    def search_in_document(
        self,
        query: str,
        document_id: int,
        limit: int = 10,
    ) -> list[FTSSearchResult]:
        """
        Search for chunks within a specific document.

        Args:
            query: The search query.
            document_id: The document ID to search within.
            limit: Maximum number of results to return.

        Returns:
            List of search results from the specified document.
        """
        if not query.strip():
            return []

        safe_query = self._prepare_query(query)

        with self.db._connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.id as chunk_id,
                    bm25(chunks_fts) as rank,
                    c.content,
                    c.document_id,
                    c.heading_path,
                    d.path as document_path,
                    d.asset_type
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.rowid = c.id
                JOIN documents d ON c.document_id = d.id
                WHERE chunks_fts MATCH ?
                    AND c.document_id = ?
                    AND c.deleted_at IS NULL
                ORDER BY rank
                LIMIT ?
                """,
                (safe_query, document_id, limit),
            ).fetchall()

            return [
                FTSSearchResult(
                    chunk_id=row["chunk_id"],
                    rank=row["rank"],
                    content=row["content"],
                    document_id=row["document_id"],
                    document_path=row["document_path"],
                    heading_path=row["heading_path"],
                    asset_type=row["asset_type"],
                )
                for row in rows
            ]

    def _prepare_query(self, query: str) -> str:
        """
        Prepare a query for FTS5 search.

        Handles simple queries by adding implicit OR between terms,
        and preserves quoted phrases for exact matching.

        Args:
            query: The raw search query.

        Returns:
            FTS5-compatible query string.
        """
        # If query has quotes, user is doing advanced FTS5 - leave as-is
        if '"' in query:
            return query

        # Split into terms and process each
        # Quote terms with special characters (hyphen, etc.) to prevent
        # FTS5 from interpreting them as column references
        terms = query.split()
        safe_terms = []
        for term in terms:
            upper = term.upper()
            # Skip FTS5 operator words (standalone)
            if upper in ("AND", "OR", "NOT", "NEAR"):
                safe_terms.append(term)
            # Quote terms with special chars
            elif "-" in term or ":" in term or "." in term:
                safe_terms.append(f'"{term}"')
            else:
                safe_terms.append(f"{term}*")  # Prefix search

        if len(safe_terms) == 1:
            return safe_terms[0]

        # Join terms with OR for broader matching
        return " OR ".join(safe_terms)

    def highlight_matches(
        self,
        query: str,
        chunk_id: int,
        start_marker: str = "<mark>",
        end_marker: str = "</mark>",
    ) -> str | None:
        """
        Get highlighted content for a specific chunk.

        Args:
            query: The search query.
            chunk_id: The chunk ID to highlight.
            start_marker: HTML/marker to start highlighting.
            end_marker: HTML/marker to end highlighting.

        Returns:
            Highlighted content string, or None if not found.
        """
        safe_query = self._prepare_query(query)

        with self.db._connection() as conn:
            row = conn.execute(
                """
                SELECT highlight(chunks_fts, 0, ?, ?) as highlighted
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                    AND rowid = ?
                """,
                (start_marker, end_marker, safe_query, chunk_id),
            ).fetchone()

            return row["highlighted"] if row else None
