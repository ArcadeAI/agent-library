"""
Vector store operations using sqlite-vec.

This module provides vector similarity search functionality
using the sqlite-vec extension. Supports multi-modal embeddings
(text, code, vision) stored in separate vector tables.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from librarian.config import (
    CODE_EMBEDDING_DIMENSION,
    ENABLE_CODE_EMBEDDINGS,
    ENABLE_VISION_EMBEDDINGS,
    VISION_EMBEDDING_DIMENSION,
)
from librarian.storage._common import deleted_filter
from librarian.storage.database import get_effective_embedding_dimension, serialize_embedding
from librarian.types import EmbeddingModality

if TYPE_CHECKING:
    from librarian.storage.database import Database

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from a vector similarity search."""

    chunk_id: int
    distance: float
    content: str
    document_id: int
    document_path: str
    heading_path: str | None
    asset_type: str


class VectorStore:
    """
    Vector store operations using sqlite-vec.

    Uses cosine distance for normalized embeddings.
    Distance returned is 1 - cosine_similarity, so lower = more similar.

    Supports multi-modal embeddings stored in separate tables:
    - chunk_embeddings: TEXT modality
    - vec_chunks_code: CODE modality (CodeBERT)
    - vec_chunks_vision: VISION modality (CLIP)
    """

    # Mapping of modalities to their vector table names
    _MODALITY_TABLES: ClassVar[dict[EmbeddingModality, str]] = {
        EmbeddingModality.TEXT: "chunk_embeddings",
        EmbeddingModality.CODE: "vec_chunks_code",
        EmbeddingModality.VISION: "vec_chunks_vision",
    }

    def __init__(self, database: "Database") -> None:
        """
        Initialize the vector store.

        Args:
            database: The database instance to use.
        """
        self.db = database

    def _get_table_for_modality(self, modality: EmbeddingModality) -> str:
        """
        Get the vector table name for a modality.

        Args:
            modality: The embedding modality.

        Returns:
            Table name for the modality.
        """
        return self._MODALITY_TABLES.get(modality, "chunk_embeddings")

    def _get_dimension_for_modality(self, modality: EmbeddingModality) -> int:
        """
        Get the expected embedding dimension for a modality.

        Args:
            modality: The embedding modality.

        Returns:
            Expected embedding dimension.
        """
        if modality == EmbeddingModality.CODE:
            return CODE_EMBEDDING_DIMENSION
        elif modality == EmbeddingModality.VISION:
            return VISION_EMBEDDING_DIMENSION
        return get_effective_embedding_dimension()

    def _is_modality_enabled(self, modality: EmbeddingModality) -> bool:
        """
        Check if a modality is enabled.

        Args:
            modality: The embedding modality.

        Returns:
            True if the modality is enabled.
        """
        if modality == EmbeddingModality.TEXT:
            return True
        elif modality == EmbeddingModality.CODE:
            return ENABLE_CODE_EMBEDDINGS
        elif modality == EmbeddingModality.VISION:
            return ENABLE_VISION_EMBEDDINGS
        return False

    def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.0,
        include_deleted: bool = False,
    ) -> list[VectorSearchResult]:
        """
        Search for similar chunks using vector similarity.

        Args:
            query_embedding: The query embedding vector.
            limit: Maximum number of results to return.
            min_similarity: Minimum similarity score (0-1) to include.
            include_deleted: When True, soft-deleted (tombstoned) chunks are
                included; by default they are filtered out.

        Returns:
            List of search results ordered by similarity (most similar first).
        """
        expected_dim = get_effective_embedding_dimension()
        if len(query_embedding) != expected_dim:
            msg = (
                f"Query embedding dimension {len(query_embedding)} "
                f"does not match expected {expected_dim}"
            )
            raise ValueError(msg)

        query_blob = serialize_embedding(query_embedding)

        with self.db._connection() as conn:
            # sqlite-vec uses distance (lower is more similar)
            # We convert to similarity for easier understanding
            rows = conn.execute(
                f"""
                SELECT
                    ce.chunk_id,
                    ce.distance,
                    c.content,
                    c.document_id,
                    c.heading_path,
                    d.path as document_path,
                    d.asset_type
                FROM chunk_embeddings ce
                JOIN chunks c ON ce.chunk_id = c.id
                JOIN documents d ON c.document_id = d.id
                WHERE ce.embedding MATCH ?
                    AND k = ?
                    {deleted_filter(include_deleted)}
                ORDER BY ce.distance ASC
                """,  # noqa: S608 - deleted_filter returns a fixed internal literal
                (query_blob, limit * 2),  # Get extra for filtering
            ).fetchall()

            results = []
            for row in rows:
                # Convert distance to similarity (1 - distance for cosine)
                similarity = 1.0 - row["distance"]
                if similarity >= min_similarity:
                    results.append(
                        VectorSearchResult(
                            chunk_id=row["chunk_id"],
                            distance=row["distance"],
                            content=row["content"],
                            document_id=row["document_id"],
                            document_path=row["document_path"],
                            heading_path=row["heading_path"],
                            asset_type=row["asset_type"],
                        )
                    )
                    if len(results) >= limit:
                        break

            return results

    def search_with_exclusions(
        self,
        query_embedding: list[float],
        exclude_chunk_ids: set[int],
        limit: int = 10,
    ) -> list[VectorSearchResult]:
        """
        Search for similar chunks, excluding specified chunk IDs.

        This is useful for MMR (Maximal Marginal Relevance) where we need
        to find similar chunks while excluding already selected ones.

        Args:
            query_embedding: The query embedding vector.
            exclude_chunk_ids: Set of chunk IDs to exclude from results.
            limit: Maximum number of results to return.

        Returns:
            List of search results.
        """
        # Get more results than needed to account for exclusions
        results = self.search(query_embedding, limit=limit + len(exclude_chunk_ids))
        return [r for r in results if r.chunk_id not in exclude_chunk_ids][:limit]

    def search_by_modality(
        self,
        query_embedding: list[float],
        modality: EmbeddingModality,
        limit: int = 10,
        min_similarity: float = 0.0,
        include_deleted: bool = False,
    ) -> list[VectorSearchResult]:
        """
        Search a specific modality's vector table.

        Args:
            query_embedding: The query embedding vector.
            modality: The embedding modality to search (TEXT, CODE, VISION).
            limit: Maximum number of results to return.
            min_similarity: Minimum similarity score (0-1) to include.
            include_deleted: When True, soft-deleted chunks are included.

        Returns:
            List of search results ordered by similarity.

        Raises:
            ValueError: If modality is not enabled or embedding dimension doesn't match.
        """
        if not self._is_modality_enabled(modality):
            logger.warning(f"Modality {modality.value} is not enabled")
            return []

        expected_dim = self._get_dimension_for_modality(modality)
        if len(query_embedding) != expected_dim:
            msg = (
                f"Query embedding dimension {len(query_embedding)} "
                f"does not match expected {expected_dim} for {modality.value}"
            )
            raise ValueError(msg)

        table = self._get_table_for_modality(modality)
        query_blob = serialize_embedding(query_embedding)

        with self.db._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    ve.chunk_id,
                    ve.distance,
                    c.content,
                    c.document_id,
                    c.heading_path,
                    d.path as document_path,
                    d.asset_type
                FROM {table} ve
                JOIN chunks c ON ve.chunk_id = c.id
                JOIN documents d ON c.document_id = d.id
                WHERE ve.embedding MATCH ?
                    AND k = ?
                    {deleted_filter(include_deleted)}
                ORDER BY ve.distance ASC
                """,  # noqa: S608
                (query_blob, limit * 2),
            ).fetchall()

            results = []
            for row in rows:
                similarity = 1.0 - row["distance"]
                if similarity >= min_similarity:
                    results.append(
                        VectorSearchResult(
                            chunk_id=row["chunk_id"],
                            distance=row["distance"],
                            content=row["content"],
                            document_id=row["document_id"],
                            document_path=row["document_path"],
                            heading_path=row["heading_path"],
                            asset_type=row["asset_type"],
                        )
                    )
                    if len(results) >= limit:
                        break

            return results

    def search_all_modalities(
        self,
        embeddings_by_modality: dict[EmbeddingModality, list[float]],
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> list[VectorSearchResult]:
        """
        Search across all enabled modality tables and merge results.

        Args:
            embeddings_by_modality: Dict mapping modality to its query embedding.
            limit: Maximum total number of results to return.
            min_similarity: Minimum similarity score (0-1) to include.

        Returns:
            Merged and re-ranked list of search results.
        """
        all_results: list[VectorSearchResult] = []

        for modality, embedding in embeddings_by_modality.items():
            if self._is_modality_enabled(modality):
                try:
                    results = self.search_by_modality(
                        embedding,
                        modality,
                        limit=limit,
                        min_similarity=min_similarity,
                    )
                    all_results.extend(results)
                except ValueError as e:
                    logger.warning(f"Skipping {modality.value} search: {e}")

        # Sort by distance (lower is better) and deduplicate by chunk_id
        all_results.sort(key=lambda r: r.distance)
        seen_chunks: set[int] = set()
        unique_results: list[VectorSearchResult] = []
        for r in all_results:
            if r.chunk_id not in seen_chunks:
                seen_chunks.add(r.chunk_id)
                unique_results.append(r)
                if len(unique_results) >= limit:
                    break

        return unique_results

    def get_embedding(
        self, chunk_id: int, modality: EmbeddingModality | None = None
    ) -> list[float] | None:
        """
        Get the embedding for a specific chunk.

        Args:
            chunk_id: The chunk ID.
            modality: Optional modality to search. If None, searches all tables.

        Returns:
            The embedding vector if found, None otherwise.
        """
        from librarian.storage.database import deserialize_embedding

        # If modality specified, search only that table
        if modality is not None:
            table = self._get_table_for_modality(modality)
            with self.db._connection() as conn:
                row = conn.execute(
                    f"SELECT embedding FROM {table} WHERE chunk_id = ?",  # noqa: S608
                    (chunk_id,),
                ).fetchone()
                if row and row["embedding"]:
                    return deserialize_embedding(row["embedding"])
            return None

        # Otherwise search all tables
        tables = ["chunk_embeddings", "vec_chunks_code", "vec_chunks_vision"]
        with self.db._connection() as conn:
            for table in tables:
                row = conn.execute(
                    f"SELECT embedding FROM {table} WHERE chunk_id = ?",  # noqa: S608
                    (chunk_id,),
                ).fetchone()
                if row and row["embedding"]:
                    return deserialize_embedding(row["embedding"])
        return None

    def update_embedding(self, chunk_id: int, embedding: list[float]) -> None:
        """
        Update the embedding for a chunk.

        Args:
            chunk_id: The chunk ID.
            embedding: The new embedding vector.
        """
        with self.db._lock, self.db._connection() as conn:
            # Delete existing embedding if any
            conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,))
            # Insert new embedding
            conn.execute(
                "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, serialize_embedding(embedding)),
            )

    def batch_update_embeddings(self, chunk_embeddings: list[tuple[int, list[float]]]) -> None:
        """
        Update embeddings for multiple chunks in a batch.

        Args:
            chunk_embeddings: List of (chunk_id, embedding) tuples.
        """
        with self.db._lock, self.db._connection() as conn:
            for chunk_id, embedding in chunk_embeddings:
                conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,))
                conn.execute(
                    "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, serialize_embedding(embedding)),
                )
