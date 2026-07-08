"""
Vector similarity search on Postgres via pgvector.

Mirrors :class:`librarian.storage.vector_store.VectorStore` (the sqlite-vec
implementation) and returns the same :class:`VectorSearchResult` dataclass so
the retrieval layer is substrate-agnostic. Cosine distance comes from pgvector's
``<=>`` operator, matching sqlite-vec's ``distance_metric=cosine`` semantics
(``similarity = 1 - distance``).
"""

import logging
from typing import ClassVar

from librarian.config import (
    CODE_EMBEDDING_DIMENSION,
    ENABLE_CODE_EMBEDDINGS,
    ENABLE_VISION_EMBEDDINGS,
    VISION_EMBEDDING_DIMENSION,
)
from librarian.storage._common import deleted_filter
from librarian.storage.database import get_effective_embedding_dimension
from librarian.storage.postgres.database import PostgresDatabase, parse_vector, vector_literal
from librarian.storage.vector_store import VectorSearchResult
from librarian.types import EmbeddingModality

logger = logging.getLogger(__name__)

__all__ = ["PgVectorStore"]


class PgVectorStore:
    """pgvector-backed vector store implementing the read-side protocol."""

    _MODALITY_TABLES: ClassVar[dict[EmbeddingModality, str]] = {
        EmbeddingModality.TEXT: "chunk_embeddings",
        EmbeddingModality.CODE: "vec_chunks_code",
        EmbeddingModality.VISION: "vec_chunks_vision",
    }

    def __init__(self, database: PostgresDatabase) -> None:
        self.db = database

    def _table_for_modality(self, modality: EmbeddingModality) -> str:
        return self._MODALITY_TABLES.get(modality, "chunk_embeddings")

    def _dimension_for_modality(self, modality: EmbeddingModality) -> int:
        if modality == EmbeddingModality.CODE:
            return CODE_EMBEDDING_DIMENSION
        if modality == EmbeddingModality.VISION:
            return VISION_EMBEDDING_DIMENSION
        return get_effective_embedding_dimension()

    def _is_modality_enabled(self, modality: EmbeddingModality) -> bool:
        if modality == EmbeddingModality.TEXT:
            return True
        if modality == EmbeddingModality.CODE:
            return ENABLE_CODE_EMBEDDINGS
        if modality == EmbeddingModality.VISION:
            return ENABLE_VISION_EMBEDDINGS
        return False

    def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        min_similarity: float = 0.0,
        include_deleted: bool = False,
    ) -> list[VectorSearchResult]:
        expected_dim = get_effective_embedding_dimension()
        if len(query_embedding) != expected_dim:
            msg = (
                f"Query embedding dimension {len(query_embedding)} "
                f"does not match expected {expected_dim}"
            )
            raise ValueError(msg)
        return self._search_table(
            "chunk_embeddings", query_embedding, limit, min_similarity, include_deleted
        )

    def search_by_modality(
        self,
        query_embedding: list[float],
        modality: EmbeddingModality,
        limit: int = 10,
        min_similarity: float = 0.0,
        include_deleted: bool = False,
    ) -> list[VectorSearchResult]:
        if not self._is_modality_enabled(modality):
            logger.warning("Modality %s is not enabled", modality.value)
            return []
        expected_dim = self._dimension_for_modality(modality)
        if len(query_embedding) != expected_dim:
            msg = (
                f"Query embedding dimension {len(query_embedding)} "
                f"does not match expected {expected_dim} for {modality.value}"
            )
            raise ValueError(msg)
        table = self._table_for_modality(modality)
        return self._search_table(table, query_embedding, limit, min_similarity, include_deleted)

    def _search_table(
        self,
        table: str,
        query_embedding: list[float],
        limit: int,
        min_similarity: float,
        include_deleted: bool = False,
    ) -> list[VectorSearchResult]:
        literal = vector_literal(query_embedding)
        with self.db._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    ve.chunk_id AS chunk_id,
                    (ve.embedding <=> %s::vector) AS distance,
                    c.content AS content,
                    c.document_id AS document_id,
                    c.heading_path AS heading_path,
                    d.path AS document_path,
                    d.asset_type AS asset_type
                FROM {table} ve
                JOIN chunks c ON ve.chunk_id = c.id
                JOIN documents d ON c.document_id = d.id
                WHERE TRUE {deleted_filter(include_deleted)}
                ORDER BY ve.embedding <=> %s::vector
                LIMIT %s
                """,  # noqa: S608 - table/deleted_filter are fixed internal literals
                (literal, literal, limit * 2),
            ).fetchall()

        results: list[VectorSearchResult] = []
        for row in rows:
            distance = float(row["distance"])
            similarity = 1.0 - distance
            if similarity >= min_similarity:
                results.append(
                    VectorSearchResult(
                        chunk_id=row["chunk_id"],
                        distance=distance,
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
        results = self.search(query_embedding, limit=limit + len(exclude_chunk_ids))
        return [r for r in results if r.chunk_id not in exclude_chunk_ids][:limit]

    def get_embedding(
        self, chunk_id: int, modality: EmbeddingModality | None = None
    ) -> list[float] | None:
        tables = (
            [self._table_for_modality(modality)]
            if modality is not None
            else ["chunk_embeddings", "vec_chunks_code", "vec_chunks_vision"]
        )
        with self.db._connection() as conn:
            for table in tables:
                row = conn.execute(
                    f"SELECT embedding::text AS embedding FROM {table} WHERE chunk_id = %s",  # noqa: S608
                    (chunk_id,),
                ).fetchone()
                if row and row["embedding"]:
                    return parse_vector(row["embedding"])
        return None
