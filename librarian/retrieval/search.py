"""
Hybrid search and MMR retrieval.

This module provides hybrid search combining vector similarity
and full-text search, with Max Marginal Relevance (MMR) for
diverse result selection. Supports multi-modal search with
specialized embeddings for code and vision content.

Multi-modal search runs parallel queries across all enabled modalities
(TEXT, CODE, VISION) and merges results with fair normalization.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

from librarian.config import (
    ENABLE_CODE_EMBEDDINGS,
    ENABLE_CROSS_MODAL_SEARCH,
    ENABLE_VISION_EMBEDDINGS,
    HYBRID_ALPHA,
    MMR_LAMBDA,
    MODALITY_WEIGHT_CODE,
    MODALITY_WEIGHT_FTS,
    MODALITY_WEIGHT_TEXT,
    MODALITY_WEIGHT_VISION,
    SEARCH_LIMIT,
)
from librarian.processing.embed import get_embedder_for_modality
from librarian.storage.factory import get_read_storage
from librarian.types import AssetType, EmbeddingModality, SearchResult

if TYPE_CHECKING:
    from librarian.processing.embed import Embedder

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    Hybrid searcher combining vector and full-text search.

    Supports:
    - Pure vector search
    - Pure full-text search
    - Hybrid search with configurable weighting
    - Max Marginal Relevance (MMR) for diverse results
    """

    def __init__(
        self,
        embedder: "Embedder",
        hybrid_alpha: float | None = None,
        mmr_lambda: float | None = None,
    ) -> None:
        """
        Initialize the hybrid searcher.

        Args:
            embedder: Embedder instance for generating query embeddings.
            hybrid_alpha: Weight for vector vs FTS (0=FTS only, 1=vector only).
            mmr_lambda: MMR diversity parameter (0=max diversity, 1=max relevance).
        """
        self.embedder = embedder
        self.hybrid_alpha = hybrid_alpha if hybrid_alpha is not None else HYBRID_ALPHA
        self.mmr_lambda = mmr_lambda if mmr_lambda is not None else MMR_LAMBDA

        # Resolve the read stores from the active storage backend (sqlite or
        # postgres) so search runs against whichever substrate ingest wrote to.
        storage = get_read_storage()
        self.db = storage.metadata
        self.vector_store = storage.vectors
        self.fts_store = storage.fts

    def search(
        self,
        query: str,
        limit: int | None = None,
        use_mmr: bool = True,
        filter_document_ids: list[int] | None = None,
        asset_types: list[AssetType] | None = None,
        include_deleted: bool = False,
    ) -> list[SearchResult]:
        """
        Perform hybrid search across all enabled modalities.

        When cross-modal search is enabled (default), searches TEXT, CODE,
        and VISION modalities in parallel with fair normalization so all
        content types have equal opportunity to appear in results.

        Args:
            query: The search query.
            limit: Maximum number of results to return.
            use_mmr: Whether to use MMR for diverse results.
            filter_document_ids: Optional list of document IDs to search within.
            asset_types: Optional list of asset types to filter results by.
            include_deleted: When True, soft-deleted chunks are included; by
                default the retrieval layer filters them out.

        Returns:
            List of search results ordered by relevance.
        """
        limit = limit or SEARCH_LIMIT

        # Use multi-modal search if cross-modal search is enabled
        if ENABLE_CROSS_MODAL_SEARCH:
            return self.multi_modal_search(
                query,
                limit=limit,
                use_mmr=use_mmr,
                filter_document_ids=filter_document_ids,
                asset_types=asset_types,
                include_deleted=include_deleted,
            )

        # Fall back to original text-only hybrid search
        vector_results = self._vector_search(query, limit * 2, include_deleted=include_deleted)
        fts_results = self._fts_search(query, limit * 2, include_deleted=include_deleted)

        # Combine and score results
        combined = self._combine_results(vector_results, fts_results)

        # Filter by document IDs if specified
        if filter_document_ids:
            combined = [r for r in combined if r.document_id in filter_document_ids]

        # Filter by asset types if specified
        if asset_types:
            combined = [r for r in combined if r.asset_type in asset_types]

        # Apply MMR if requested
        if use_mmr and len(combined) > limit:
            combined = self._apply_mmr(query, combined, limit)
        else:
            combined = combined[:limit]

        return combined

    def vector_search(
        self,
        query: str,
        limit: int | None = None,
        include_deleted: bool = False,
    ) -> list[SearchResult]:
        """
        Perform pure vector similarity search.

        Args:
            query: The search query.
            limit: Maximum number of results.
            include_deleted: When True, soft-deleted chunks are included.

        Returns:
            List of search results.
        """
        limit = limit or SEARCH_LIMIT
        return self._vector_search(query, limit, include_deleted=include_deleted)

    def keyword_search(
        self,
        query: str,
        limit: int | None = None,
        include_deleted: bool = False,
    ) -> list[SearchResult]:
        """
        Perform pure keyword/FTS search.

        Args:
            query: The search query.
            limit: Maximum number of results.
            include_deleted: When True, soft-deleted chunks are included.

        Returns:
            List of search results.
        """
        limit = limit or SEARCH_LIMIT
        return self._fts_search(query, limit, include_deleted=include_deleted)

    def multi_modal_search(
        self,
        query: str,
        limit: int | None = None,
        use_mmr: bool = True,
        filter_document_ids: list[int] | None = None,
        asset_types: list[AssetType] | None = None,
        include_deleted: bool = False,
    ) -> list[SearchResult]:
        """
        Perform search across all enabled modalities with fair weighting.

        Searches TEXT, CODE (if enabled), and VISION (if enabled) vector tables
        in parallel, plus FTS. Results are normalized within each modality and
        then combined with equal weighting.

        This ensures all content types (documents, code, images) have equal
        opportunity to appear in search results.

        Args:
            query: The search query.
            limit: Maximum number of results to return.
            use_mmr: Whether to use MMR for diverse results.
            filter_document_ids: Optional list of document IDs to filter.
            asset_types: Optional list of asset types to filter.

        Returns:
            List of search results from all modalities.
        """
        limit = limit or SEARCH_LIMIT

        # Collect results from all enabled modalities
        modality_results: dict[str, list[SearchResult]] = {}

        # 1. TEXT modality (always enabled)
        text_results = self._vector_search(query, limit * 2, include_deleted=include_deleted)
        if text_results:
            modality_results["text"] = text_results

        # 2. CODE modality (if enabled)
        if ENABLE_CODE_EMBEDDINGS:
            code_results = self.vector_search_by_modality(
                query, EmbeddingModality.CODE, limit * 2, include_deleted=include_deleted
            )
            if code_results:
                modality_results["code"] = code_results

        # 3. VISION modality (if enabled)
        if ENABLE_VISION_EMBEDDINGS:
            vision_results = self.vector_search_by_modality(
                query, EmbeddingModality.VISION, limit * 2, include_deleted=include_deleted
            )
            if vision_results:
                modality_results["vision"] = vision_results

        # 4. FTS (keyword search)
        fts_results = self._fts_search(query, limit * 2, include_deleted=include_deleted)
        if fts_results:
            modality_results["fts"] = fts_results

        # Normalize and merge results
        combined = self._merge_multi_modal_results(modality_results)

        # Filter by document IDs if specified
        if filter_document_ids:
            combined = [r for r in combined if r.document_id in filter_document_ids]

        # Filter by asset types if specified
        if asset_types:
            combined = [r for r in combined if r.asset_type in asset_types]

        # Apply MMR if requested
        if use_mmr and len(combined) > limit:
            combined = self._apply_mmr(query, combined, limit)
        else:
            combined = combined[:limit]

        return combined

    def _merge_multi_modal_results(
        self,
        modality_results: dict[str, list[SearchResult]],
    ) -> list[SearchResult]:
        """
        Merge results from multiple modalities with fair normalization.

        Each modality's scores are normalized to 0-1 (best in modality = 1.0),
        then combined with equal weighting. Chunks found in multiple modalities
        get boosted scores.

        Args:
            modality_results: Dict mapping modality name to its results.

        Returns:
            Merged and scored results.
        """
        # Define weights for each modality (equal by default)
        modality_weights = {
            "text": MODALITY_WEIGHT_TEXT,
            "code": MODALITY_WEIGHT_CODE,
            "vision": MODALITY_WEIGHT_VISION,
            "fts": MODALITY_WEIGHT_FTS,
        }

        # Normalize scores within each modality
        normalized: dict[str, dict[int, float]] = {}
        for modality, results in modality_results.items():
            if not results:
                continue

            # Find max score in this modality for normalization
            max_score = max(r.score for r in results) if results else 1.0
            if max_score == 0:
                max_score = 1.0

            # Normalize to 0-1 scale
            normalized[modality] = {r.chunk_id: r.score / max_score for r in results}

        # Build combined results
        combined: dict[int, SearchResult] = {}
        chunk_modality_scores: dict[int, dict[str, float]] = {}

        # Collect all chunks and their modality scores
        for modality, results in modality_results.items():
            for r in results:
                chunk_id = r.chunk_id

                # Store the result object (first occurrence wins)
                if chunk_id not in combined:
                    combined[chunk_id] = SearchResult(
                        chunk_id=r.chunk_id,
                        document_id=r.document_id,
                        document_path=r.document_path,
                        content=r.content,
                        heading_path=r.heading_path,
                        score=0.0,
                        vector_score=r.vector_score,
                        fts_score=r.fts_score,
                        snippet=r.snippet,
                        asset_type=r.asset_type,
                    )
                    chunk_modality_scores[chunk_id] = {}

                # Store normalized score for this modality
                if modality in normalized and chunk_id in normalized[modality]:
                    chunk_modality_scores[chunk_id][modality] = normalized[modality][chunk_id]

                # Update specific scores on the result
                if modality == "fts":
                    combined[chunk_id].fts_score = r.fts_score
                    combined[chunk_id].snippet = r.snippet
                elif modality == "text":
                    combined[chunk_id].vector_score = r.vector_score

        # Score each chunk on the modalities it actually matched. Dividing by
        # the total weight of *all* enabled modalities (the previous approach)
        # pinned every single-modality hit to 1/N — the observed 0.25 ceiling
        # when all four modalities were enabled. Chunks that agree across
        # multiple modalities get a small overlap bonus.
        for chunk_id, result in combined.items():
            scores = chunk_modality_scores.get(chunk_id, {})
            if not scores:
                result.score = 0.0
                continue

            present_weight = sum(modality_weights.get(m, 1.0) for m in scores)
            weighted_sum = sum(norm * modality_weights.get(m, 1.0) for m, norm in scores.items())
            avg = weighted_sum / present_weight
            overlap_bonus = 1.0 + 0.05 * (len(scores) - 1)
            result.score = min(1.0, avg * overlap_bonus)

        # Sort by combined score
        results = sorted(combined.values(), key=lambda x: x.score, reverse=True)

        return results

    def vector_search_by_modality(
        self,
        query: str,
        modality: EmbeddingModality,
        limit: int | None = None,
        include_deleted: bool = False,
    ) -> list[SearchResult]:
        """
        Perform vector search using modality-specific embeddings.

        Uses specialized embedding models (CodeBERT for code, CLIP for vision)
        when enabled, providing better semantic matching for that content type.

        Args:
            query: The search query (natural language description).
            modality: The embedding modality to search (CODE, VISION, TEXT).
            limit: Maximum number of results.

        Returns:
            List of search results from the specified modality.
            Empty list if modality is not enabled.
        """
        limit = limit or SEARCH_LIMIT

        # Get modality-specific embedder
        embedder = get_embedder_for_modality(modality)
        if embedder is None:
            logger.info(f"Modality {modality.value} not enabled, returning empty results")
            return []

        # Generate query embedding with modality-specific model
        try:
            query_embedding = embedder.embed_query(query)
        except ImportError as e:
            logger.warning("Modality %s failed to load model: %s", modality.value, e)
            return []

        # Search the modality-specific vector table
        results = self.vector_store.search_by_modality(
            query_embedding, modality, limit=limit, include_deleted=include_deleted
        )

        return [
            SearchResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                document_path=r.document_path,
                content=r.content,
                heading_path=r.heading_path,
                score=1.0 - r.distance,
                vector_score=1.0 - r.distance,
                asset_type=AssetType(r.asset_type),
            )
            for r in results
        ]

    def _vector_search(
        self, query: str, limit: int, include_deleted: bool = False
    ) -> list[SearchResult]:
        """Perform vector similarity search."""
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(
            query_embedding, limit=limit, include_deleted=include_deleted
        )

        return [
            SearchResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                document_path=r.document_path,
                content=r.content,
                heading_path=r.heading_path,
                score=1.0 - r.distance,  # Convert distance to similarity
                vector_score=1.0 - r.distance,
                asset_type=AssetType(r.asset_type),
            )
            for r in results
        ]

    def _fts_search(
        self, query: str, limit: int, include_deleted: bool = False
    ) -> list[SearchResult]:
        """Perform full-text search."""
        results = self.fts_store.search(query, limit=limit, include_deleted=include_deleted)

        # Normalize FTS scores (BM25 scores are negative, more negative = better)
        if not results:
            return []

        # Convert BM25 scores to positive similarities (0-1 scale)
        # BM25 returns negative scores where more negative = better match
        # So abs(rank) for the best match is highest
        max_rank = max(abs(r.rank) for r in results) if results else 1.0
        if max_rank == 0:
            max_rank = 1.0

        return [
            SearchResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                document_path=r.document_path,
                content=r.content,
                heading_path=r.heading_path,
                score=abs(r.rank) / max_rank,  # Normalize to 0-1, higher = better
                fts_score=abs(r.rank) / max_rank,
                snippet=r.snippet,
                asset_type=AssetType(r.asset_type) if r.asset_type else AssetType.TEXT,
            )
            for r in results
        ]

    def _combine_results(
        self,
        vector_results: list[SearchResult],
        fts_results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Combine vector and FTS results with weighted scoring.

        Args:
            vector_results: Results from vector search.
            fts_results: Results from FTS search.

        Returns:
            Combined and scored results.
        """
        # Build lookup by chunk_id
        combined: dict[int, SearchResult] = {}

        # Add vector results
        for r in vector_results:
            combined[r.chunk_id] = SearchResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                document_path=r.document_path,
                content=r.content,
                heading_path=r.heading_path,
                score=0.0,
                vector_score=r.vector_score,
                fts_score=None,
                asset_type=r.asset_type,
            )

        # Add/merge FTS results
        for r in fts_results:
            if r.chunk_id in combined:
                combined[r.chunk_id].fts_score = r.fts_score
                combined[r.chunk_id].snippet = r.snippet
            else:
                combined[r.chunk_id] = SearchResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    document_path=r.document_path,
                    content=r.content,
                    heading_path=r.heading_path,
                    score=0.0,
                    vector_score=None,
                    fts_score=r.fts_score,
                    snippet=r.snippet,
                    asset_type=r.asset_type,
                )

        # Calculate combined scores
        for result in combined.values():
            vec_score = result.vector_score or 0.0
            fts_score = result.fts_score or 0.0

            # Weighted combination
            result.score = self.hybrid_alpha * vec_score + (1 - self.hybrid_alpha) * fts_score

        # Sort by combined score
        results = sorted(combined.values(), key=lambda x: x.score, reverse=True)

        return results

    def _apply_mmr(
        self,
        query: str,
        candidates: list[SearchResult],
        limit: int,
    ) -> list[SearchResult]:
        """
        Apply Max Marginal Relevance to select diverse results.

        MMR balances relevance to query with diversity among results.
        For multi-modal results, allocates slots proportionally to each
        modality and runs MMR within each group, ensuring fair representation.

        Args:
            query: The search query.
            candidates: Candidate results to select from.
            limit: Number of results to select.

        Returns:
            Diversified list of results.
        """
        if len(candidates) <= limit:
            return candidates

        # Group candidates by embedding dimension (modality)
        # This allows us to handle TEXT (384-dim), CODE (768-dim), VISION (512-dim)
        dimension_groups: dict[int, list[tuple[SearchResult, np.ndarray]]] = {}

        for result in candidates:
            embedding = self.vector_store.get_embedding(result.chunk_id)
            if embedding:
                dim = len(embedding)
                if dim not in dimension_groups:
                    dimension_groups[dim] = []
                dimension_groups[dim].append((result, np.array(embedding)))

        # If no embeddings found, fall back to simple ranking
        if not dimension_groups:
            return candidates[:limit]

        # Get query embeddings for each modality
        query_embeddings: dict[int, np.ndarray] = {}

        # Text embedding (always available)
        text_embedding = np.array(self.embedder.embed_query(query))
        query_embeddings[len(text_embedding)] = text_embedding

        # Code embedding (if enabled and dimension present)
        if ENABLE_CODE_EMBEDDINGS:
            try:
                code_embedder = get_embedder_for_modality(EmbeddingModality.CODE)
                if code_embedder:
                    code_embedding = np.array(code_embedder.embed_query(query))
                    query_embeddings[len(code_embedding)] = code_embedding
            except Exception:
                logger.debug(
                    "Failed to generate code query embedding, skipping CODE modality in MMR"
                )

        # Vision embedding (if enabled and dimension present)
        if ENABLE_VISION_EMBEDDINGS:
            try:
                vision_embedder = get_embedder_for_modality(EmbeddingModality.VISION)
                if vision_embedder:
                    vision_embedding = np.array(vision_embedder.embed_query(query))
                    query_embeddings[len(vision_embedding)] = vision_embedding
            except Exception:
                logger.debug(
                    "Failed to generate vision query embedding, skipping VISION modality in MMR"
                )

        # Separate candidates with and without query embeddings
        modality_candidates: dict[int, list[tuple[SearchResult, np.ndarray]]] = {}
        non_mmr_candidates: list[SearchResult] = []

        for dim, group in dimension_groups.items():
            if dim in query_embeddings:
                modality_candidates[dim] = group
            else:
                for result, _emb in group:
                    non_mmr_candidates.append(result)

        # If no MMR candidates, return non-MMR candidates by score
        if not modality_candidates:
            non_mmr_candidates.sort(key=lambda x: x.score, reverse=True)
            return non_mmr_candidates[:limit]

        # Calculate total candidates for proportional allocation
        total_mmr = sum(len(group) for group in modality_candidates.values())
        total_non_mmr = len(non_mmr_candidates)
        total_all = total_mmr + total_non_mmr

        # Allocate slots proportionally to each modality
        # Ensure each non-empty modality gets at least 1 slot
        modality_slots: dict[int, int] = {}

        # First pass: allocate proportional slots
        for dim, group in modality_candidates.items():
            proportion = len(group) / total_all
            slots = max(1, round(limit * proportion))  # At least 1 slot
            modality_slots[dim] = slots

        # Non-MMR slots
        non_mmr_slots = 0
        if total_non_mmr > 0:
            non_mmr_slots = max(1, round(limit * total_non_mmr / total_all))

        # Adjust if we over-allocated
        total_allocated = sum(modality_slots.values()) + non_mmr_slots
        if total_allocated > limit:
            # Scale down proportionally
            scale = limit / total_allocated
            for dim in modality_slots:
                modality_slots[dim] = max(1, int(modality_slots[dim] * scale))
            non_mmr_slots = max(0, limit - sum(modality_slots.values()))

        # Run MMR within each modality group
        all_selected: list[SearchResult] = []

        for dim, group in modality_candidates.items():
            slots = modality_slots.get(dim, 0)
            if slots == 0:
                continue

            query_emb = query_embeddings[dim]

            # Run MMR for this modality
            selected_from_modality = self._mmr_for_group(group, query_emb, slots)
            all_selected.extend(selected_from_modality)

        # Add non-MMR candidates sorted by score
        if non_mmr_slots > 0:
            non_mmr_candidates.sort(key=lambda x: x.score, reverse=True)
            all_selected.extend(non_mmr_candidates[:non_mmr_slots])

        # Sort final selection by original score for display
        all_selected.sort(key=lambda x: x.score, reverse=True)

        return all_selected[:limit]

    def _mmr_for_group(
        self,
        group: list[tuple[SearchResult, np.ndarray]],
        query_embedding: np.ndarray,
        limit: int,
    ) -> list[SearchResult]:
        """
        Run MMR selection within a single modality group.

        Args:
            group: List of (SearchResult, embedding) tuples.
            query_embedding: Query embedding for this modality.
            limit: Number of results to select.

        Returns:
            Selected results.
        """
        if len(group) <= limit:
            return [r for r, _e in group]

        # Build embeddings lookup
        embeddings = {r.chunk_id: emb for r, emb in group}

        selected: list[SearchResult] = []
        remaining = list(group)

        while len(selected) < limit and remaining:
            best_score = -float("inf")
            best_idx = 0

            for i, (_candidate, cand_emb) in enumerate(remaining):
                # Relevance to query
                relevance = self._cosine_similarity(query_embedding, cand_emb)

                # Diversity: max similarity to already selected
                if selected:
                    max_sim = max(
                        self._cosine_similarity(cand_emb, embeddings[s.chunk_id]) for s in selected
                    )
                else:
                    max_sim = 0.0

                # MMR score
                mmr_score = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx)[0])

        return selected

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def search_in_document(
        self,
        query: str,
        document_id: int,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """
        Search within a specific document.

        Args:
            query: The search query.
            document_id: The document ID to search within.
            limit: Maximum number of results.

        Returns:
            List of search results from the document.
        """
        return self.search(
            query,
            limit=limit,
            use_mmr=False,
            filter_document_ids=[document_id],
        )
