"""
Write-side data models shared between the orchestrator and storage.

The orchestrator does all the slow work (parsing, chunking, embedding) and
produces a :class:`PreparedDocument` -- a fully-resolved document with
deterministic ids and embeddings ready to be written. Storage backends consume
it inside a single transaction. Keeping these models in the storage layer avoids
a storage -> orchestrator import (which would invert the dependency).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from librarian.types import AssetType, EmbeddingModality

__all__ = ["PreparedChunk", "PreparedDocument"]


@dataclass
class PreparedChunk:
    """A chunk resolved to its deterministic id and (optional) embedding."""

    chunk_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    heading_path: str | None = None
    chunk_source_uri: str | None = None
    asset_type: AssetType = AssetType.TEXT
    modality: EmbeddingModality = EmbeddingModality.TEXT
    embedding: list[float] | None = None
    model_version: str | None = None
    # Type-specific metadata (e.g. VLM caption + ``processing_status``) persisted
    # to the ``chunks.modality_data`` JSON column.
    modality_data: dict[str, Any] | None = None


@dataclass
class PreparedDocument:
    """A document resolved to its deterministic id and prepared chunks."""

    document_id: str
    path: str
    title: str | None
    content: str
    metadata: dict[str, Any]
    asset_type: AssetType = AssetType.TEXT
    document_source_uri: str | None = None
    source_created_at: datetime | None = None
    document_size: int | None = None
    file_mtime: float | None = None
    chunks: list[PreparedChunk] = field(default_factory=list)
