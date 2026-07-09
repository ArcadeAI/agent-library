"""
Core types for librarian multi-modal support.

Defines all type representations used throughout the system:
- Enums: Asset types, source types, languages, strategies
- Ingested types: ParsedDocument, Section (from parsers)
- Computed types: TextChunk (from chunking)
- Storage types: Document, Chunk (database records)
- Retrieval types: SearchResult (search output)
- Codebase types: CodebaseMetadata, SourceConfig
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# =============================================================================
# Enums
# =============================================================================


class AssetType(str, Enum):
    """Asset type categorization."""

    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    PDF = "pdf"
    MULTIMODAL = "multimodal"


class SourceType(str, Enum):
    """Source type categorization."""

    DOCUMENTS = "documents"
    CODEBASE = "codebase"
    KNOWLEDGE_BASE = "knowledge_base"
    ASSETS = "assets"
    MIXED = "mixed"


class ChunkingStrategy(str, Enum):
    """Chunking strategies for different content types."""

    HEADERS = "headers"
    PARAGRAPHS = "paragraphs"
    SENTENCES = "sentences"
    FIXED = "fixed"
    CODE_BLOCKS = "code_blocks"
    PAGES = "pages"
    VISUAL_PATCHES = "visual_patches"


class EmbeddingModality(str, Enum):
    """Embedding modality types."""

    TEXT = "text"
    CODE = "code"
    VISION = "vision"
    HYBRID = "hybrid"


class ProgrammingLanguage(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    CSHARP = "csharp"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    SHELL = "shell"
    OTHER = "other"


class CodeSymbolType(str, Enum):
    """Types of code symbols."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    INTERFACE = "interface"
    TYPE = "type"
    MODULE = "module"


class SearchMode(str, Enum):
    """Search modes."""

    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"


class ProcessingStatus(str, Enum):
    """Per-chunk processing outcome recorded in ``chunks.modality_data``.

    Drives the ``libr reprocess`` retry loop (see the vision pipeline, #53):

    - OK: content was produced successfully.
    - FAILED: a transient failure (VLM/OCR error, missing config) -- retryable;
      this is what ``libr reprocess`` matches by default.
    - UNCAPTIONED: an image indexed with captions disabled -- retryable once
      ``IMAGE_GENERATE_CAPTIONS`` is on (the backfill story).
    - OCR_UNAVAILABLE: a PDF wanted OCR but the OCR dependencies were absent --
      retryable once they're installed.
    - UNSUPPORTED: a terminal skip (e.g. an image format no VLM provider
      accepts); reprocess deliberately does not match this.
    """

    OK = "ok"
    FAILED = "failed"
    UNCAPTIONED = "uncaptioned"
    OCR_UNAVAILABLE = "ocr_unavailable"
    UNSUPPORTED = "unsupported"


class LibraryView(str, Enum):
    """View selector for the unified get_library_overview tool.

    - SECTIONS: top-level subdirectories of each source with doc counts
      and sample titles. Best for "where do I add new content?"
    - STATS: library totals + current configuration knobs.
    - TREE: recursive filesystem structure of each source (uses depth).
    """

    SECTIONS = "sections"
    STATS = "stats"
    TREE = "tree"


# Timeframe is canonically defined in librarian.utils.timeframe
# Re-exported here for backward compatibility
from librarian.utils.timeframe import Timeframe as Timeframe  # noqa: E402

# =============================================================================
# Ingested Types (from parsers)
# =============================================================================


@dataclass
class Section:
    """
    Represents a section of a document.

    A section is defined by a header and contains content until the next
    header of the same or higher level.
    """

    title: str
    level: int
    content: str
    start_pos: int
    end_pos: int
    children: list["Section"] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """
    Represents a parsed document with multi-modal support.

    Contains extracted structure, metadata, and content from any supported
    file type (markdown, code, PDF, image).
    """

    path: str
    title: str | None
    content: str
    metadata: dict[str, Any]
    sections: list[Section]
    raw_content: str | bytes
    asset_type: AssetType
    modality_data: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Computed Types (from transform pipeline)
# =============================================================================


@dataclass
class TextChunk:
    """
    Represents a chunk of text ready for embedding.

    Created by chunkers from parsed documents, with position tracking
    for source attribution.
    """

    content: str
    index: int
    start_char: int
    end_char: int
    heading_path: str | None = None
    metadata: dict[str, Any] | None = None
    # Per-chunk type-specific metadata persisted to ``chunks.modality_data``
    # (e.g. ``{"processing_status": "failed"}`` for a VLM caption that errored).
    modality_data: dict[str, Any] | None = None


# =============================================================================
# Storage Types (database records)
# =============================================================================


@dataclass
class Document:
    """
    Represents a document record in the database.

    Stores full document content and metadata for retrieval and display.
    """

    id: int | None
    path: str
    title: str | None
    content: str
    metadata: dict[str, Any]
    asset_type: AssetType = AssetType.TEXT
    created_at: datetime | None = None
    updated_at: datetime | None = None
    file_mtime: float | None = None


@dataclass
class Chunk:
    """
    Represents a chunk record in the database.

    Links to a parent document and stores chunk content with embeddings.
    """

    id: int | None
    document_id: int
    content: str
    heading_path: str | None
    chunk_index: int
    start_char: int
    end_char: int
    embedding: list[float] | None = None
    asset_type: AssetType = AssetType.TEXT
    modality: EmbeddingModality = EmbeddingModality.TEXT
    auxiliary_embeddings: dict[str, list[float]] | None = None


# =============================================================================
# Retrieval Types (search output)
# =============================================================================


@dataclass
class SearchResult:
    """
    Represents a search result returned by the retrieval system.

    Contains matched chunk, scores, and document context with modality info.
    """

    chunk_id: int
    document_id: int
    document_path: str
    content: str
    heading_path: str | None
    score: float
    vector_score: float | None = None
    fts_score: float | None = None
    snippet: str | None = None
    asset_type: AssetType = AssetType.TEXT
    modality: EmbeddingModality = EmbeddingModality.TEXT
    modality_data: dict[str, Any] | None = None


# =============================================================================
# Codebase Types
# =============================================================================


@dataclass
class CodebaseMetadata:
    """Metadata for codebase sources."""

    git_remote: str | None = None
    primary_language: ProgrammingLanguage | None = None
    languages: list[ProgrammingLanguage] = field(default_factory=list)
    framework: str | None = None
    dependencies: dict[str, Any] = field(default_factory=dict)
    exclude_patterns: list[str] = field(default_factory=list)
    include_patterns: list[str] = field(default_factory=list)
    max_file_size_kb: int = 1024


@dataclass
class SourceConfig:
    """Configuration for a registered source."""

    name: str
    path: str
    source_type: SourceType = SourceType.DOCUMENTS
    recursive: bool = True
    added_at: datetime | None = None
    last_indexed: datetime | None = None
    codebase_metadata: CodebaseMetadata | None = None


# =============================================================================
# Code Analysis Types
# =============================================================================


@dataclass
class CodeSymbol:
    """Represents a code symbol (function, class, etc)."""

    name: str
    symbol_type: CodeSymbolType
    line_start: int
    line_end: int
    docstring: str | None = None
    signature: str | None = None
    parent: str | None = None


@dataclass
class CodeReference:
    """Represents a reference to a code symbol."""

    file_path: str
    line_number: int
    context: str
    usage_type: str
