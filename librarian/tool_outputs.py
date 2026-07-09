"""TypedDict schemas for Librarian MCP tool outputs.

Defining structured outputs lets MCP clients (and the LLMs driving them)
discover the exact shape of every tool's response without inspecting docs.

Optional fields use the inheritance + ``total=False`` pattern instead of
``NotRequired`` so that pydantic's TypedDict-to-model conversion (used by
``arcade_core.catalog.create_model_from_typeddict``) can introspect them.

Imports ``TypedDict`` from ``typing_extensions`` rather than ``typing``:
pydantic refuses ``typing.TypedDict`` on Python < 3.12, since the stdlib
version doesn't surface the per-field metadata pydantic needs.
"""

from typing import Any

from typing_extensions import TypedDict


class IndexFileResultBase(TypedDict):
    path: str
    status: str  # "created" | "updated" | "skipped"


class IndexFileResult(IndexFileResultBase, total=False):
    """Per-file outcome inside a directory-index run."""

    chunks: int
    title: str
    asset_type: str
    reason: str


class IndexErrorEntry(TypedDict):
    path: str
    error: str


class _IndexDirectoryRequired(TypedDict):
    directory: str
    total_files: int
    indexed: int
    updated: int
    skipped: int
    errors: list[IndexErrorEntry]
    files: list[IndexFileResult]


class IndexDirectoryOutput(_IndexDirectoryRequired, total=False):
    message: str


class SiblingItem(TypedDict):
    name: str
    type: str  # "file" | "directory"


class LocationInfo(TypedDict):
    source: str | None
    source_path: str | None
    directory: str
    breadcrumb: list[str]
    breadcrumb_display: str


class AddContext(TypedDict):
    siblings: list[SiblingItem]
    sibling_count: int


class _AddOutputRequired(TypedDict):
    status: str  # "stored" | "stored_partial" | "stored_file_only"
    message: str
    path: str
    chunks: int
    indexed: bool
    location: LocationInfo


class AddOutput(_AddOutputRequired, total=False):
    title: str | None
    context: AddContext
    warning: str


class UpdateOutput(TypedDict):
    status: str  # "updated"
    message: str
    path: str
    title: str | None
    chunks: int


class SearchHit(TypedDict):
    chunk_id: str  # deterministic hash (v0.14); falls back to the surrogate id as str
    document_id: int
    document_path: str
    content: str
    heading_path: str | None
    score: float
    snippet: str | None
    asset_type: str
    chunk_source_uri: str | None
    chunk_index: int | None
    document_size: int | None
    source_created_at: str | None
    # ISO-8601 tombstone timestamp when the chunk is soft-deleted, else None.
    # Only non-null hits appear when a search is run with include_deleted=True,
    # so an agent can tell "removed from source, kept for history" from current.
    deleted_at: str | None
    # Type-specific metadata (e.g. ``{"processing_status": "failed"}`` for an
    # image whose VLM caption errored). Present only when the chunk carries it.
    modality_data: dict[str, Any] | None


class ContextChunk(TypedDict):
    """One neighbor chunk returned by ``expand_context``.

    Same identity/shape as a search hit's chunk, minus the relevance score:
    these are positional neighbors, not ranked matches.
    """

    chunk_id: str  # deterministic hash (v0.14); falls back to the surrogate id as str
    document_id: int
    document_path: str
    content: str
    heading_path: str | None
    chunk_index: int
    asset_type: str
    chunk_source_uri: str | None
    # ISO-8601 tombstone timestamp when soft-deleted (only populated for
    # neighbors returned under include_deleted=True), else None.
    deleted_at: str | None


class _ReadOutputRequired(TypedDict):
    id: int | None
    path: str
    content: str
    indexed: bool


class ReadOutput(_ReadOutputRequired, total=False):
    title: str | None
    metadata: dict[str, Any]
    created_at: str | None
    updated_at: str | None
    note: str


class _RemoveOutputRequired(TypedDict):
    path: str
    removed_from_index: bool
    message: str


class RemoveOutput(_RemoveOutputRequired, total=False):
    file_deleted: bool
    note: str


class DocumentSummary(TypedDict):
    id: int | None
    path: str
    title: str | None
    metadata: dict[str, Any]
    created_at: str | None
    updated_at: str | None


class SourceSummary(TypedDict):
    name: str
    path: str
    type: str  # "file" | "directory"
    document_count: int
    chunk_count: int
    exists: bool
    recursive: bool
    added_at: str | None


class LibraryConfig(TypedDict):
    documents_path: str
    chunk_size: int
    chunk_overlap: int
    search_limit: int
    mmr_lambda: float
    hybrid_alpha: float


class LibraryStats(TypedDict):
    document_count: int
    chunk_count: int
    embedding_count: int
    config: LibraryConfig


# =============================================================================
# get_library_overview — unified introspection (sections | stats | tree)
# =============================================================================


class _OverviewSectionBase(TypedDict):
    name: str
    path: str
    description: str


class OverviewSection(_OverviewSectionBase, total=False):
    """One subdirectory entry inside the SECTIONS view."""

    document_count: int
    has_subdirectories: bool
    subdirectories: list[str]


class _OverviewSourceBlockBase(TypedDict):
    source: str
    available: bool


class OverviewSourceBlock(_OverviewSourceBlockBase, total=False):
    """One source block inside the SECTIONS view."""

    source_path: str
    type: str  # "file" | "directory"
    sections: list[OverviewSection]
    total_documents: int | None
    description: str
    path: str
    error: str


class _TreeNodeBase(TypedDict):
    name: str
    type: str  # "file" | "directory"
    path: str


class TreeNode(_TreeNodeBase, total=False):
    """One node in the recursive TREE view.

    `children` is typed as ``list[dict]`` rather than ``list["TreeNode"]`` to
    keep arcade-mcp-server's pydantic schema generator from recursing on the
    self-reference. At runtime each entry is still a TreeNode-shaped dict.
    """

    children: list[dict[str, Any]]
    subdirectory_count: int
    file_count: int
    error: str


class _TreeSourceBlockBase(TypedDict):
    name: str
    path: str
    exists: bool


class TreeSourceBlock(_TreeSourceBlockBase, total=False):
    """One source block inside the TREE view."""

    is_file: bool
    structure: TreeNode
    error: str


class _OverviewResultBase(TypedDict):
    view: str  # "sections" | "stats" | "tree"


class OverviewResult(_OverviewResultBase, total=False):
    """get_library_overview return — keys present depend on view."""

    # SECTIONS
    sections: list[OverviewSourceBlock]
    total_sources: int
    usage_hint: str
    default_path: str
    message: str
    # STATS — flattened from LibraryStats so the caller doesn't have to
    # navigate a nested dict to pluck `document_count`.
    document_count: int
    chunk_count: int
    embedding_count: int
    config: LibraryConfig
    # TREE
    sources: list[TreeSourceBlock]


# =============================================================================
# suggest_library_location
# =============================================================================


class SimilarDocument(TypedDict):
    title: str
    relevance: float


class _LocationSuggestionBase(TypedDict):
    path: str
    source: str
    confidence: float
    breadcrumb_display: str
    reason: str


class LocationSuggestion(_LocationSuggestionBase, total=False):
    """One ranked entry in suggest_library_location."""

    similar_documents: list[SimilarDocument]


class _SuggestResultBase(TypedDict):
    title: str
    suggestions: list[LocationSuggestion]


class SuggestResult(_SuggestResultBase, total=False):
    """suggest_library_location aggregate result."""

    search_method: str  # "hybrid" | "keyword" | "none"
    usage_hint: str
    message: str
    default_path: str
