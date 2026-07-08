#!/usr/bin/env python3
"""
Librarian MCP Server - Agent Knowledge Library.

A personal knowledge library for AI agents to store, search, and retrieve
any text, documents, notes, and information. Think of it as an agent's
personal library that persists across sessions.

Usage:
    uv run python -m librarian.server stdio    # For Claude Desktop, CLI tools
    uv run python -m librarian.server http     # For Cursor, VS Code (HTTP streaming)

Note: invoke via `-m librarian.server`, not `librarian/server.py`. The latter
puts the `librarian/` directory on sys.path[0], where `librarian/types.py`
shadows stdlib `types` and crashes import on `from types import GenericAlias`.
"""

import logging
import sys
from pathlib import Path
from typing import Annotated, Any

from arcade_core.metadata import (
    Behavior,
    Operation,
    ToolMetadata,
)
from arcade_mcp_server import Context, MCPApp
from arcade_tdk.errors import RetryableToolError, ToolExecutionError

from librarian import __version__
from librarian.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_PATH,
    ENABLE_CODE_EMBEDDINGS,
    ENABLE_OPTIONAL_TOOLS,
    ENABLE_VISION_EMBEDDINGS,
    HYBRID_ALPHA,
    MMR_LAMBDA,
    SEARCH_LIMIT,
    SERVER_HOST,
    SERVER_PORT,
    ensure_directories,
)
from librarian.processing.embed import get_embedder
from librarian.processing.parsers.base import FileReadError, FileReadTimeoutError
from librarian.retrieval.search import HybridSearcher
from librarian.sources.ignore import (
    GitignoreMatcher,
    LibrarianTrackMatcher,
    normalize_force_include,
    should_skip_file,
)
from librarian.storage.database import get_database
from librarian.storage.factory import get_metadata_store, get_storage
from librarian.tool_outputs import (
    AddOutput,
    ContextChunk,
    DocumentSummary,
    IndexDirectoryOutput,
    LibraryConfig,
    OverviewResult,
    OverviewSection,
    OverviewSourceBlock,
    ReadOutput,
    RemoveOutput,
    SearchHit,
    SuggestResult,
    TreeNode,
    TreeSourceBlock,
    UpdateOutput,
)
from librarian.types import AssetType, EmbeddingModality, LibraryView, SearchMode
from librarian.utils.timeframe import Timeframe, get_timeframe_bounds, parse_date_string

logger = logging.getLogger(__name__)

# Create the MCP application
app = MCPApp(
    name="Librarian",
    version=__version__,
    log_level="INFO",
)

# Ensure required directories exist
ensure_directories()


# =============================================================================
# Helper Functions
# =============================================================================


def _get_searcher() -> HybridSearcher:
    """Get a configured hybrid searcher instance."""
    embedder = get_embedder()
    return HybridSearcher(embedder)


def _process_and_index_file(file_path: Path) -> dict[str, Any]:
    """Process a file and add it to the index via the v0.14 Orchestrator."""
    from librarian.orchestrator import Orchestrator

    storage = get_storage()
    return Orchestrator(storage=storage).index_file(file_path)


def _should_skip_file(
    file_path: Path,
    supported_extensions: set[str],
    gitignore_matcher: GitignoreMatcher | None = None,
    force_include: frozenset[Path] | None = None,
    track_matcher: LibrarianTrackMatcher | None = None,
) -> bool:
    """Check if a file should be skipped during indexing."""
    return should_skip_file(
        file_path,
        supported_extensions,
        gitignore_matcher,
        force_include=force_include,
        track_matcher=track_matcher,
    )


def _resolve_path(raw_path: str, kind: str = "path") -> Path:
    """Resolve and validate a file or directory path.

    Raises RetryableToolError on invalid input so the LLM can correct the path
    and retry. The error carries plain instructions for the next step in
    `additional_prompt_content`.
    """
    if not raw_path or not raw_path.strip():
        raise RetryableToolError(
            message=f"You need to provide a {kind}.",
            additional_prompt_content=(
                f"Pass an absolute {kind} (one starting with '/'). "
                "If you don't know the path, call list_library_contents to see "
                "indexed documents, or search_library to find one by content."
            ),
        )

    path = Path(raw_path).expanduser()

    try:
        path = path.resolve()
    except OSError as e:
        raise RetryableToolError(
            message=f"That {kind} ({raw_path}) couldn't be resolved.",
            additional_prompt_content=(
                f"Try again with a clean absolute {kind}. "
                "Drop any quoting, environment variables, or special characters."
            ),
        ) from e

    return path


# =============================================================================
# Library Ingestion Tools
# =============================================================================


@app.tool(  # type: ignore[arg-type]
    metadata=ToolMetadata(
        behavior=Behavior(
            operations=[Operation.CREATE, Operation.UPDATE],
            read_only=False,
            destructive=False,
            idempotent=True,
            open_world=False,
        ),
    ),
)
async def index_directory_to_library(
    context: Context,
    directory: Annotated[str, "Absolute path to directory containing files to add to the library"],
    include_ignored: Annotated[
        bool,
        "If True, index files even when matched by a .gitignore under the directory.",
    ] = False,
    force_include: Annotated[
        list[str] | None,
        (
            "Files or directories to always index, even when matched by a .gitignore "
            "or by the skip-dirs baseline (node_modules, __pycache__, etc.). "
            "Pointing at a directory force-includes everything underneath. "
            "Has no effect on unsupported or binary file types."
        ),
    ] = None,
) -> Annotated[
    IndexDirectoryOutput,
    "Per-directory index summary with counts and a per-file status list.",
]:
    """
    Add all documents from a directory to the agent's library.

    Use this to bulk-import text files, notes, documentation, or any
    content into the library for later search and retrieval.
    The library persists across sessions, so content added here will
    be available in future conversations.

    Recursively indexes all supported file types. Files that haven't
    changed since last indexing are automatically skipped.
    """
    # Resolve up front so discovered file paths are canonical and match the
    # resolved id/path the orchestrator writes (keeps change-detection stable).
    dir_path = (Path(directory) if directory else Path(DOCUMENTS_PATH)).resolve()

    if not dir_path.exists():
        raise RetryableToolError(
            message=f"There's no directory at {dir_path}.",
            additional_prompt_content=(
                "Try one of these:\n"
                "  - Pass an absolute path that exists on disk.\n"
                "  - Call get_library_overview to see directories already registered.\n"
                "  - Ask the user where the files live."
            ),
        )

    if not dir_path.is_dir():
        raise RetryableToolError(
            message=f"{dir_path} is a file, not a directory.",
            additional_prompt_content=(
                "If you want to index every file under a folder, pass that folder's path. "
                "If you just want to add one file, use add_to_library instead."
            ),
        )

    from librarian.processing.parsers.registry import get_registry

    registry = get_registry()
    supported_extensions = registry.get_supported_extensions()

    gitignore_matcher = None if include_ignored else GitignoreMatcher(dir_path)
    track_matcher = LibrarianTrackMatcher(dir_path)
    forced_paths = normalize_force_include(force_include)

    # Walk the tree once and filter inline. Globbing once per supported
    # extension would re-walk the whole directory ~20-25x.
    all_files = [
        f
        for f in dir_path.glob("**/*")
        if not f.is_symlink()
        and f.is_file()
        and not _should_skip_file(
            f,
            supported_extensions,
            gitignore_matcher,
            force_include=forced_paths,
            track_matcher=track_matcher,
        )
    ]

    if not all_files:
        return IndexDirectoryOutput(
            directory=str(dir_path),
            total_files=0,
            indexed=0,
            updated=0,
            skipped=0,
            errors=[],
            files=[],
            message=(
                "No supported files found under this directory. "
                f"Supported extensions: {', '.join(sorted(supported_extensions))}. "
                "Install optional parsers with `uv pip install -e '.[pdf]'` or "
                "`uv pip install -e '.[vision]'` for additional formats."
            ),
        )

    results: IndexDirectoryOutput = {
        "directory": str(dir_path),
        "total_files": len(all_files),
        "indexed": 0,
        "updated": 0,
        "skipped": 0,
        "errors": [],
        "files": [],
    }

    # Build storage + orchestrator once for the whole run. Constructing them per
    # file would re-run migrate() (schema introspection + index/DDL + commit) on
    # every iteration, defeating the migrate-once singleton.
    from librarian.orchestrator import Orchestrator

    storage = get_storage()
    db = storage.metadata
    orchestrator = Orchestrator(storage=storage)

    for file_path in all_files:
        try:
            # Check if document exists and compare modification times
            existing = db.get_document_by_path(str(file_path))

            if existing:
                # Get current file modification time
                try:
                    current_mtime = file_path.stat().st_mtime
                except (OSError, TimeoutError):
                    # Can't stat file (cloud not synced, etc.) - skip it
                    logger.info("Skipping %s: cannot read file metadata", file_path.name)
                    results["files"].append({
                        "path": str(file_path),
                        "status": "skipped",
                        "reason": "cannot read file metadata",
                    })
                    results["skipped"] += 1
                    continue

                stored_mtime = existing.file_mtime

                # Skip if file hasn't been modified since last index
                if stored_mtime is not None and current_mtime <= stored_mtime:
                    results["files"].append({
                        "path": str(file_path),
                        "status": "skipped",
                    })
                    results["skipped"] += 1
                    continue

            result = orchestrator.index_file(file_path)
            results["files"].append(result)  # type: ignore[arg-type]

            if result["status"] == "created":
                results["indexed"] += 1
            else:
                results["updated"] += 1

        except ImportError as e:
            # Missing optional dependency (e.g., Pillow for images, pypdf for PDFs)
            logger.info("Skipping %s: %s", file_path.name, e)
            results["files"].append({
                "path": str(file_path),
                "status": "skipped",
                "reason": str(e),
            })
            results["skipped"] += 1
        except (FileReadTimeoutError, TimeoutError) as e:
            # File not available (iCloud not synced, network timeout, etc.)
            logger.warning("Skipping %s: %s", file_path.name, e)
            results["files"].append({
                "path": str(file_path),
                "status": "skipped",
                "reason": f"file read timeout: {e}",
            })
            results["skipped"] += 1
        except FileReadError as e:
            # I/O error (permissions, disk issues, etc.)
            logger.warning("Skipping %s: %s", file_path.name, e)
            results["files"].append({
                "path": str(file_path),
                "status": "skipped",
                "reason": str(e),
            })
            results["skipped"] += 1
        except Exception as e:
            logger.exception("Error indexing %s", file_path)
            results["errors"].append({"path": str(file_path), "error": str(e)})

    return results


def _get_sources_config() -> list[dict[str, Any]]:
    """
    Load sources configuration from the sources.json file.

    Returns:
        List of source configuration dictionaries, or empty list if file
        doesn't exist or is invalid.
    """
    import json

    sources_file = Path.home() / ".librarian" / "sources.json"
    if not sources_file.exists():
        return []
    try:
        with open(sources_file) as f:
            result = json.load(f)
            return result if isinstance(result, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _find_source_for_path(path: Path, sources: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find which source a path belongs to."""
    path_str = str(path.resolve())
    for source in sources:
        source_path = source.get("path", "")
        if path_str.startswith(source_path):
            return source
    return None


def _build_breadcrumb(path: Path, source: dict[str, Any] | None) -> list[str]:
    """Build breadcrumb path from source root to file."""
    if not source:
        return [path.name]

    source_path = Path(source.get("path", ""))
    try:
        relative = path.relative_to(source_path)
    except ValueError:
        return [path.name]
    else:
        return [source.get("name", source_path.name), *relative.parts]


def _get_siblings(path: Path, include_dirs: bool = True) -> list[dict[str, str]]:
    """
    Get sibling files and directories in the same parent folder.

    Args:
        path: The file path to find siblings for.
        include_dirs: Whether to include directories in output.

    Returns:
        List of sibling items (max 20) with name and type.
    """
    parent = path.parent
    if not parent.exists():
        return []

    siblings = []
    try:
        for item in sorted(parent.iterdir()):
            if item.name.startswith("."):
                continue
            if item == path:
                continue
            if item.is_dir() and include_dirs:
                siblings.append({"name": item.name, "type": "directory"})
            elif item.is_file() and item.suffix == ".md":
                siblings.append({"name": item.name, "type": "file"})
    except PermissionError:
        return []

    return siblings[:20]


@app.tool(  # type: ignore[arg-type]
    metadata=ToolMetadata(
        behavior=Behavior(
            operations=[Operation.CREATE],
            read_only=False,
            destructive=False,
            idempotent=False,
            open_world=False,
        ),
    ),
)
async def add_to_library(
    context: Context,
    content: Annotated[str, "The text content to store in the library"],
    title: Annotated[str, "A title or filename for this content (without .md extension)"],
    directory: Annotated[
        str, "Absolute path to directory for storage. Use get_library_overview to find valid paths."
    ] = "",
    tags: Annotated[list[str] | None, "Optional tags to categorize this content"] = None,
    metadata: Annotated[dict[str, Any] | None, "Optional additional metadata"] = None,
) -> Annotated[
    AddOutput,
    "Storage result, including final path, indexed flag, and breadcrumb location.",
]:
    """
    Store new content in the agent's library.

    IMPORTANT: Before adding content, use get_library_overview() to see available
    locations and their purposes. Pass the full directory path from that tool.

    Use this to save any text, notes, information, or knowledge that should
    be remembered and searchable later. The content is indexed for both
    semantic (meaning-based) and keyword search, making it easy to find
    relevant information in future conversations.

    Examples of what to store:
    - Meeting notes and summaries
    - Research findings and insights
    - Code documentation and explanations
    - Personal notes and reminders
    - Any information worth remembering
    """
    if not title or not title.strip():
        raise RetryableToolError(
            message="You need to pass a title.",
            additional_prompt_content=(
                "Pick a short descriptive title like 'Q2 Roadmap Notes' or "
                "'Auth Refactor Plan'. The title becomes the filename, so use "
                "letters, numbers, hyphens or spaces — no slashes."
            ),
        )

    dir_path = Path(directory) if directory else Path(DOCUMENTS_PATH)

    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RetryableToolError(
            message=f"Couldn't create the directory {dir_path}.",
            additional_prompt_content=(
                "Pick a directory you can write to. Call get_library_overview "
                "to see directories that already work, then pass one of those "
                "as the `directory` argument."
            ),
        ) from e

    filename = title.replace("/", "-").replace("\\", "-").strip()
    if not filename.endswith(".md"):
        filename = filename + ".md"

    file_path = dir_path / filename

    if file_path.exists():
        raise RetryableToolError(
            message=f"A document called '{title}' already lives at {file_path}.",
            additional_prompt_content=(
                "Decide which one you want:\n"
                f"  - To replace it, call update_library_doc with path='{file_path}'.\n"
                "  - To keep both, pick a different title and call this tool again."
            ),
        )

    meta = metadata or {}
    if tags:
        meta["tags"] = tags

    if meta:
        import yaml

        frontmatter_str = "---\n" + yaml.dump(meta, default_flow_style=False) + "---\n\n"
        content = frontmatter_str + content

    try:
        file_path.write_text(content, encoding="utf-8")
    except OSError as e:
        raise ToolExecutionError(
            message=f"Couldn't save the file to {file_path}.",
            developer_message=(
                "Try a different directory the user has write access to. "
                "Call get_library_overview to see ones that already work, "
                "then call add_to_library again with that `directory`."
            ),
        ) from e

    sources = _get_sources_config()
    source = _find_source_for_path(file_path, sources)
    breadcrumb = _build_breadcrumb(file_path, source)
    siblings = _get_siblings(file_path)

    location_info = {
        "source": source.get("name") if source else None,
        "source_path": source.get("path") if source else None,
        "directory": str(dir_path),
        "breadcrumb": breadcrumb,
        "breadcrumb_display": " → ".join(breadcrumb),
    }

    try:
        result = _process_and_index_file(file_path)
        return AddOutput(
            status="stored",
            message=f"Content '{title}' has been added to your library",
            path=str(file_path),
            title=result["title"],
            chunks=result["chunks"],
            indexed=True,
            location=location_info,  # type: ignore[typeddict-item]
            context={
                "siblings": siblings,  # type: ignore[typeddict-item]
                "sibling_count": len(siblings),
            },
        )
    except Exception as e:
        # Indexing failed (likely embedding service unavailable).
        # File is still saved - try to store document metadata without embeddings.
        logger.warning("Full indexing failed, storing without embeddings: %s", e)

        try:
            db = get_database()
            from librarian.processing.parsers.md import MarkdownParser
            from librarian.types import Document

            parser = MarkdownParser()
            parsed = parser.parse_file(file_path)
            file_mtime = file_path.stat().st_mtime

            doc = Document(
                id=None,
                path=str(file_path),
                title=parsed.title,
                content=parsed.content,
                metadata=parsed.metadata,
                file_mtime=file_mtime,
            )
            db.insert_document(doc)

            return AddOutput(
                status="stored_partial",
                message=(
                    f"Content '{title}' saved but indexing incomplete "
                    "(embedding service unavailable)"
                ),
                path=str(file_path),
                title=parsed.title,
                chunks=0,
                indexed=False,
                warning=(
                    "File saved but not fully indexed. Keyword search will work, "
                    "but semantic search won't find this content until re-indexed."
                ),
                location=location_info,  # type: ignore[typeddict-item]
                context={
                    "siblings": siblings,  # type: ignore[typeddict-item]
                    "sibling_count": len(siblings),
                },
            )
        except Exception:
            # File is on disk but neither indexing nor metadata storage worked.
            logger.exception("Failed to store document metadata")
            return AddOutput(
                status="stored_file_only",
                message=f"Content '{title}' saved to disk but not indexed",
                path=str(file_path),
                chunks=0,
                indexed=False,
                location=location_info,  # type: ignore[typeddict-item]
                warning=f"Indexing pipeline error: {e}",
            )


@app.tool(  # type: ignore[arg-type]
    metadata=ToolMetadata(
        behavior=Behavior(
            operations=[Operation.UPDATE],
            read_only=False,
            destructive=False,
            idempotent=True,
            open_world=False,
        ),
    ),
)
async def update_library_doc(
    context: Context,
    path: Annotated[str, "Absolute path to the document to update"],
    content: Annotated[str, "The new content to replace the existing content"],
) -> Annotated[
    UpdateOutput,
    "Confirmation of update plus chunk count from the re-index step.",
]:
    """
    Update existing content in the agent's library.

    Use this to modify or replace content that was previously stored.
    The updated content will be re-indexed for search.
    """
    file_path = _resolve_path(path, "document path")

    if not file_path.exists():
        raise RetryableToolError(
            message=f"There's no document at {path}.",
            additional_prompt_content=(
                "Find the right path first:\n"
                "  - search_library — find a document by content\n"
                "  - list_library_contents — see every indexed document and its path"
            ),
        )

    try:
        file_path.write_text(content, encoding="utf-8")
    except OSError as e:
        raise ToolExecutionError(
            message=f"Couldn't write the new content to {file_path}.",
            developer_message=(
                "The file may be read-only, the disk may be full, or the volume "
                "may be unmounted. Tell the user what happened and stop — "
                "retrying won't help until the underlying issue is fixed."
            ),
        ) from e

    try:
        result = _process_and_index_file(file_path)
    except Exception as e:
        logger.exception("Reindex failed for %s", file_path)
        raise ToolExecutionError(
            message=f"Saved {file_path} but the search index didn't update.",
            developer_message=(
                "The file is on disk but won't show up in search yet. "
                "Call index_directory_to_library on the parent directory to "
                "rebuild the index. If that also fails, call get_library_overview(view='stats') "
                "to confirm the embedding service is reachable."
            ),
        ) from e

    return UpdateOutput(
        status="updated",
        message="Content has been updated in your library",
        path=str(file_path),
        title=result["title"],
        chunks=result["chunks"],
    )


# =============================================================================
# Unified Search Tool
# =============================================================================


@app.tool(  # type: ignore[arg-type]
    metadata=ToolMetadata(
        behavior=Behavior(
            operations=[Operation.READ],
            read_only=True,
            destructive=False,
            idempotent=True,
            open_world=False,
        ),
    ),
)
async def search_library(
    context: Context,
    query: Annotated[str, "What to search for in the library"],
    mode: Annotated[
        SearchMode,
        "Search method: hybrid (semantic + keyword, default), semantic (meaning-based), or keyword (exact match)",
    ] = SearchMode.HYBRID,
    asset_type: Annotated[
        AssetType | None,
        "Filter by content type: text, code, image, pdf. Omit to search all types.",
    ] = None,
    timeframe: Annotated[
        Timeframe | None,
        "Filter by time period: today, yesterday, this_week, last_week, this_month, last_month, last_7_days, last_30_days, this_year",
    ] = None,
    start_date: Annotated[
        str | None,
        "Start date for custom range (YYYY-MM-DD). Use instead of timeframe for precise ranges.",
    ] = None,
    end_date: Annotated[
        str | None,
        "End date for custom range (YYYY-MM-DD). Use with start_date.",
    ] = None,
    limit: Annotated[int, "Maximum number of results to return"] = 10,
    include_deleted: Annotated[
        bool,
        "Include soft-deleted (tombstoned) chunks in results. Off by default; "
        "turn on to surface content removed from its source but kept for history.",
    ] = False,
) -> Annotated[
    list[SearchHit],
    "Ranked list of matching chunks with score, snippet, and asset type.",
]:
    """
    Search the agent's library for relevant information.

    This is the primary way to find stored knowledge. Supports three search modes:
    - hybrid (default): Combines semantic understanding with keyword matching for best results
    - semantic: Finds content with similar meaning, even without exact word matches
    - keyword: Finds content containing the specific words in your query

    Use this when you need to:
    - Find previously stored information
    - Look up notes or documentation
    - Recall context from past conversations
    - Search code, PDFs, images, or text documents
    """
    if not query.strip():
        return []

    # MCP clients that omit a parameter can surface here as None even when
    # the Python signature declares a default; coerce explicitly so the
    # HYBRID branch below isn't entered with mode=None.
    if mode is None:
        mode = SearchMode.HYBRID
    # cross_modal was a deprecated alias for hybrid; accept the raw string
    # if it arrives from a stale client (pydantic will have rejected it at
    # the input layer, so this only catches direct-dispatch paths).
    if isinstance(mode, str) and mode == "cross_modal":
        mode = SearchMode.HYBRID
    if limit is None:
        limit = 10
    if include_deleted is None:
        include_deleted = False

    db = get_metadata_store()
    filter_doc_ids: list[int] | None = None

    # Apply timeframe filter if specified
    if timeframe:
        start_dt, end_dt = get_timeframe_bounds(timeframe)
        filter_doc_ids = db.get_document_ids_in_timerange(start_dt, end_dt)
        if not filter_doc_ids:
            return []
    elif start_date:
        # Custom date range
        parsed_start = parse_date_string(start_date)
        if not parsed_start:
            raise RetryableToolError(
                message=f"start_date={start_date!r} isn't a date I can parse.",
                additional_prompt_content=(
                    "Pass start_date as 'YYYY-MM-DD' (for example '2026-03-15'). "
                    "If the user said something like 'this week' or 'last 30 days', "
                    "use the `timeframe` parameter instead — it accepts 'today', "
                    "'yesterday', 'this_week', 'last_week', 'this_month', "
                    "'last_month', 'last_7_days', 'last_30_days', or 'this_year'."
                ),
            )

        parsed_end = parse_date_string(end_date) if end_date else None
        if end_date and not parsed_end:
            raise RetryableToolError(
                message=f"end_date={end_date!r} isn't a date I can parse.",
                additional_prompt_content=(
                    "Pass end_date as 'YYYY-MM-DD' (for example '2026-03-15'), "
                    "or simply leave it out to use 'now' as the range end."
                ),
            )

        if (
            parsed_end
            and parsed_end.hour == 0
            and parsed_end.minute == 0
            and parsed_end.second == 0
        ):
            parsed_end = parsed_end.replace(hour=23, minute=59, second=59, microsecond=999999)

        if not parsed_end:
            from datetime import datetime

            parsed_end = datetime.now()

        filter_doc_ids = db.get_document_ids_in_timerange(parsed_start, parsed_end)
        if not filter_doc_ids:
            return []

    # Build asset type filter
    asset_type_filter: list[AssetType] | None = [asset_type] if asset_type else None

    searcher = _get_searcher()

    # Execute search based on mode
    try:
        if mode == SearchMode.SEMANTIC:
            # Use modality-specific embeddings when available
            if asset_type == AssetType.CODE and ENABLE_CODE_EMBEDDINGS:
                results = searcher.vector_search_by_modality(
                    query, EmbeddingModality.CODE, limit=limit, include_deleted=include_deleted
                )
            elif asset_type == AssetType.IMAGE and ENABLE_VISION_EMBEDDINGS:
                results = searcher.vector_search_by_modality(
                    query, EmbeddingModality.VISION, limit=limit, include_deleted=include_deleted
                )
            else:
                results = searcher.vector_search(
                    query, limit=limit, include_deleted=include_deleted
                )
                # Filter by asset type if specified (vector_search doesn't support it natively)
                if asset_type_filter:
                    results = [r for r in results if r.asset_type in asset_type_filter][:limit]
        elif mode == SearchMode.KEYWORD:
            results = searcher.keyword_search(query, limit=limit, include_deleted=include_deleted)
            # Filter by asset type if specified
            if asset_type_filter:
                results = [r for r in results if r.asset_type in asset_type_filter][:limit]
        else:
            # HYBRID
            results = searcher.search(
                query,
                limit=limit,
                use_mmr=True,
                filter_document_ids=filter_doc_ids,
                asset_types=asset_type_filter,
                include_deleted=include_deleted,
            )
    except Exception as e:
        logger.exception("search_library failed (mode=%s)", mode.value)
        raise RetryableToolError(
            message=f"The {mode.value} search backend isn't responding.",
            additional_prompt_content=(
                "Try again with mode='keyword'. That uses plain full-text search "
                "and doesn't depend on the embedding service. "
                "If keyword mode also fails, call get_library_overview(view='stats') — the index "
                "may be empty (in which case ask the user to add content first)."
            ),
        ) from e

    # Apply document ID filter for non-hybrid modes (hybrid handles it internally)
    if filter_doc_ids and mode in (SearchMode.SEMANTIC, SearchMode.KEYWORD):
        filter_set = set(filter_doc_ids)
        results = [r for r in results if r.document_id in filter_set]

    # Enrich with the v0.14 public columns. SearchResult still carries the
    # integer surrogate (chunks.id); map it to the deterministic text chunk_id
    # and the additive fields. Legacy rows (pre-rebuild) fall back to the
    # surrogate id rendered as a string.
    enrichment = db.get_chunk_public_fields([r.chunk_id for r in results])

    hits: list[SearchHit] = []
    for r in results:
        extra = enrichment.get(r.chunk_id, {})
        text_chunk_id = extra.get("chunk_id") or str(r.chunk_id)
        hits.append(
            SearchHit(
                chunk_id=text_chunk_id,
                document_id=r.document_id,
                document_path=r.document_path,
                content=r.content,
                heading_path=r.heading_path,
                score=round(r.score, 4),
                snippet=r.snippet,
                asset_type=r.asset_type.value,
                chunk_source_uri=extra.get("chunk_source_uri"),
                chunk_index=extra.get("chunk_index"),
                document_size=extra.get("document_size"),
                source_created_at=extra.get("source_created_at"),
                deleted_at=extra.get("deleted_at"),
            )
        )
    return hits


# Upper bound on each side of an expand_context window. Guards against a caller
# asking for the whole document (thousands of chunks, full content each) in one
# MCP response; mirrors the negative-value clamp on the same params.
_EXPAND_CONTEXT_MAX = 50


def _clamp_window(value: int | None, default: int = 2) -> int:
    """Coerce a window size to ``[0, _EXPAND_CONTEXT_MAX]`` (None/negative -> default)."""
    if value is None or value < 0:
        return default
    return min(value, _EXPAND_CONTEXT_MAX)


@app.tool(  # type: ignore[arg-type]
    metadata=ToolMetadata(
        behavior=Behavior(
            operations=[Operation.READ],
            read_only=True,
            destructive=False,
            idempotent=True,
            open_world=False,
        ),
    ),
)
async def expand_context(
    context: Context,
    chunk_id: Annotated[
        str,
        "The chunk_id to expand around (the value returned by search_library).",
    ],
    before: Annotated[int, "Number of chunks to retrieve before the given chunk"] = 2,
    after: Annotated[int, "Number of chunks to retrieve after the given chunk"] = 2,
    include_deleted: Annotated[
        bool,
        "Include soft-deleted neighbor chunks. Off by default.",
    ] = False,
) -> Annotated[
    list[ContextChunk],
    "Neighboring chunks from the same document, in source order.",
]:
    """
    Retrieve the chunks surrounding a search hit, in source order.

    Search returns isolated chunks, so a fragmentary match (e.g. a reply that
    just says "yes, do it") can be hard to interpret on its own. Call this with
    that chunk's `chunk_id` to pull its neighbors from the same document and
    reconstruct the surrounding context.

    Returns up to `before + after` chunks (fewer near document boundaries, and
    at most 50 per side), ordered by their position in the document. The anchor
    chunk itself is not repeated — you already have it from search.
    """
    before = _clamp_window(before)
    after = _clamp_window(after)
    if include_deleted is None:
        include_deleted = False

    anchor_id = str(chunk_id).strip() if chunk_id else ""
    if not anchor_id:
        raise RetryableToolError(
            message="expand_context needs a chunk_id.",
            additional_prompt_content=(
                "Pass the `chunk_id` from a search_library result. Run "
                "search_library first, then call expand_context with the "
                "chunk_id of the hit you want more context around."
            ),
        )

    db = get_metadata_store()
    try:
        neighbors = db.get_chunk_context(
            anchor_id, before=before, after=after, include_deleted=include_deleted
        )
    except Exception as e:
        # Backend unreachable, or a serve-only process pointed at a database the
        # v0.14 write path never migrated (no chunks.chunk_id column yet).
        # Degrade like search_library instead of dead-ending on a non-retryable
        # error the agent can't act on.
        logger.exception("expand_context failed (chunk_id=%s)", chunk_id)
        raise RetryableToolError(
            message="Couldn't expand context right now.",
            additional_prompt_content=(
                "The library index may be empty or not built yet. Add content "
                "with index_directory_to_library first, then retry. To confirm "
                "the index exists, call get_library_overview(view='stats')."
            ),
        ) from e

    if neighbors:
        return [
            ContextChunk(
                chunk_id=n.chunk_id or str(n.internal_id),
                document_id=n.document_id,
                document_path=n.document_path,
                content=n.content,
                heading_path=n.heading_path,
                chunk_index=n.chunk_index,
                asset_type=n.asset_type,
                chunk_source_uri=n.chunk_source_uri,
                deleted_at=n.deleted_at,
            )
            for n in neighbors
        ]

    # Empty window: three distinct causes, each needs different guidance. The
    # anchor lookup ignores deletion, so an existing-but-tombstoned anchor gets
    # here with default include_deleted=False.
    if not include_deleted:
        # A soft delete tombstones every chunk of a document at once, so an agent
        # that found this anchor via include_deleted=True and dropped the flag
        # lands exactly here — steer it to the one thing that would work.
        tombstoned = db.get_chunk_context(
            anchor_id, before=before, after=after, include_deleted=True
        )
        if tombstoned:
            raise RetryableToolError(
                message=f"The neighbors of chunk_id={chunk_id!r} are soft-deleted.",
                additional_prompt_content=(
                    "Call expand_context again with include_deleted=True to include "
                    "tombstoned chunks (content removed from its source but kept "
                    "for history)."
                ),
            )

    if not db.chunk_exists(anchor_id):
        raise RetryableToolError(
            message=f"No chunk found for chunk_id={chunk_id!r}.",
            additional_prompt_content=(
                "Double-check the chunk_id came from a recent search_library "
                "result — expand_context takes the `chunk_id` field of a hit."
            ),
        )

    # The anchor exists but stands alone (single-chunk document): an empty list
    # is the honest answer, matching search_library's empty-result convention.
    return []


# =============================================================================
# Library Management Tools
# =============================================================================


@app.tool(  # type: ignore[arg-type]
    metadata=ToolMetadata(
        behavior=Behavior(
            operations=[Operation.READ],
            read_only=True,
            destructive=False,
            idempotent=True,
            open_world=False,
        ),
    ),
)
async def read_from_library(
    context: Context,
    path: Annotated[str, "Absolute path to the document to read"],
) -> Annotated[
    ReadOutput,
    "Full document content plus metadata. `indexed=False` if read directly from disk.",
]:
    """
    Read the full content of a document from the library.

    Use this after searching to get the complete content of a
    document, rather than just the matching snippets.
    """
    db = get_metadata_store()
    doc = db.get_document_by_path(path)

    if not doc:
        file_path = Path(path)
        if file_path.exists():
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError as e:
                raise RetryableToolError(
                    message=f"{path} is a binary file, not text.",
                    additional_prompt_content=(
                        "read_from_library only returns text. For PDFs, images, "
                        "or other binaries, use index_directory_to_library on the "
                        "parent directory first — that routes each file through "
                        "the right parser. Then search_library will surface its "
                        "content."
                    ),
                ) from e
            except (OSError, PermissionError) as e:
                raise ToolExecutionError(
                    message=f"The file at {path} exists but can't be opened.",
                    developer_message=(
                        "Likely causes: the volume is unmounted, the file is "
                        "locked, or permissions changed. Tell the user the path "
                        "is unreachable and ask them to verify it."
                    ),
                ) from e
            return ReadOutput(
                id=None,
                path=path,
                content=content,
                indexed=False,
                note="This file is on disk but not in the library index yet.",
            )
        raise RetryableToolError(
            message=f"Nothing in the library at {path}.",
            additional_prompt_content=(
                "Find the right path first:\n"
                "  - search_library — find a document by content keywords\n"
                "  - list_library_contents — see every indexed document and its path"
            ),
        )

    return ReadOutput(
        id=doc.id,
        path=doc.path,
        title=doc.title,
        content=doc.content,
        metadata=doc.metadata,
        created_at=str(doc.created_at) if doc.created_at else None,
        updated_at=str(doc.updated_at) if doc.updated_at else None,
        indexed=True,
    )


@app.tool(  # type: ignore[arg-type]
    metadata=ToolMetadata(
        behavior=Behavior(
            operations=[Operation.DELETE],
            read_only=False,
            destructive=True,
            idempotent=True,
            open_world=False,
        ),
    ),
)
async def remove_from_library(
    context: Context,
    path: Annotated[str, "Absolute path to the document to remove"],
    delete_file: Annotated[bool, "Also delete the file from disk (permanent)"] = False,
) -> Annotated[
    RemoveOutput,
    "Outcome of removal: index status and (if requested) on-disk delete result.",
]:
    """
    Remove a document from the agent's library.

    By default, this only removes from the search index (the file
    remains on disk). Set delete_file=True to permanently delete.
    """
    db = get_database()

    deleted = db.delete_document_by_path(path)

    result: RemoveOutput = {
        "path": path,
        "removed_from_index": deleted,
        "message": "Document removed from library" if deleted else "Document was not in library",
    }

    if delete_file:
        file_path = Path(path)
        if file_path.exists():
            try:
                file_path.unlink()
                result["file_deleted"] = True
                result["message"] = "Document permanently deleted"
            except (OSError, PermissionError) as e:
                # Index removal already happened; the disk delete is what failed.
                # Surface as an execution error so the LLM sees the partial state.
                raise ToolExecutionError(
                    message=(
                        f"The document was removed from the index, "
                        f"but the file at {path} couldn't be deleted from disk."
                    ),
                    developer_message=(
                        "Tell the user the document is gone from search but the "
                        "file is still on disk, and that they need to delete it "
                        "manually (or fix the permission issue and call this "
                        "tool again)."
                    ),
                ) from e
        else:
            result["file_deleted"] = False
            result["note"] = (
                f"Requested file delete but no file exists at {path}. "
                "It was already gone, or the path pointed only at an index entry."
            )

    return result


@app.tool(  # type: ignore[arg-type]
    metadata=ToolMetadata(
        behavior=Behavior(
            operations=[Operation.READ],
            read_only=True,
            destructive=False,
            idempotent=True,
            open_world=False,
        ),
    ),
)
async def list_library_contents(
    context: Context,
    limit: Annotated[int, "Maximum number of documents to list"] = 100,
) -> Annotated[
    list[DocumentSummary],
    "Summary list of indexed documents with id, path, title, and timestamps.",
]:
    """
    List all documents stored in the agent's library.

    Returns a summary of each document including title, path,
    and when it was added/updated.
    """
    db = get_metadata_store()
    documents = db.list_documents()[:limit]

    return [
        DocumentSummary(
            id=doc.id,
            path=doc.path,
            title=doc.title,
            metadata=doc.metadata,
            created_at=str(doc.created_at) if doc.created_at else None,
            updated_at=str(doc.updated_at) if doc.updated_at else None,
        )
        for doc in documents
    ]


# =============================================================================
# Optional Tools (enabled via LIBRARIAN_ENABLE_OPTIONAL_TOOLS=true)
# =============================================================================

_READ_ONLY_BEHAVIOR = Behavior(
    operations=[Operation.READ],
    read_only=True,
    destructive=False,
    idempotent=True,
    open_world=False,
)


if ENABLE_OPTIONAL_TOOLS:

    def _build_directory_doc_index(
        documents: list[Any],
    ) -> tuple[dict[str, list[Any]], dict[str, int]]:
        """Index documents by parent dir and by every prefix for O(1) lookups."""
        from collections import defaultdict

        docs_by_dir: dict[str, list[Any]] = defaultdict(list)
        for doc in documents:
            docs_by_dir[str(Path(doc.path).parent)].append(doc)

        count_by_prefix: dict[str, int] = defaultdict(int)
        for doc in documents:
            for parent in Path(doc.path).parents:
                count_by_prefix[str(parent)] += 1

        return dict(docs_by_dir), dict(count_by_prefix)

    def _view_sections(include_file_counts: bool) -> OverviewResult:
        """Build the SECTIONS view: top-level subdirs per source with doc counts."""
        sources = _get_sources_config()
        if not sources:
            return OverviewResult(
                view=LibraryView.SECTIONS.value,
                sections=[],
                total_sources=0,
                message="No sources registered. Use 'libr add <path>' to add a source.",
                default_path=DOCUMENTS_PATH,
            )

        db = get_metadata_store()
        docs_by_dir, count_by_prefix = _build_directory_doc_index(db.list_documents())

        blocks: list[OverviewSourceBlock] = []
        for source in sources:
            source_path = Path(source.get("path", ""))
            source_name = source.get("name", source_path.name)
            source_path_str = str(source_path)

            if not source_path.exists():
                blocks.append(
                    OverviewSourceBlock(
                        source=source_name,
                        available=False,
                        path=source_path_str,
                        error="Source path does not exist",
                    )
                )
                continue

            if source.get("is_file"):
                blocks.append(
                    OverviewSourceBlock(
                        source=source_name,
                        available=True,
                        path=str(source_path.parent),
                        type="file",
                        description=f"Single file: {source_path.name}",
                    )
                )
                continue

            root_section: OverviewSection = OverviewSection(
                name=f"{source_name} (root)",
                path=source_path_str,
                description="Root level of this source",
            )
            if include_file_counts:
                root_section["document_count"] = len(docs_by_dir.get(source_path_str, []))
            sub_sections: list[OverviewSection] = [root_section]

            try:
                for item in sorted(source_path.iterdir()):
                    if item.name.startswith(".") or not item.is_dir():
                        continue

                    item_str = str(item)
                    dir_doc_count = count_by_prefix.get(item_str, 0)
                    direct_docs = docs_by_dir.get(item_str, [])
                    sample_titles = [d.title for d in direct_docs[:3] if d.title]

                    description = f"Contains {dir_doc_count} documents"
                    if sample_titles:
                        description += f" (e.g., {', '.join(sample_titles[:2])})"

                    try:
                        subdirs = [
                            p.name
                            for p in item.iterdir()
                            if p.is_dir() and not p.name.startswith(".")
                        ]
                    except PermissionError:
                        subdirs = []

                    section: OverviewSection = OverviewSection(
                        name=item.name,
                        path=item_str,
                        description=description,
                    )
                    if include_file_counts:
                        section["document_count"] = dir_doc_count
                    if subdirs:
                        section["has_subdirectories"] = True
                        section["subdirectories"] = subdirs[:10]
                    sub_sections.append(section)
            except PermissionError:
                pass

            blocks.append(
                OverviewSourceBlock(
                    source=source_name,
                    available=True,
                    source_path=source_path_str,
                    type="directory",
                    sections=sub_sections,
                    total_documents=(
                        count_by_prefix.get(source_path_str, 0) if include_file_counts else None
                    ),
                )
            )

        return OverviewResult(
            view=LibraryView.SECTIONS.value,
            sections=blocks,
            total_sources=len(sources),
            usage_hint=(
                "Use the 'path' value from any section as the 'directory' "
                "parameter in add_to_library()"
            ),
        )

    def _view_stats() -> OverviewResult:
        """Build the STATS view: library totals + active configuration."""
        stats = get_metadata_store().get_stats()
        return OverviewResult(
            view=LibraryView.STATS.value,
            document_count=stats["document_count"],
            chunk_count=stats["chunk_count"],
            embedding_count=stats["embedding_count"],
            config=LibraryConfig(
                documents_path=DOCUMENTS_PATH,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                search_limit=SEARCH_LIMIT,
                mmr_lambda=MMR_LAMBDA,
                hybrid_alpha=HYBRID_ALPHA,
            ),
        )

    def _build_tree(path: Path, current_depth: int, max_depth: int) -> TreeNode:
        """Recursively build a directory tree, capped at max_depth."""
        if not path.exists():
            return TreeNode(
                name=path.name, type="directory", path=str(path), error="Path does not exist"
            )
        if path.is_file():
            return TreeNode(name=path.name, type="file", path=str(path))

        node: TreeNode = TreeNode(name=path.name, type="directory", path=str(path))

        if current_depth >= max_depth:
            try:
                items = list(path.iterdir())
                node["subdirectory_count"] = sum(
                    1 for p in items if p.is_dir() and not p.name.startswith(".")
                )
                node["file_count"] = sum(1 for p in items if p.is_file() and p.suffix == ".md")
            except PermissionError:
                node["error"] = "Permission denied"
            return node

        # children is typed list[dict] in TreeNode (to keep arcade's pydantic
        # schema generator from recursing on the self-reference); at runtime
        # each entry is a TreeNode-shaped dict.
        try:
            children: list[dict[str, Any]] = []
            for item in sorted(path.iterdir()):
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    children.append(dict(_build_tree(item, current_depth + 1, max_depth)))
                elif item.suffix == ".md":
                    children.append(dict(TreeNode(name=item.name, type="file", path=str(item))))
            node["children"] = children
        except PermissionError:
            node["error"] = "Permission denied"
        return node

    def _view_tree(depth: int) -> OverviewResult:
        """Build the TREE view: recursive filesystem walk per source."""
        sources = _get_sources_config()
        if not sources:
            return OverviewResult(
                view=LibraryView.TREE.value,
                sources=[],
                total_sources=0,
                message="No sources registered.",
            )

        source_blocks: list[TreeSourceBlock] = []
        for source in sources:
            source_path = Path(source.get("path", ""))
            source_name = source.get("name", source_path.name)
            if not source_path.exists():
                source_blocks.append(
                    TreeSourceBlock(
                        name=source_name,
                        path=str(source_path),
                        exists=False,
                        error="Source path does not exist",
                    )
                )
                continue
            source_blocks.append(
                TreeSourceBlock(
                    name=source_name,
                    path=str(source_path),
                    exists=True,
                    is_file=source.get("is_file", False),
                    structure=_build_tree(source_path, 0, depth),
                )
            )
        return OverviewResult(
            view=LibraryView.TREE.value,
            sources=source_blocks,
            total_sources=len(sources),
        )

    @app.tool(metadata=ToolMetadata(behavior=_READ_ONLY_BEHAVIOR))  # type: ignore[arg-type]
    async def get_library_overview(
        context: Context,
        view: Annotated[
            LibraryView,
            "Which slice of the library to return: 'sections' (top-level subdirs with "
            "doc counts and sample titles — best for choosing where to add content), "
            "'stats' (library totals and current config), or 'tree' (recursive "
            "filesystem walk, depth-controlled).",
        ] = LibraryView.SECTIONS,
        depth: Annotated[
            int,
            "How many directory levels to show. Only applies when view='tree'; "
            "ignored otherwise. 1 = top-level children only.",
        ] = 1,
    ) -> OverviewResult:
        """
        Inspect the library structure.

        THIS IS THE PRIMARY TOOL TO INSPECT THE LIBRARY BEFORE ADDING CONTENT.

        Three views are available:
        - 'sections' (default): top-level subdirectories of each source with
          document counts and sample titles. Each section's path is what you
          pass as `directory` to add_to_library(). Use this when you want to
          place new content somewhere sensible.
        - 'stats': aggregate document/chunk counts and the current chunking +
          search configuration. Use this for "how big is my library?".
        - 'tree': recursive filesystem tree per source, capped at `depth`.
          Use this when you need to browse organization in detail.

        Example workflow for adding content:
        1. Call get_library_overview() to see available locations
        2. Pick the path of the right section
        3. Call add_to_library(content=..., title=..., directory=<that path>)
        """
        if view is None:
            view = LibraryView.SECTIONS
        if view == LibraryView.STATS:
            return _view_stats()
        if view == LibraryView.TREE:
            return _view_tree(depth)
        return _view_sections(include_file_counts=True)

    @app.tool(metadata=ToolMetadata(behavior=_READ_ONLY_BEHAVIOR))  # type: ignore[arg-type]
    async def suggest_library_location(
        context: Context,
        title: Annotated[str, "The title of the content you want to add"],
        content_summary: Annotated[str, "A brief description of what the content is about"] = "",
    ) -> SuggestResult:
        """
        Get suggestions for where to store new content in the library.

        Analyzes the title and optional summary to suggest the best location(s)
        based on existing library organization and similar documents.

        Returns ranked suggestions with paths ready to use with add_to_library().
        """
        from librarian.tool_outputs import LocationSuggestion, SimilarDocument

        sources = _get_sources_config()
        if not sources:
            return SuggestResult(
                title=title,
                suggestions=[],
                message="No sources configured",
                default_path=DOCUMENTS_PATH,
            )

        search_query = f"{title} {content_summary}".strip()

        # Try hybrid search first; fall back through keyword to none.
        searcher = _get_searcher()
        search_method = "hybrid"
        try:
            similar_results = searcher.search(search_query, limit=10, use_mmr=True)
        except Exception as e:
            logger.warning("Hybrid search failed, falling back to keyword search: %s", e)
            try:
                similar_results = searcher.keyword_search(search_query, limit=10)
                search_method = "keyword"
            except Exception:
                similar_results = []
                search_method = "none"

        # Aggregate by parent directory.
        location_scores: dict[str, dict[str, Any]] = {}
        for result in similar_results:
            doc_path = Path(result.document_path)
            parent_dir = str(doc_path.parent)
            if parent_dir not in location_scores:
                source = _find_source_for_path(doc_path, sources)
                location_scores[parent_dir] = {
                    "path": parent_dir,
                    "source": source.get("name") if source else "Unknown",
                    "score": 0.0,
                    "similar_docs": [],
                    "breadcrumb": _build_breadcrumb(doc_path.parent, source),
                }
            location_scores[parent_dir]["score"] += result.score
            if len(location_scores[parent_dir]["similar_docs"]) < 3:
                location_scores[parent_dir]["similar_docs"].append(
                    SimilarDocument(
                        title=Path(result.document_path).stem,
                        relevance=round(result.score, 3),
                    )
                )

        sorted_locations = sorted(location_scores.values(), key=lambda x: x["score"], reverse=True)
        max_score = sorted_locations[0]["score"] if sorted_locations else 1.0

        suggestions: list[LocationSuggestion] = [
            LocationSuggestion(
                path=loc["path"],
                source=loc["source"],
                confidence=round(loc["score"] / max_score if max_score > 0 else 0.0, 2),
                breadcrumb_display=" → ".join(loc["breadcrumb"]),
                reason=f"Similar to: {', '.join(d['title'] for d in loc['similar_docs'])}",
                similar_documents=loc["similar_docs"],
            )
            for loc in sorted_locations[:5]
        ]

        # Fall back to source roots if no semantic matches.
        if not suggestions:
            for source in sources:
                if source.get("is_file"):
                    continue
                source_path = source.get("path", "")
                if Path(source_path).exists():
                    suggestions.append(
                        LocationSuggestion(
                            path=source_path,
                            source=source.get("name", source_path),
                            confidence=0.5,
                            breadcrumb_display=source.get("name", source_path),
                            reason="No similar content found - suggesting source roots",
                        )
                    )

        return SuggestResult(
            title=title,
            suggestions=suggestions,
            search_method=search_method,
            usage_hint=(
                "Use the suggested 'path' as the 'directory' parameter in add_to_library()"
            ),
        )


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    transport_arg = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport_arg not in ("http", "stdio"):
        transport_arg = "stdio"

    # Support --port and --host flags
    port = SERVER_PORT
    host = SERVER_HOST
    import contextlib

    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            with contextlib.suppress(ValueError):
                port = int(sys.argv[i + 1])
        elif arg == "--host" and i + 1 < len(sys.argv):
            host = sys.argv[i + 1]

    app.run(transport=transport_arg, host=host, port=port)  # type: ignore[arg-type]
