#!/usr/bin/env python3
"""
Librarian CLI - Context Management Service.

A command-line interface for managing, indexing, and searching
documents with vector and full-text search capabilities.

Usage:
    libr --help
    libr add <path>          # Add file or directory as source
    libr rm <source>         # Remove source and its documents
    libr list                # Show sources
    libr search "query"      # Search documents
    libr index               # Show index stats
    libr index build         # Rebuild index
    libr docs                # Show sources with doc counts
    libr docs list           # List indexed documents
"""

import asyncio
import fnmatch
import json
import os
import subprocess
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Optional

# Suppress verbose logging BEFORE any librarian imports
os.environ.setdefault("LOGURU_LEVEL", "ERROR")

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from librarian.sources.ignore import (
    GitignoreMatcher,
    LibrarianTrackMatcher,
    normalize_force_include,
    should_skip_file,
)

# Initialize Typer app
app = typer.Typer(
    name="libr",
    help="Librarian - Context Management Service",
    add_completion=True,
    rich_markup_mode="rich",
    invoke_without_command=True,
    no_args_is_help=True,
    pretty_exceptions_short=True,
)

# Sub-commands
index_app = typer.Typer(help="Index management", invoke_without_command=True)
docs_app = typer.Typer(help="Document operations", invoke_without_command=True)
config_app = typer.Typer(help="Configuration", invoke_without_command=True)

app.add_typer(index_app, name="index")
app.add_typer(docs_app, name="docs")
app.add_typer(config_app, name="config")

console = Console()

# Command groups exempt from the startup schema guard: the `index` group owns the
# rebuild/clean/clobber remedies (and guards its own stats view), while `version`
# and `config` don't touch the library database.
_SCHEMA_GUARD_EXEMPT = {None, "index", "version", "config"}


@app.callback()
def _main(ctx: typer.Context) -> None:
    """Top-level entry: refuse to run normal commands on a pre-v0.14 database."""
    if ctx.invoked_subcommand in _SCHEMA_GUARD_EXEMPT:
        return
    _guard_schema_or_exit()


# Config file path
CONFIG_DIR = Path.home() / ".librarian"
SOURCES_FILE = CONFIG_DIR / "sources.json"
SETTINGS_FILE = CONFIG_DIR / "settings.json"


def _apply_settings_to_env() -> None:
    """Load settings.json and apply as environment variables.

    This must run BEFORE any librarian modules are imported so that
    config.py picks up the values via os.getenv().

    Priority: explicit env vars > settings.json > defaults
    """
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                for key, value in json.load(f).items():
                    if key not in os.environ:
                        os.environ[key] = (
                            str(value).lower() if isinstance(value, bool) else str(value)
                        )
        except (json.JSONDecodeError, OSError):
            pass


# Apply settings before any librarian imports
_apply_settings_to_env()


# =============================================================================
# Enums
# =============================================================================


class OutputFormat(str, Enum):
    """Output format options."""

    TABLE = "table"
    JSON = "json"
    PATHS = "paths"


class SearchMode(str, Enum):
    """Search mode options. 'vector' is accepted as an alias for 'semantic'
    to match the MCP tool's advertised modes without breaking existing CLI
    callers that pass --mode vector."""

    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    VECTOR = "vector"
    KEYWORD = "keyword"


class Timeframe(str, Enum):
    """Time-based filter options."""

    TODAY = "today"
    YESTERDAY = "yesterday"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


# =============================================================================
# Helpers
# =============================================================================


def _load_sources() -> list[dict[str, Any]]:
    """Load registered sources from config."""
    if not SOURCES_FILE.exists():
        return []
    with open(SOURCES_FILE) as f:
        result: list[dict[str, Any]] = json.load(f)
        return result


def _save_sources(sources: list[dict[str, Any]]) -> None:
    """Save sources to config."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(SOURCES_FILE, "w") as f:
        json.dump(sources, f, indent=2)


def _load_settings() -> dict[str, Any]:
    """Load user settings from config."""
    if not SETTINGS_FILE.exists():
        return {}
    with open(SETTINGS_FILE) as f:
        return json.load(f)


def _save_settings(settings: dict[str, Any]) -> None:
    """Save user settings to config."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


def _run_async(coro: Any) -> Any:
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _get_config() -> dict[str, Any]:
    """Lazy import of config to avoid early logging."""
    from librarian.config import (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        CODE_EMBEDDING_DIMENSION,
        CODE_EMBEDDING_MODEL,
        CODE_EMBEDDING_PROVIDER,
        DATABASE_PATH,
        DOCUMENTS_PATH,
        EMBEDDING_MODEL,
        EMBEDDING_PROVIDER,
        ENABLE_CODE_EMBEDDINGS,
        ENABLE_OCR,
        ENABLE_VISION_EMBEDDINGS,
        HYBRID_ALPHA,
        MMR_LAMBDA,
        OCR_LANGUAGE,
        SEARCH_LIMIT,
        VISION_EMBEDDING_DIMENSION,
        VISION_EMBEDDING_MODEL,
        ensure_directories,
    )

    return {
        "DOCUMENTS_PATH": DOCUMENTS_PATH,
        "DATABASE_PATH": DATABASE_PATH,
        "EMBEDDING_PROVIDER": EMBEDDING_PROVIDER,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "ENABLE_OCR": ENABLE_OCR,
        "OCR_LANGUAGE": OCR_LANGUAGE,
        "CHUNK_SIZE": CHUNK_SIZE,
        "CHUNK_OVERLAP": CHUNK_OVERLAP,
        "SEARCH_LIMIT": SEARCH_LIMIT,
        "MMR_LAMBDA": MMR_LAMBDA,
        "HYBRID_ALPHA": HYBRID_ALPHA,
        "ENABLE_CODE_EMBEDDINGS": ENABLE_CODE_EMBEDDINGS,
        "CODE_EMBEDDING_MODEL": CODE_EMBEDDING_MODEL,
        "CODE_EMBEDDING_DIMENSION": CODE_EMBEDDING_DIMENSION,
        "CODE_EMBEDDING_PROVIDER": CODE_EMBEDDING_PROVIDER,
        "ENABLE_VISION_EMBEDDINGS": ENABLE_VISION_EMBEDDINGS,
        "VISION_EMBEDDING_MODEL": VISION_EMBEDDING_MODEL,
        "VISION_EMBEDDING_DIMENSION": VISION_EMBEDDING_DIMENSION,
        "ensure_directories": ensure_directories,
    }


def _should_skip_file(
    file_path: Path,
    supported_extensions: set[str],
    gitignore_matcher: "GitignoreMatcher | None" = None,
    force_include: "frozenset[Path] | None" = None,
    track_matcher: "LibrarianTrackMatcher | None" = None,
) -> bool:
    """Check if a file should be skipped during indexing."""
    return should_skip_file(
        file_path,
        supported_extensions,
        gitignore_matcher,
        force_include=force_include,
        track_matcher=track_matcher,
    )


def _find_source(name_or_path: str) -> dict | None:
    """Find a source by name or path."""
    sources = _load_sources()
    for s in sources:
        if s.get("name") == name_or_path or s["path"] == name_or_path:
            return s
    return None


def _get_source_doc_count(source: dict) -> int:
    """Get count of indexed docs for a source."""
    cfg = _get_config()
    cfg["ensure_directories"]()
    from librarian.storage.factory import get_metadata_store

    db = get_metadata_store()
    documents = db.list_documents()
    source_path = source["path"]
    return sum(1 for d in documents if d.path.startswith(source_path))


def _filter_display_sources(
    sources: list[dict[str, Any]], include_all: bool = False
) -> list[dict[str, Any]]:
    """Filter out test data sources from display unless --all is specified."""
    if include_all:
        return sources
    return [s for s in sources if "tests/data" not in s.get("path", "")]


def _index_path(file_path: Path, verbose: bool = False) -> dict[str, Any]:
    """Index a single file and return result."""
    from librarian.server import _process_and_index_file

    result = _process_and_index_file(file_path)
    if verbose:
        status = result.get("status", "")
        if status == "created":
            rprint(f"  [green]+[/green] {file_path}")
        elif status == "updated":
            rprint(f"  [yellow]~[/yellow] {file_path}")
    return result


def _guard_schema_or_exit() -> None:
    """Refuse to run against a pre-v0.14 database; point the user at the rebuild.

    Detection is cheap (a plain sqlite connection, no sqlite-vec load) so this is
    safe to call at the start of every command that touches the library.
    """
    from librarian.storage.schema_version import (
        REBUILD_MESSAGE,
        schema_status_for_path,
    )

    db_path = _get_config().get("DATABASE_PATH")
    if not db_path:
        return
    if schema_status_for_path(db_path) == "v0.13":
        rprint(
            Panel(
                REBUILD_MESSAGE,
                title="[red]Database upgrade required[/red]",
                border_style="red",
            )
        )
        raise typer.Exit(1)


def _reindex_sources(sources: list[dict[str, Any]], verbose: bool = False) -> tuple[int, int]:
    """Re-ingest the given sources via the Orchestrator-backed ingest path.

    Returns ``(total_indexed, total_errors)``.
    """
    from librarian.server import index_directory_to_library as server_ingest

    total_indexed = 0
    total_errors = 0
    for src in sources:
        src_path = Path(src["path"])
        if not src_path.exists():
            rprint(f"[yellow]Skipping missing:[/yellow] {src['name']}")
            continue

        rprint(f"  {src['name']}...")

        if src.get("is_file"):
            try:
                _index_path(src_path, verbose)
                total_indexed += 1
            except Exception as e:
                rprint(f"  [red]Error:[/red] {e}")
                total_errors += 1
        else:
            result = _run_async(
                server_ingest(
                    context=None,  # type: ignore[arg-type]
                    directory=str(src_path),
                    include_ignored=bool(src.get("include_ignored", False)),
                    force_include=list(src.get("force_include") or []) or None,
                )
            )
            total_indexed += result.get("indexed", 0) + result.get("updated", 0)
            total_errors += len(result.get("errors", []))

            if verbose:
                for file_info in result.get("files", []):
                    if file_info.get("status") in ("created", "updated"):
                        rprint(f"    [green]+[/green] {file_info.get('path', '')}")

    return total_indexed, total_errors


def _index_registered_source(
    source: dict[str, Any], source_path: Path, verbose: bool = False
) -> None:
    """Index an already-registered source without modifying the source registry."""
    from librarian.server import index_directory_to_library as server_ingest

    if source.get("is_file", source_path.is_file()):
        rprint("[cyan]Indexing file...[/cyan]")
        try:
            result = _index_path(source_path, verbose)
        except Exception as e:
            rprint(f"[red]Error indexing:[/red] {e}")
            raise typer.Exit(1) from None

        rprint(
            Panel(
                f"[green]Source indexed![/green]\n\n"
                f"Name: [cyan]{source.get('name', source_path.name)}[/cyan]\n"
                f"Path: [blue]{source_path}[/blue]\n"
                f"Chunks: [yellow]{result.get('chunks', 0)}[/yellow]",
                title="Source Indexed",
            )
        )
        return

    rprint("[cyan]Indexing directory...[/cyan]")
    result = _run_async(
        server_ingest(
            context=None,  # type: ignore[arg-type]
            directory=str(source_path),
            include_ignored=bool(source.get("include_ignored", False)),
            force_include=list(source.get("force_include") or []) or None,
        )
    )

    errors = result.get("errors", [])
    for err in errors:
        rprint(f"  [red]Error:[/red] {err.get('path', '')}: {err.get('error', '')}")

    skipped = result.get("skipped", 0)
    summary = (
        f"[green]Source indexed![/green]\n\n"
        f"Name: [cyan]{source.get('name', source_path.name)}[/cyan]\n"
        f"Path: [blue]{source_path}[/blue]\n"
        f"Files found: [yellow]{result.get('total_files', 0)}[/yellow]\n"
        f"Indexed: [cyan]{result.get('indexed', 0)}[/cyan]\n"
        f"Updated: [cyan]{result.get('updated', 0)}[/cyan]"
    )
    if skipped:
        summary += f"\nSkipped: [dim]{skipped}[/dim]"
    if errors:
        summary += f"\nErrors: [red]{len(errors)}[/red]"

    rprint(Panel(summary, title="Source Indexed"))
    if errors:
        raise typer.Exit(1)


def _confirm_rebuild(db_path: Path, *, no_backup: bool, confirm: bool) -> None:
    """Gate the destructive rebuild behind an explicit confirmation.

    Raises ``typer.Exit`` if the user declines. ``--yes`` skips the prompt.
    ``--no-backup`` escalates to a typed confirmation because it is an
    irreversible, unrecoverable wipe.
    """
    if confirm:
        return
    if no_backup:
        rprint(
            Panel(
                f"[red]This will permanently delete[/red] [blue]{db_path}[/blue] "
                "with [red]no backup[/red].\n\n"
                "[red]This cannot be undone.[/red]",
                title="[red]Irreversible wipe (--no-backup)[/red]",
                border_style="red",
            )
        )
        typed = typer.prompt("Type 'rebuild' to confirm", default="", show_default=False)
        if typed.strip().lower() != "rebuild":
            rprint("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)
        return
    rprint(
        Panel(
            f"[yellow]This will wipe and recreate[/yellow] [blue]{db_path}[/blue] "
            "under the v0.14 schema, then re-ingest configured sources.\n\n"
            "A backup of the current database is taken first.",
            title="Rebuild database",
        )
    )
    if not typer.confirm("Continue?"):
        rprint("[yellow]Cancelled.[/yellow]")
        raise typer.Exit(0)


def _rebuild_v014(*, no_backup: bool = False, verbose: bool = False, confirm: bool = False) -> None:
    """Interactive v0.13 -> v0.14 rebuild remedy.

    Confirms with the user, backs up the existing database (unless ``no_backup``),
    wipes and recreates it under the current schema, then re-ingests every
    configured source. The backup/wipe mechanics live in
    :mod:`librarian.storage.rebuild`; this function only owns the prompts and the
    source re-ingestion.
    """
    from librarian.storage import rebuild

    cfg = _get_config()
    cfg["ensure_directories"]()
    db_path = Path(cfg["DATABASE_PATH"])

    _confirm_rebuild(db_path, no_backup=no_backup, confirm=confirm)

    if db_path.exists() and not no_backup:
        # backup_database raises before any wipe, so a failed backup never loses data.
        backup = rebuild.backup_database(db_path)
        rprint(f"[green]Backed up[/green] database to [blue]{backup}[/blue]")
    elif db_path.exists():
        rprint("[yellow]Skipping backup[/yellow] (--no-backup).")

    rebuild.recreate_empty(db_path)
    rprint("[cyan]Recreated the database under the v0.14 schema.[/cyan]")

    sources = _load_sources()
    if not sources:
        rprint("[dim]No sources registered; nothing to re-ingest.[/dim]")
        rprint("[green]Rebuild complete.[/green]")
        return

    rprint("[cyan]Re-ingesting sources...[/cyan]")
    indexed, errors = _reindex_sources(sources, verbose)
    rprint(
        Panel(
            f"[green]Rebuild complete![/green]\n\n"
            f"Indexed: [cyan]{indexed}[/cyan]\n"
            f"Errors: [red]{errors}[/red]",
            title="Rebuild Results",
        )
    )


def _get_editor() -> str:
    """Get the user's preferred editor."""
    return os.environ.get("EDITOR", os.environ.get("VISUAL", "vim"))


def _open_in_editor(path: str) -> None:
    """Open a file in the user's editor."""
    editor = _get_editor()
    subprocess.run([editor, path], check=False)


def _copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard. Returns True on success."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
        elif sys.platform == "linux":
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode(),
                check=True,
            )
        elif sys.platform == "win32":
            subprocess.run(["clip"], input=text.encode(), check=True)
        else:
            return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True


def _matches_patterns(path: str, patterns: list[str]) -> bool:
    """Check if path matches any of the glob patterns."""
    for pattern in patterns:
        # Match against full path and filename
        if fnmatch.fnmatch(path, f"*{pattern}*"):
            return True
        if fnmatch.fnmatch(path, f"*/{pattern}"):
            return True
        if fnmatch.fnmatch(Path(path).name, pattern):
            return True
    return False


def _get_timeframe_bounds(timeframe: Timeframe) -> tuple[datetime, datetime]:
    """Get start and end datetime for a timeframe."""
    from librarian.utils.timeframe import Timeframe as TF
    from librarian.utils.timeframe import get_timeframe_bounds

    tf_map = {
        Timeframe.TODAY: TF.TODAY,
        Timeframe.YESTERDAY: TF.YESTERDAY,
        Timeframe.WEEK: TF.THIS_WEEK,
        Timeframe.MONTH: TF.THIS_MONTH,
        Timeframe.YEAR: TF.THIS_YEAR,
    }
    return get_timeframe_bounds(tf_map[timeframe])


# =============================================================================
# libr list - Show sources
# =============================================================================


@app.command("list")
def list_sources(
    all_sources: Annotated[
        bool, typer.Option("--all", "-a", help="Include hidden/test sources")
    ] = False,
    output_json: Annotated[bool, typer.Option("--json", help="Output as JSON")] = False,
) -> None:
    """List all registered document sources."""
    sources = _filter_display_sources(_load_sources(), include_all=all_sources)

    if output_json:
        print(json.dumps(sources, indent=2, default=str))
        return

    if not sources:
        rprint(
            Panel(
                "[yellow]No sources registered yet.[/yellow]\n\n"
                "Add a source with: [cyan]libr add <path>[/cyan]",
                title="Document Sources",
            )
        )
        return

    table = Table(title="Document Sources", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green")
    table.add_column("Path", style="blue", overflow="fold")
    table.add_column("Type", style="magenta")
    table.add_column("Status", style="yellow")

    for source in sources:
        path = Path(source["path"])
        status = "Active" if path.exists() else "Missing"
        src_type = "file" if source.get("is_file") else "dir"
        table.add_row(source.get("name", path.name), str(path), src_type, status)

    console.print(table)

    # Check for duplicate names and warn
    from collections import Counter

    name_counts = Counter(s.get("name") for s in sources)
    duplicates = {name: count for name, count in name_counts.items() if count > 1}

    if duplicates:
        rprint()
        rprint("[yellow]Warning:[/yellow] Sources with duplicate names detected:")
        for name, count in duplicates.items():
            rprint(f"  '{name}' appears [bold]{count}[/bold] times")
            paths = [s["path"] for s in sources if s.get("name") == name]
            for p in paths:
                rprint(f"    - {p}")
        rprint()
        rprint("To remove a specific source, use:")
        rprint("  [cyan]libr rm <name> --path <full-path>[/cyan]")
        rprint("  [cyan]libr rm <full-path>[/cyan]")


# =============================================================================
# libr add - Add source (file or directory)
# =============================================================================


@app.command("add")
def add_source(
    path: Annotated[str, typer.Argument(help="Path to file or directory to add")],
    name: Annotated[
        Optional[str], typer.Option("--name", "-n", help="Custom name for the source")
    ] = None,
    depth: Annotated[
        int, typer.Option("--depth", "-d", help="Limit recursion depth (0=current dir only)")
    ] = -1,
    pattern: Annotated[
        Optional[str],
        typer.Option("--pattern", "-p", help="Glob pattern to include (e.g., 'notes/*.md')"),
    ] = None,
    exclude: Annotated[
        Optional[list[str]],
        typer.Option("--exclude", "-e", help="Patterns to exclude (can repeat)"),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be indexed without doing it")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show files being indexed")
    ] = False,
    include_ignored: Annotated[
        bool,
        typer.Option(
            "--include-ignored",
            help="Index files even when matched by a .gitignore in the source tree",
        ),
    ] = False,
    force_include: Annotated[
        Optional[list[str]],
        typer.Option(
            "--force-include",
            help=(
                "Path to always index, even when matched by .gitignore or by the "
                "skip-dirs baseline (node_modules, __pycache__, .venv, etc.). "
                "Pointing at a directory force-includes everything beneath it. "
                "Can be repeated. A .librariantrack file inside the source has "
                "the same effect for patterns it lists."
            ),
        ),
    ] = None,
) -> None:
    """Add a file or directory as a source and index it recursively."""
    cfg = _get_config()
    cfg["ensure_directories"]()

    source_path = Path(path).resolve()

    if not source_path.exists():
        rprint(f"[red]Error:[/red] Path does not exist: {source_path}")
        raise typer.Exit(1)

    sources = _load_sources()

    # Check if already exists
    for existing in sources:
        if Path(existing["path"]).resolve() == source_path:
            rprint(f"[yellow]Source already registered:[/yellow] {source_path}")
            if dry_run:
                rprint("[dim]Dry run: would index the existing source.[/dim]")
                return
            _index_registered_source(existing, source_path, verbose)
            return

    is_file = source_path.is_file()

    # Get supported extensions from parser registry
    from librarian.processing.parsers.registry import get_registry

    registry = get_registry()
    supported_extensions = registry.get_supported_extensions()

    # Validate file type
    if is_file and source_path.suffix.lower() not in supported_extensions:
        rprint(f"[red]Error:[/red] Unsupported file type: {source_path.suffix}")
        rprint(f"Supported: {', '.join(sorted(supported_extensions))}")
        raise typer.Exit(1)

    # Find files to index
    if is_file:
        files_to_index = [source_path]
    else:
        # Find all files with supported extensions
        files_to_index = []
        for ext in supported_extensions:
            if depth == 0:
                files_to_index.extend(source_path.glob(f"*{ext}"))
            else:
                files_to_index.extend(source_path.rglob(f"*{ext}"))

        gitignore_matcher = None if include_ignored else GitignoreMatcher(source_path)
        track_matcher = LibrarianTrackMatcher(source_path)
        forced_paths = normalize_force_include(force_include)

        # Filter out system/binary files and .gitignore matches
        files_to_index = [
            f
            for f in files_to_index
            if not _should_skip_file(
                f,
                supported_extensions,
                gitignore_matcher,
                force_include=forced_paths,
                track_matcher=track_matcher,
            )
        ]

        # Apply pattern filter
        if pattern:
            files_to_index = [f for f in files_to_index if fnmatch.fnmatch(str(f), f"*{pattern}")]

        # Apply exclusions
        if exclude:
            files_to_index = [f for f in files_to_index if not _matches_patterns(str(f), exclude)]

    # Dry run - just show what would be indexed
    if dry_run:
        rprint(f"\n[bold]Dry run[/bold] - would index {len(files_to_index)} files:\n")
        for f in files_to_index[:50]:
            rprint(f"  [green]+[/green] {f}")
        if len(files_to_index) > 50:
            rprint(f"  [dim]... and {len(files_to_index) - 50} more[/dim]")
        return

    # Create source entry
    source_name = name or source_path.name

    # Warn if this name already exists
    existing_names = [s.get("name") for s in sources if s.get("name") == source_name]
    if existing_names:
        rprint(f"[yellow]Warning:[/yellow] A source named '{source_name}' already exists.")
        rprint(
            "  This will create duplicate names. Consider using --name to specify a unique name:"
        )
        rprint(f"  [cyan]libr add {path} --name {source_name}-2[/cyan]")
        rprint()
        if not typer.confirm("Continue anyway?", default=False):
            rprint("[yellow]Cancelled.[/yellow]")
            return

    source = {
        "name": source_name,
        "path": str(source_path),
        "type": "local",
        "is_file": is_file,
        "recursive": depth != 0,
        "depth": depth,
        "pattern": pattern,
        "exclude": exclude,
        "include_ignored": include_ignored,
        "force_include": list(force_include) if force_include else [],
        "added_at": datetime.now().isoformat(),
    }

    sources.append(source)
    _save_sources(sources)

    # Index the source
    from librarian.server import index_directory_to_library as server_ingest

    if is_file:
        rprint("[cyan]Indexing file...[/cyan]")
        try:
            result = _index_path(source_path, verbose)
            rprint(
                Panel(
                    f"[green]Source added and indexed![/green]\n\n"
                    f"Name: [cyan]{source['name']}[/cyan]\n"
                    f"Path: [blue]{source_path}[/blue]\n"
                    f"Chunks: [yellow]{result.get('chunks', 0)}[/yellow]",
                    title="Source Added",
                )
            )
        except Exception as e:
            rprint(f"[red]Error indexing:[/red] {e}")
            raise typer.Exit(1) from None
    else:
        rprint("[cyan]Indexing directory...[/cyan]")
        result = _run_async(
            server_ingest(
                context=None,  # type: ignore[arg-type]
                directory=str(source_path),
                include_ignored=include_ignored,
                force_include=list(force_include) if force_include else None,
            )
        )

        skipped_deps = []
        for file_info in result.get("files", []):
            fpath = file_info.get("path", "")
            status = file_info.get("status", "")
            reason = file_info.get("reason", "")
            if verbose:
                if status == "created":
                    rprint(f"  [green]+[/green] {fpath}")
                elif status == "updated":
                    rprint(f"  [yellow]~[/yellow] {fpath}")
                elif status == "skipped" and reason:
                    rprint(f"  [dim]-[/dim] {Path(fpath).name} [dim](skipped)[/dim]")
            if status == "skipped" and reason and "Install with" in reason:
                skipped_deps.append(reason)

        # Show unique dependency warnings
        seen_warnings: set[str] = set()
        for reason in skipped_deps:
            # Extract the first line (the main message) for deduplication
            key = reason.split("\n")[0] if "\n" in reason else reason
            if key not in seen_warnings:
                seen_warnings.add(key)
                rprint(f"  [yellow]Note:[/yellow] {key}")

        errors = result.get("errors", [])
        if errors:
            for err in errors:
                rprint(f"  [red]Error:[/red] {err.get('path', '')}: {err.get('error', '')}")

        skipped = result.get("skipped", 0)
        status_line = (
            "[red]Source added, but indexing had errors.[/red]"
            if errors
            else "[green]Source added and indexed![/green]"
        )
        summary = (
            f"{status_line}\n\n"
            f"Name: [cyan]{source['name']}[/cyan]\n"
            f"Path: [blue]{source_path}[/blue]\n"
            f"Files found: [yellow]{len(files_to_index)}[/yellow]\n"
            f"Indexed: [cyan]{result.get('indexed', 0)}[/cyan]"
        )
        if skipped:
            summary += f"\nSkipped: [dim]{skipped}[/dim]"
        if errors:
            summary += f"\nErrors: [red]{len(errors)}[/red]"

        rprint(Panel(summary, title="Source Added"))
        if errors:
            raise typer.Exit(1)


# =============================================================================
# libr rm - Remove source
# =============================================================================


@app.command("rm")
def remove_source(
    name: Annotated[str, typer.Argument(help="Name or path of the source to remove")],
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation")] = False,
    path: Annotated[
        Optional[str],
        typer.Option("--path", help="Specify path if multiple sources share the same name"),
    ] = None,
) -> None:
    """Remove a source and its documents from the index."""
    cfg = _get_config()
    cfg["ensure_directories"]()

    sources = _load_sources()
    to_remove = None

    # First try to match by exact path (most specific)
    for source in sources:
        if source.get("path") == name or source.get("path") == str(Path(name).resolve()):
            to_remove = source
            break

    # If not found by path, try by name
    if not to_remove:
        matches = [s for s in sources if s.get("name") == name]

        if len(matches) == 0:
            rprint(f"[red]Error:[/red] Source not found: {name}")
            rprint("\nAvailable sources:")
            for s in sources:
                rprint(f"  - {s.get('name')} ({s.get('path')})")
            raise typer.Exit(1)
        elif len(matches) == 1:
            to_remove = matches[0]
        else:
            # Multiple sources with same name
            if path:
                # User specified path to disambiguate
                to_remove = next((s for s in matches if s.get("path") == path), None)
                if not to_remove:
                    rprint(f"[red]Error:[/red] No source named '{name}' at path: {path}")
                    raise typer.Exit(1)
            else:
                # Ambiguous - show options
                rprint(f"[red]Error:[/red] Multiple sources named '{name}' found:")
                rprint()
                for i, s in enumerate(matches, 1):
                    rprint(f"  [{i}] {s.get('path')}")
                rprint()
                rprint("Please specify which one using --path:")
                rprint(f"  libr rm {name} --path <path>")
                rprint()
                rprint("Or remove by full path:")
                rprint("  libr rm <full-path>")
                raise typer.Exit(1)

    if not to_remove:
        rprint(f"[red]Error:[/red] Source not found: {name}")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Remove source '{name}' and all its documents from the index?")
        if not confirm:
            rprint("[yellow]Cancelled.[/yellow]")
            return

    # Remove documents from index through the storage factory so removal honors
    # STORAGE_BACKEND (on Postgres this must delete from Postgres, not a stray
    # SQLite file). delete_document_by_path is the backend-agnostic hard delete.
    from librarian.storage.factory import get_metadata_store, get_storage

    source_path = to_remove["path"]
    storage = get_storage()
    removed_count = 0

    for doc in get_metadata_store().list_documents():
        if doc.path.startswith(source_path) and storage.delete_document_by_path(doc.path):
            removed_count += 1

    # Remove from sources list
    sources.remove(to_remove)
    _save_sources(sources)

    rprint(
        Panel(
            f"[green]Source removed![/green]\n\n"
            f"Name: [cyan]{name}[/cyan]\n"
            f"Documents removed: [yellow]{removed_count}[/yellow]",
            title="Source Removed",
        )
    )


# =============================================================================
# libr index - Index management
# =============================================================================


@index_app.callback(invoke_without_command=True)
def index_stats(
    ctx: typer.Context,
    output_json: Annotated[bool, typer.Option("--json", help="Output as JSON")] = False,
    rebuild: Annotated[
        bool,
        typer.Option(
            "--rebuild",
            help="Back up, wipe, and recreate the index under the v0.14 schema, "
            "then re-ingest configured sources.",
        ),
    ] = False,
    no_backup: Annotated[
        bool,
        typer.Option("--no-backup", help="Skip the pre-rebuild database backup."),
    ] = False,
    confirm: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip the rebuild confirmation prompt.")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show files during rebuild.")
    ] = False,
) -> None:
    """Show index statistics, or run the v0.13 -> v0.14 rebuild with --rebuild."""
    if ctx.invoked_subcommand is not None:
        return

    cfg = _get_config()
    cfg["ensure_directories"]()

    if rebuild:
        _rebuild_v014(no_backup=no_backup, verbose=verbose, confirm=confirm)
        return

    # Stats is a "normal" read; refuse on a pre-v0.14 database (the index group is
    # exempt from the top-level guard so the rebuild path stays runnable).
    _guard_schema_or_exit()

    # Stats is backend-agnostic (get_stats is on the metadata protocol), so it
    # goes through the storage factory and honors STORAGE_BACKEND.
    from librarian.storage.factory import get_metadata_store

    stats_data = get_metadata_store().get_stats()

    if output_json:
        print(json.dumps(stats_data, indent=2, default=str))
        return

    table = Table(title="Index Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="green")
    table.add_column("Value", style="yellow", justify="right")

    table.add_row("Documents", str(stats_data.get("document_count", 0)))
    table.add_row("Chunks", str(stats_data.get("chunk_count", 0)))
    table.add_row("Embeddings", str(stats_data.get("embedding_count", 0)))
    table.add_row("Database", str(stats_data.get("database_path", "N/A")))

    console.print(table)

    sources = _load_sources()
    if sources:
        rprint(f"\n[dim]Sources: {len(sources)}[/dim]")


@index_app.command("build")
def index_build(
    source: Annotated[
        Optional[str], typer.Option("--source", "-s", help="Rebuild only this source")
    ] = None,
    confirm: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show files")] = False,
) -> None:
    """Rebuild the entire index from scratch."""
    cfg = _get_config()
    cfg["ensure_directories"]()

    # The `index` group is exempt from the startup guard so `--rebuild` stays
    # runnable, but `build` mutates (and the `--source` path deletes a source's
    # rows) before touching the schema. Refuse on a pre-v0.14 database up front
    # so we never delete data and then abort mid-migration.
    _guard_schema_or_exit()

    all_sources = _load_sources()
    if not all_sources:
        rprint("[yellow]No sources registered. Add a source first:[/yellow]")
        rprint("  [cyan]libr add <path>[/cyan]")
        raise typer.Exit(1)

    # Filter to specific source if requested
    if source:
        sources = [s for s in all_sources if s.get("name") == source]
        if not sources:
            rprint(f"[red]Error:[/red] Source not found: {source}")
            raise typer.Exit(1)
    else:
        sources = all_sources

    if not confirm:
        msg = f"specific source: {source}" if source else f"{len(sources)} sources"
        rprint(
            Panel(
                f"[yellow]This will rebuild the index for {msg}.[/yellow]\n\n"
                "This may take a while.",
                title="Rebuild Index",
            )
        )
        if not typer.confirm("Continue?"):
            rprint("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    from librarian.storage.database import get_database

    db = get_database()

    # If rebuilding specific source, only clear that source's documents
    if source:
        rprint(f"[cyan]Clearing documents for {source}...[/cyan]")
        src_data = sources[0]
        documents = db.list_documents()
        for doc in documents:
            if doc.path.startswith(src_data["path"]) and doc.id:
                db.delete_chunks_by_document(doc.id)
                db.delete_document(doc.id)
    else:
        rprint("[cyan]Clearing index...[/cyan]")
        db.clear_all()

    rprint("[cyan]Rebuilding...[/cyan]")
    total_indexed, total_errors = _reindex_sources(sources, verbose)

    rprint(
        Panel(
            f"[green]Rebuild complete![/green]\n\n"
            f"Indexed: [cyan]{total_indexed}[/cyan]\n"
            f"Errors: [red]{total_errors}[/red]",
            title="Build Results",
        )
    )


@index_app.command("clean")
def index_clean(
    confirm: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
) -> None:
    """Remove all indexed data (keeps sources list)."""
    cfg = _get_config()
    cfg["ensure_directories"]()

    if not confirm:
        rprint(
            Panel(
                "[yellow]This will remove all indexed documents and embeddings.[/yellow]\n\n"
                "Source registrations will be kept.\n"
                "Files on disk will NOT be deleted.",
                title="Clean Index",
            )
        )
        if not typer.confirm("Continue?"):
            rprint("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    from librarian.storage.database import get_database

    db = get_database()
    db.clear_all()

    rprint("[green]Index cleared.[/green]")
    rprint("[dim]Run 'libr index build' to rebuild.[/dim]")


@index_app.command("clobber")
def index_clobber(
    confirm: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
) -> None:
    """Remove everything and reinitialize database."""
    cfg = _get_config()

    if not confirm:
        rprint(
            Panel(
                "[red]WARNING: This will delete EVERYTHING:[/red]\n\n"
                "- All indexed documents\n"
                "- All embeddings\n"
                "- Database file\n\n"
                "Source registrations will be kept.",
                title="Clobber Index",
            )
        )
        if not typer.confirm("Are you sure?"):
            rprint("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    db_path = Path(cfg["DATABASE_PATH"])
    if db_path.exists():
        db_path.unlink()
        rprint(f"[green]Deleted:[/green] {db_path}")

    # Reinitialize
    cfg["ensure_directories"]()
    from librarian.storage.database import get_database

    get_database()
    rprint("[green]Database reinitialized.[/green]")


# =============================================================================
# libr docs - Document operations
# =============================================================================


@docs_app.callback(invoke_without_command=True)
def docs_overview(ctx: typer.Context) -> None:
    """Show sources with document counts."""
    if ctx.invoked_subcommand is not None:
        return

    sources = _filter_display_sources(_load_sources())

    if not sources:
        rprint(
            Panel(
                "[yellow]No sources registered.[/yellow]\n\n"
                "Add a source with: [cyan]libr add <path>[/cyan]",
                title="Documents",
            )
        )
        return

    cfg = _get_config()
    cfg["ensure_directories"]()

    from librarian.storage.factory import get_metadata_store

    db = get_metadata_store()
    all_docs = db.list_documents()

    table = Table(title="Sources", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="green")
    table.add_column("Path", style="blue", overflow="fold")
    table.add_column("Docs", justify="right", style="yellow")
    table.add_column("Status")

    for source in sources:
        path = Path(source["path"])
        status = "[green]Active[/green]" if path.exists() else "[red]Missing[/red]"
        doc_count = sum(1 for d in all_docs if d.path.startswith(source["path"]))
        table.add_row(source.get("name", path.name), str(path), str(doc_count), status)

    console.print(table)
    rprint(f"\n[dim]Total documents: {len(all_docs)}[/dim]")


@docs_app.command("list")
def docs_list(
    source: Annotated[
        Optional[str], typer.Option("--source", "-s", help="Filter by source")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-l", help="Max documents")] = 50,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", "-f", help="Output format")
    ] = OutputFormat.TABLE,
) -> None:
    """List all indexed documents."""
    cfg = _get_config()
    cfg["ensure_directories"]()

    from librarian.storage.factory import get_metadata_store

    db = get_metadata_store()
    documents = db.list_documents()

    if source:
        src = _find_source(source)
        if src:
            documents = [d for d in documents if d.path.startswith(src["path"])]

    if not documents:
        if output_format == OutputFormat.JSON:
            print("[]")
        else:
            rprint("[yellow]No documents indexed.[/yellow]")
        return

    # JSON output
    if output_format == OutputFormat.JSON:
        output = [
            {
                "id": d.id,
                "title": d.title,
                "path": d.path,
                "type": d.asset_type.value if d.asset_type else "text",
            }
            for d in documents[:limit]
        ]
        print(json.dumps(output, indent=2))
        return

    # Paths-only output
    if output_format == OutputFormat.PATHS:
        for doc in documents[:limit]:
            print(doc.path)
        return

    # Table output
    table = Table(
        title=f"Documents ({len(documents)} total)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("ID", style="dim", width=4)
    table.add_column("Title", style="green", max_width=38)
    table.add_column("Path", style="blue", max_width=46, overflow="fold")
    table.add_column("Type", style="magenta", width=6)

    home = str(Path.home())
    for doc in documents[:limit]:
        short_path = doc.path.replace(home, "~")
        asset_type = doc.asset_type.value if doc.asset_type else "text"
        table.add_row(str(doc.id), doc.title or "Untitled", short_path, asset_type)

    console.print(table)

    if len(documents) > limit:
        rprint(f"[dim]Showing {limit} of {len(documents)}. Use --limit for more.[/dim]")


@docs_app.command("search")
def docs_search(
    query: Annotated[str, typer.Argument(help="Search query for document titles")],
    limit: Annotated[int, typer.Option("--limit", "-l", help="Max results")] = 20,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", "-f", help="Output format")
    ] = OutputFormat.TABLE,
) -> None:
    """Search document titles (not content)."""
    cfg = _get_config()
    cfg["ensure_directories"]()

    from librarian.storage.factory import get_metadata_store

    db = get_metadata_store()
    documents = db.list_documents()

    query_lower = query.lower()
    matches = [d for d in documents if d.title and query_lower in d.title.lower()]

    if not matches:
        if output_format == OutputFormat.JSON:
            print("[]")
        else:
            rprint(f"[yellow]No documents matching '{query}'[/yellow]")
        return

    if output_format == OutputFormat.JSON:
        output = [{"id": d.id, "title": d.title, "path": d.path} for d in matches[:limit]]
        print(json.dumps(output, indent=2))
        return

    if output_format == OutputFormat.PATHS:
        for doc in matches[:limit]:
            print(doc.path)
        return

    table = Table(
        title=f"Title Search: '{query}' ({len(matches)} matches)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("ID", style="dim", width=4)
    table.add_column("Title", style="green")
    table.add_column("Path", style="blue", max_width=50, overflow="fold")

    home = str(Path.home())
    for doc in matches[:limit]:
        short_path = doc.path.replace(home, "~")
        table.add_row(str(doc.id), doc.title or "Untitled", short_path)

    console.print(table)


# =============================================================================
# libr search - Content search
# =============================================================================


@app.command("search")
def search_cmd(
    query: Annotated[str, typer.Argument(help="Search query")],
    limit: Annotated[int, typer.Option("--limit", "-l", help="Max results")] = 10,
    mode: Annotated[
        SearchMode, typer.Option("--mode", "-m", help="Search mode")
    ] = SearchMode.HYBRID,
    source: Annotated[
        Optional[str], typer.Option("--source", "-s", help="Filter by source")
    ] = None,
    timeframe: Annotated[
        Optional[Timeframe], typer.Option("--timeframe", "-t", help="Time filter")
    ] = None,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", "-f", help="Output format")
    ] = OutputFormat.TABLE,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show content")] = False,
    open_result: Annotated[
        bool, typer.Option("--open", "-o", help="Open first result in editor")
    ] = False,
    copy: Annotated[
        bool, typer.Option("--copy", "-c", help="Copy first result content to clipboard")
    ] = False,
    code: Annotated[
        bool,
        typer.Option("--code", help="Search code files only (uses code embeddings if enabled)"),
    ] = False,
    images: Annotated[
        bool,
        typer.Option("--images", help="Search images only (matches VLM captions / OCR text)"),
    ] = False,
    asset_type: Annotated[
        Optional[str], typer.Option("--type", help="Filter by asset type: text, code, pdf, image")
    ] = None,
) -> None:
    """Search documents using semantic and keyword search."""
    cfg = _get_config()
    cfg["ensure_directories"]()

    from librarian.server import search_library
    from librarian.types import AssetType as LibAssetType
    from librarian.types import SearchMode as LibSearchMode

    # Map CLI search mode to library SearchMode. VECTOR is a legacy CLI
    # alias for SEMANTIC kept for backward compatibility.
    mode_map = {
        SearchMode.HYBRID: LibSearchMode.HYBRID,
        SearchMode.SEMANTIC: LibSearchMode.SEMANTIC,
        SearchMode.VECTOR: LibSearchMode.SEMANTIC,
        SearchMode.KEYWORD: LibSearchMode.KEYWORD,
    }
    lib_mode = mode_map.get(mode, LibSearchMode.HYBRID)

    # Determine asset type filter
    lib_asset_type: LibAssetType | None = None
    if code:
        lib_asset_type = LibAssetType.CODE
    elif images:
        lib_asset_type = LibAssetType.IMAGE
    elif asset_type:
        try:
            lib_asset_type = LibAssetType(asset_type.lower())
        except ValueError:
            rprint(f"[red]Unknown asset type: {asset_type}[/red]")
            raise typer.Exit(1) from None

    with console.status("Searching..."):
        results = _run_async(
            search_library(
                context=None,  # type: ignore[arg-type]
                query=query,
                mode=lib_mode,
                asset_type=lib_asset_type,
                limit=limit,
            )
        )

    # Filter by source
    if source and results:
        src = _find_source(source)
        if src:
            results = [r for r in results if r.get("document_path", "").startswith(src["path"])]

    # Filter by timeframe
    if timeframe and results:
        start_dt, end_dt = _get_timeframe_bounds(timeframe)
        filtered = []
        from librarian.storage.factory import get_metadata_store

        db = get_metadata_store()
        for r in results:
            doc = db.get_document_by_path(r.get("document_path", ""))
            if doc and doc.updated_at:
                if isinstance(doc.updated_at, str):
                    doc_dt = datetime.fromisoformat(doc.updated_at.replace("Z", "+00:00"))
                else:
                    doc_dt = doc.updated_at
                if doc_dt.tzinfo is not None:
                    doc_dt = doc_dt.astimezone().replace(tzinfo=None)
                if start_dt <= doc_dt <= end_dt:
                    filtered.append(r)
        results = filtered

    if not results:
        if output_format == OutputFormat.JSON:
            print("[]")
        else:
            rprint("[yellow]No results found.[/yellow]")
        return

    # Handle special actions
    if open_result:
        first_path = results[0].get("document_path")
        if first_path and Path(first_path).exists():
            _open_in_editor(first_path)
            rprint(f"[green]Opened:[/green] {first_path}")
        return

    if copy:
        first_content = results[0].get("content", "")
        if _copy_to_clipboard(first_content):
            rprint("[green]Copied first result to clipboard.[/green]")
        else:
            rprint("[red]Failed to copy to clipboard.[/red]")
        return

    # JSON output
    if output_format == OutputFormat.JSON:
        output = [
            {
                "score": r.get("score", 0),
                "document_path": r.get("document_path"),
                "content": r.get("content"),
                "heading_path": r.get("heading_path"),
            }
            for r in results
        ]
        print(json.dumps(output, indent=2))
        return

    # Paths-only output
    if output_format == OutputFormat.PATHS:
        seen = set()
        for r in results:
            path = r.get("document_path")
            if path and path not in seen:
                print(path)
                seen.add(path)
        return

    # Table output
    rprint(f"\n[bold]Results[/bold] - {len(results)} matches for '[green]{query}[/green]'\n")

    home = str(Path.home())
    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Path", style="blue", overflow="fold")

    for i, result in enumerate(results, 1):
        score = result.get("score", 0)
        score_color = "green" if score > 0.7 else "yellow" if score > 0.4 else "red"
        doc_path = result.get("document_path", "Unknown").replace(home, "~")
        table.add_row(str(i), f"[{score_color}]{score:.3f}[/{score_color}]", doc_path)

    console.print(table)

    if verbose and results:
        rprint("\n[bold]Content:[/bold]\n")
        for i, result in enumerate(results, 1):
            content = result.get("content", "")[:300].replace("\n", " ").strip()
            if len(result.get("content", "")) > 300:
                content += "..."
            rprint(f"[dim]{i}.[/dim] {content}\n")


@app.command("reprocess")
def reprocess_cmd(
    asset_type: Annotated[
        str,
        typer.Option("--asset-type", help="Asset type to reprocess (e.g. 'image', 'pdf')"),
    ] = "image",
    where: Annotated[
        str,
        typer.Option(
            "--where",
            help=(
                "Match on the chunk's modality_data as key=value "
                "(default: processing_status=failed)."
            ),
        ),
    ] = "processing_status=failed",
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show files being reprocessed")
    ] = False,
) -> None:
    """Retry chunks whose processing failed.

    Finds documents that own a chunk of ``--asset-type`` matching the ``--where``
    status (e.g. images whose VLM caption failed) and re-ingests them, updating
    each chunk's content and ``processing_status`` on success.
    """
    cfg = _get_config()
    cfg["ensure_directories"]()
    _guard_schema_or_exit()

    from librarian.storage.factory import get_storage
    from librarian.types import AssetType as LibAssetType

    try:
        lib_asset_type = LibAssetType(asset_type.lower())
    except ValueError:
        rprint(f"[red]Unknown asset type: {asset_type}[/red]")
        raise typer.Exit(1) from None

    if "=" not in where:
        rprint("[red]--where must be key=value, e.g. processing_status=failed[/red]")
        raise typer.Exit(1)
    key, _, value = where.partition("=")
    key, value = key.strip(), value.strip()

    storage = get_storage()
    try:
        paths = storage.documents_to_reprocess(lib_asset_type, key, value)
    except ValueError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    if not paths:
        rprint(f"[green]Nothing to reprocess[/green] (no {asset_type} chunks where {key}={value}).")
        return

    rprint(f"Reprocessing [bold]{len(paths)}[/bold] {asset_type} document(s)...")

    # Build one Orchestrator (and its vision describer + provider client) up front
    # and reuse it across every document, rather than reconstructing the whole
    # stack per file as `_index_path` would.
    from librarian.orchestrator import Orchestrator

    orchestrator = Orchestrator(storage=storage)
    reindexed = 0
    errors = 0
    for p in paths:
        fp = Path(p)
        if not fp.exists():
            rprint(f"  [yellow]missing:[/yellow] {p}")
            errors += 1
            continue
        try:
            orchestrator.index_file(fp)
            if verbose:
                rprint(f"  [green]+[/green] {p}")
            reindexed += 1
        except Exception as e:
            rprint(f"  [red]error:[/red] {p}: {e}")
            errors += 1

    remaining = storage.documents_to_reprocess(lib_asset_type, key, value)
    rprint(
        f"[green]Reprocessed {reindexed} document(s)[/green]; "
        f"{len(remaining)} still {key}={value}; {errors} error(s)."
    )


# =============================================================================
# libr config - Configuration
# =============================================================================


@config_app.callback(invoke_without_command=True)
def config_default(ctx: typer.Context) -> None:
    """Show current configuration."""
    if ctx.invoked_subcommand is not None:
        return
    config_show()


@config_app.command("show")
def config_show() -> None:
    """Show current configuration."""
    cfg = _get_config()
    settings = _load_settings()

    table = Table(title="Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="green")
    table.add_column("Value", style="yellow")
    table.add_column("Source", style="dim")

    items = [
        ("Database Path", cfg["DATABASE_PATH"], "DATABASE_PATH"),
        ("Embedding Provider", cfg["EMBEDDING_PROVIDER"], "EMBEDDING_PROVIDER"),
        ("Embedding Model", cfg["EMBEDDING_MODEL"], "EMBEDDING_MODEL"),
        ("OCR Enabled", str(cfg["ENABLE_OCR"]), "ENABLE_OCR"),
        ("OCR Language", cfg["OCR_LANGUAGE"], "OCR_LANGUAGE"),
        ("Chunk Size", str(cfg["CHUNK_SIZE"]), "CHUNK_SIZE"),
        ("Chunk Overlap", str(cfg["CHUNK_OVERLAP"]), "CHUNK_OVERLAP"),
        ("Search Limit", str(cfg["SEARCH_LIMIT"]), "SEARCH_LIMIT"),
        ("MMR Lambda", str(cfg["MMR_LAMBDA"]), "MMR_LAMBDA"),
        ("Hybrid Alpha", str(cfg["HYBRID_ALPHA"]), "HYBRID_ALPHA"),
        ("Enable Code Embeddings", str(cfg["ENABLE_CODE_EMBEDDINGS"]), "ENABLE_CODE_EMBEDDINGS"),
        ("Code Embedding Model", cfg["CODE_EMBEDDING_MODEL"], "CODE_EMBEDDING_MODEL"),
        (
            "Code Embedding Dimension",
            str(cfg["CODE_EMBEDDING_DIMENSION"]),
            "CODE_EMBEDDING_DIMENSION",
        ),
        ("Code Embedding Provider", cfg["CODE_EMBEDDING_PROVIDER"], "CODE_EMBEDDING_PROVIDER"),
        (
            "Enable Vision Embeddings",
            str(cfg["ENABLE_VISION_EMBEDDINGS"]),
            "ENABLE_VISION_EMBEDDINGS",
        ),
        ("Vision Embedding Model", cfg["VISION_EMBEDDING_MODEL"], "VISION_EMBEDDING_MODEL"),
        (
            "Vision Embedding Dimension",
            str(cfg["VISION_EMBEDDING_DIMENSION"]),
            "VISION_EMBEDDING_DIMENSION",
        ),
    ]

    for name, value, env_var in items:
        # Determine source: settings.json takes priority, then explicit env var, then default
        if env_var in settings:
            source = "settings"
        elif env_var in os.environ and env_var not in settings:
            source = "env"
        else:
            source = "default"
        table.add_row(name, value, source)

    console.print(table)


@config_app.command("path")
def config_path() -> None:
    """Show configuration file paths."""
    cfg = _get_config()
    rprint(f"Config: [cyan]{CONFIG_DIR}[/cyan]")
    rprint(f"Sources: [cyan]{SOURCES_FILE}[/cyan]")
    rprint(f"Settings: [cyan]{SETTINGS_FILE}[/cyan]")
    rprint(f"Database: [cyan]{cfg['DATABASE_PATH']}[/cyan]")


@config_app.command("models")
def config_models() -> None:
    """Check status of embedding models and download if needed."""
    from librarian.processing.embed import get_embedding_registry

    _get_config()

    registry = get_embedding_registry()

    rprint("[cyan]Checking embedding models...[/cyan]\n")

    results = registry.ensure_models()

    table = Table(title="Embedding Models", show_header=True, header_style="bold cyan")
    table.add_column("Modality", style="green")
    table.add_column("Status")

    for modality, status in results.items():
        if "loaded" in status:
            style = "green"
        elif "disabled" in status:
            style = "yellow"
        else:
            style = "red"
        table.add_row(modality, f"[{style}]{status}[/{style}]")

    console.print(table)


@config_app.command("get")
def config_get(
    key: Annotated[str, typer.Argument(help="Configuration key to get")],
) -> None:
    """Get a specific configuration value."""
    cfg = _get_config()

    if key in cfg and key != "ensure_directories":
        print(cfg[key])
    else:
        rprint(f"[red]Unknown configuration key:[/red] {key}")
        rprint("\nAvailable keys:")
        for k in cfg:
            if k != "ensure_directories":
                rprint(f"  - {k}")
        raise typer.Exit(1)


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Configuration key to set")],
    value: Annotated[str, typer.Argument(help="Value to set")],
) -> None:
    """Set a configuration value (persisted to settings file)."""
    valid_keys = {
        "DATABASE_PATH",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "SEARCH_LIMIT",
        "MMR_LAMBDA",
        "HYBRID_ALPHA",
        "ENABLE_CODE_EMBEDDINGS",
        "CODE_EMBEDDING_MODEL",
        "CODE_EMBEDDING_DIMENSION",
        "CODE_EMBEDDING_PROVIDER",
        "ENABLE_VISION_EMBEDDINGS",
        "VISION_EMBEDDING_MODEL",
        "VISION_EMBEDDING_DIMENSION",
        "ENABLE_OCR",
        "OCR_LANGUAGE",
    }

    if key not in valid_keys:
        rprint(f"[red]Invalid configuration key:[/red] {key}")
        rprint("\nValid keys:")
        for k in sorted(valid_keys):
            rprint(f"  - {k}")
        raise typer.Exit(1)

    settings = _load_settings()

    # Type conversion based on key
    int_keys = (
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "SEARCH_LIMIT",
        "CODE_EMBEDDING_DIMENSION",
        "VISION_EMBEDDING_DIMENSION",
    )
    float_keys = ("MMR_LAMBDA", "HYBRID_ALPHA")
    bool_keys = ("ENABLE_CODE_EMBEDDINGS", "ENABLE_VISION_EMBEDDINGS", "ENABLE_OCR")

    if key in int_keys:
        try:
            settings[key] = int(value)
        except ValueError:
            rprint(f"[red]Error:[/red] {key} must be an integer")
            raise typer.Exit(1) from None
    elif key in float_keys:
        try:
            settings[key] = float(value)
        except ValueError:
            rprint(f"[red]Error:[/red] {key} must be a number")
            raise typer.Exit(1) from None
    elif key in bool_keys:
        settings[key] = value.lower() in ("true", "1", "yes", "on")
    else:
        settings[key] = value

    _save_settings(settings)
    rprint(f"[green]Set {key}=[/green]{value}")
    rprint("[dim]Note: Restart 'libr serve' for changes to take effect.[/dim]")


@config_app.command("edit")
def config_edit() -> None:
    """Open settings file in your editor."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure settings file exists
    if not SETTINGS_FILE.exists():
        _save_settings({})

    editor = _get_editor()
    rprint(f"[cyan]Opening {SETTINGS_FILE} in {editor}...[/cyan]")
    _open_in_editor(str(SETTINGS_FILE))


@config_app.command("reset")
def config_reset(
    confirm: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
) -> None:
    """Reset all settings to defaults."""
    if not confirm and not typer.confirm("Reset all settings to defaults?"):
        rprint("[yellow]Cancelled.[/yellow]")
        return

    if SETTINGS_FILE.exists():
        SETTINGS_FILE.unlink()

    rprint("[green]Settings reset to defaults.[/green]")


# =============================================================================
# libr serve - MCP Server
# =============================================================================


@app.command("serve")
def serve(
    transport: Annotated[str, typer.Argument(help="Transport: stdio or http")] = "stdio",
    host: Annotated[str, typer.Option("--host", "-h", help="HTTP host")] = "",
    port: Annotated[int, typer.Option("--port", "-p", help="HTTP port")] = 0,
    log_level: Annotated[
        str, typer.Option("--log-level", help="Log level (debug, info, warning, error)")
    ] = "warning",
) -> None:
    """Start the MCP server."""
    import logging

    from librarian.config import SERVER_HOST, SERVER_PORT
    from librarian.server import app as mcp_app

    # Use config defaults when CLI args are not provided
    effective_host = host if host else SERVER_HOST
    effective_port = port if port else SERVER_PORT

    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    log_lvl = level_map.get(log_level.lower(), logging.WARNING)

    if transport == "stdio":
        logging.basicConfig(
            level=log_lvl,
            format="%(name)s: %(message)s",
            stream=sys.stderr,
        )
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)
    else:
        logging.basicConfig(level=log_lvl)
        rprint(f"[green]Starting HTTP server on {effective_host}:{effective_port}...[/green]")

    mcp_app.run(transport=transport, host=effective_host, port=effective_port)  # type: ignore[arg-type]


# =============================================================================
# libr version
# =============================================================================


@app.command("version", hidden=True)
def version() -> None:
    """Show version."""
    rprint("Librarian v0.5.0")


if __name__ == "__main__":
    app()
