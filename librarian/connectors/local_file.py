"""
LocalFileConnector -- the built-in file-source connector.

Emits :class:`DocumentUpsert` events for supported files (markdown, code, PDF,
image) discovered under the configured paths, and :class:`DocumentSoftDelete`
events for previously-seen files that have since disappeared. It reuses the
existing parser registry to decide which files are supported and what asset type
they are.

This connector is the v0.14 replacement for the file-walking half of the old
``IndexingService``. It is stateless and database-free: it receives the last
cursor (a map of path -> mtime) and yields the changes since then. The
orchestrator handles parsing, chunking, embedding and storage.

File discovery applies the same three-layer skip logic as the main-branch
indexing path (``librarian.sources.ignore``): force-include / ``.librariantrack``
override everything, then ``.gitignore`` aggregation under each source root, then
the skip-dirs / unsupported-extension / hidden-file baseline.
"""

import logging
import os
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from librarian.connectors.base import (
    ChangeEvent,
    Connector,
    DocumentSoftDelete,
    DocumentUpsert,
)
from librarian.processing.parsers.base import safe_read_text
from librarian.processing.parsers.registry import get_registry
from librarian.sources.ignore import (
    GitignoreMatcher,
    LibrarianTrackMatcher,
    normalize_force_include,
    should_skip_file,
)
from librarian.types import AssetType

logger = logging.getLogger(__name__)

__all__ = ["LocalFileConnector"]

# Source type recorded on events and used in deterministic id hashing.
SOURCE_TYPE = "file"

# Asset types whose content is text and can be carried inline as ``raw_content``
# (routed through the parser registry's ``parse_content``). Binary assets are
# left for the orchestrator to read from disk via ``parse_file``.
_TEXTUAL_ASSETS = {AssetType.TEXT, AssetType.CODE}


class LocalFileConnector(Connector):
    """Connector that ingests files from the local filesystem."""

    name = "local_file"

    def __init__(
        self,
        paths: list[str | Path] | str | Path,
        *,
        recursive: bool = True,
        include_ignored: bool = False,
        force_include: list[str] | None = None,
    ) -> None:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self._paths = [Path(p).expanduser() for p in paths]
        self._recursive = recursive
        self._include_ignored = include_ignored
        self._force_include = normalize_force_include(force_include)
        self._registry = get_registry()
        self._supported_extensions = self._registry.get_supported_extensions()

    def initial_state(self) -> dict[str, Any]:
        return {"mtimes": {}}

    async def fetch_changes(self, state: dict[str, Any]) -> AsyncIterator[ChangeEvent]:
        known: dict[str, float] = dict(state.get("mtimes", {}))
        current = self._discover()
        seen: set[str] = set()

        for path in current:
            key = str(path)
            seen.add(key)
            try:
                stat_result = path.stat()
            except OSError as e:
                logger.warning("Skipping %s: %s", path, e)
                continue
            mtime = stat_result.st_mtime

            if known.get(key) == mtime:
                continue  # unchanged since last sync

            event = self.build_upsert(path, stat_result)
            if event is None:
                continue
            known[key] = mtime
            event.checkpoint = {"mtimes": dict(known)}
            yield event

        # Files we tracked before but no longer see -> soft delete (tombstone).
        for key in list(known.keys()):
            if key not in seen and key in state.get("mtimes", {}):
                del known[key]
                yield DocumentSoftDelete(
                    source_type=SOURCE_TYPE,
                    source_native_id=key,
                    deletion_reason="source file removed",
                    checkpoint={"mtimes": dict(known)},
                )

    def _discover(self) -> list[Path]:
        files: list[Path] = []
        for base in self._paths:
            if base.is_file():
                # An explicitly-named file is rooted at its own directory for
                # the purpose of .gitignore / .librariantrack resolution.
                if not self._is_skipped(base, base.parent):
                    files.append(base)
            elif base.is_dir():
                # Build the .gitignore / .librariantrack matchers once per root.
                gitignore_matcher = None if self._include_ignored else GitignoreMatcher(base)
                track_matcher = LibrarianTrackMatcher(base)
                pattern = "**/*" if self._recursive else "*"
                for p in base.glob(pattern):
                    # Skip symlinks: glob does not recurse into symlinked dirs but
                    # does yield symlinked files, and is_file() follows them. A
                    # symlink whose target lives outside the source root would
                    # otherwise be read into the searchable corpus.
                    if p.is_symlink():
                        continue
                    if p.is_file() and not self._skip(p, gitignore_matcher, track_matcher):
                        files.append(p)
        # Deterministic order makes cursors and tests stable.
        return sorted({p.resolve() for p in files})

    def _is_skipped(self, file_path: Path, root: Path) -> bool:
        gitignore_matcher = None if self._include_ignored else GitignoreMatcher(root)
        track_matcher = LibrarianTrackMatcher(root)
        return self._skip(file_path, gitignore_matcher, track_matcher)

    def _skip(
        self,
        file_path: Path,
        gitignore_matcher: GitignoreMatcher | None,
        track_matcher: LibrarianTrackMatcher,
    ) -> bool:
        return should_skip_file(
            file_path,
            self._supported_extensions,
            gitignore_matcher,
            force_include=self._force_include,
            track_matcher=track_matcher,
        )

    def build_upsert(self, path: Path, stat_result: os.stat_result) -> DocumentUpsert | None:
        """Build a :class:`DocumentUpsert` for a single file (or ``None`` to skip).

        Exposed so the orchestrator's single-file shim can reuse the exact same
        parser-registry routing and inline-text decisions as the streaming path.

        Takes the caller's already-computed :class:`os.stat_result` rather than
        re-``stat``-ing: both callers stat the file immediately before calling
        this, so passing it through keeps it to one syscall per file and — more
        importantly — keeps the recorded ``file_mtime`` and ``source_created_at``
        derived from the *same* stat, so they can't disagree if a concurrent
        write races between two stats.

        The path is canonicalized so the deterministic id derived from
        ``source_native_id`` is identical regardless of whether the caller passed
        a relative, symlinked, or ``..``-laden spelling of the same file.
        """
        path = path.resolve()
        parser, asset_type = self._registry.get_parser(path)
        if parser is None:
            return None

        raw_content: str | None = None
        if asset_type in _TEXTUAL_ASSETS:
            try:
                raw_content = safe_read_text(path)
            except Exception as e:
                logger.warning("Could not read %s: %s", path, e)
                return None

        return DocumentUpsert(
            source_type=SOURCE_TYPE,
            source_native_id=str(path),
            asset_type=asset_type,
            document_source_uri=path.as_uri(),
            raw_content=raw_content,
            mimetype=path.suffix.lstrip("."),
            metadata={"file_mtime": stat_result.st_mtime},
            source_created_at=self._source_created_at(stat_result),
        )

    @staticmethod
    def _source_created_at(stat_result: os.stat_result) -> datetime:
        """Best-effort file creation timestamp (UTC) from an existing stat.

        Uses the filesystem birth time where the platform exposes it
        (``st_birthtime`` on macOS/BSD), otherwise falls back to the modification
        time so the column is always populated for file-sourced documents.
        """
        created_ts = getattr(stat_result, "st_birthtime", None) or stat_result.st_mtime
        return datetime.fromtimestamp(created_ts, tz=timezone.utc)
