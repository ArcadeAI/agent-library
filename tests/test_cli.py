"""Tests for CLI behavior."""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

from librarian import cli
from tests.test_health import _prepared_doc, _storage, _write

LONG_WINDOWS_PATH = (
    r"C:\Users\example\Documents\Codex\2026-05-16"
    r"\concurso-browser-deckbuilder\docs\planning"
    r"\vgf-56-revolt-fx-distribution-and-ui-editor-plan.md"
)
LONG_WINDOWS_FILENAME = "vgf-56-revolt-fx-distribution-and-ui-editor-plan.md"


def assert_table_preserves_long_path_reference(result: Any) -> None:
    assert result.exit_code == 0
    assert "\ufffd" not in result.output
    assert "…" not in result.output
    normalized_output = "".join(
        char for char in result.output if char.isalnum() or char in "\\/:._-"
    )
    assert LONG_WINDOWS_FILENAME in normalized_output


def fake_document() -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        title="VGF-56 plan",
        path=LONG_WINDOWS_PATH,
        asset_type=SimpleNamespace(value="text"),
    )


def test_add_directory_exits_nonzero_when_indexing_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Directory add should fail automation when indexing reports errors."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    fixture = docs_dir / "fixture.md"
    fixture.write_text("# Synthetic Markdown Fixture\n\nContent.", encoding="utf-8")

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "SOURCES_FILE", config_dir / "sources.json")
    monkeypatch.setattr(cli, "SETTINGS_FILE", config_dir / "settings.json")
    monkeypatch.setattr(cli, "_get_config", lambda: {"ensure_directories": lambda: None})
    monkeypatch.setattr(cli, "console", Console(width=500, color_system=None))

    async def fake_server_ingest(
        context: Any,
        directory: str,
        include_ignored: bool = False,
        force_include: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "directory": directory,
            "total_files": 1,
            "indexed": 0,
            "updated": 0,
            "skipped": 0,
            "errors": [{"path": fixture.name, "error": "parser exploded"}],
            "files": [],
        }

    monkeypatch.setattr("librarian.server.index_directory_to_library", fake_server_ingest)

    # Force a wide terminal so Rich does not wrap the error message off-screen.
    monkeypatch.setenv("COLUMNS", "500")
    result = CliRunner().invoke(cli.app, ["add", str(docs_dir), "--verbose"])

    assert result.exit_code == 1
    assert "Errors:" in result.output
    assert "parser exploded" in result.output


def test_add_existing_source_reindexes_without_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registered source may still need indexing in the active backend."""
    source = tmp_path / "README.md"
    source.write_text("# README\n\nstorage", encoding="utf-8")

    config_dir = tmp_path / "config"
    sources_file = config_dir / "sources.json"
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "SOURCES_FILE", sources_file)
    monkeypatch.setattr(cli, "SETTINGS_FILE", config_dir / "settings.json")
    monkeypatch.setattr(cli, "_get_config", lambda: {"ensure_directories": lambda: None})
    monkeypatch.setattr(cli, "console", Console(width=500, color_system=None))

    cli._save_sources([
        {
            "name": "README.md",
            "path": str(source),
            "type": "local",
            "is_file": True,
        }
    ])

    indexed: list[Path] = []

    def fake_index_path(file_path: Path, verbose: bool = False) -> dict[str, Any]:
        indexed.append(file_path)
        return {"status": "created", "chunks": 1}

    monkeypatch.setattr(cli, "_index_path", fake_index_path)

    result = CliRunner().invoke(cli.app, ["add", str(source)])

    assert result.exit_code == 0
    assert indexed == [source.resolve()]
    assert "Source already registered" in result.output
    assert "Source indexed" in result.output
    assert len(cli._load_sources()) == 1


def test_search_table_wraps_long_windows_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Narrow table output should wrap paths instead of replacing them with ellipses."""
    monkeypatch.setattr(cli, "_get_config", lambda: {"ensure_directories": lambda: None})
    monkeypatch.setattr(cli, "console", Console(width=80, color_system=None))

    async def fake_search_library(**kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "score": 1.0,
                "document_path": LONG_WINDOWS_PATH,
                "content": "matched content",
                "heading_path": None,
            }
        ]

    monkeypatch.setattr("librarian.server.search_library", fake_search_library)

    result = CliRunner().invoke(
        cli.app,
        [
            "search",
            "VGF-56 UI editor Fabric Tweakpane Pixi Layout Pixi UI",
            "--format",
            "table",
        ],
    )

    assert_table_preserves_long_path_reference(result)


def test_list_table_wraps_long_windows_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Source listing should not truncate long paths with an ellipsis."""
    monkeypatch.setattr(cli, "console", Console(width=80, color_system=None))
    monkeypatch.setattr(
        cli,
        "_load_sources",
        lambda: [{"name": "docs", "path": LONG_WINDOWS_PATH, "is_file": False}],
    )

    result = CliRunner().invoke(cli.app, ["list"])

    assert_table_preserves_long_path_reference(result)


def test_docs_overview_wraps_long_windows_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Document source overview should wrap long paths."""
    monkeypatch.setattr(cli, "_get_config", lambda: {"ensure_directories": lambda: None})
    monkeypatch.setattr(cli, "console", Console(width=80, color_system=None))
    monkeypatch.setattr(
        cli,
        "_load_sources",
        lambda: [{"name": "docs", "path": LONG_WINDOWS_PATH, "is_file": False}],
    )
    monkeypatch.setattr(
        "librarian.storage.factory.get_metadata_store",
        lambda: SimpleNamespace(list_documents=lambda: [fake_document()]),
    )

    result = CliRunner().invoke(cli.app, ["docs"])

    assert_table_preserves_long_path_reference(result)


def test_docs_list_table_wraps_long_windows_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Document listing should wrap long paths in table output."""
    monkeypatch.setattr(cli, "_get_config", lambda: {"ensure_directories": lambda: None})
    monkeypatch.setattr(cli, "console", Console(width=80, color_system=None))
    monkeypatch.setattr(
        "librarian.storage.factory.get_metadata_store",
        lambda: SimpleNamespace(list_documents=lambda: [fake_document()]),
    )

    result = CliRunner().invoke(cli.app, ["docs", "list", "--format", "table"])

    assert_table_preserves_long_path_reference(result)


def test_docs_search_table_wraps_long_windows_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Document title search should wrap long paths in table output."""
    monkeypatch.setattr(cli, "_get_config", lambda: {"ensure_directories": lambda: None})
    monkeypatch.setattr(cli, "console", Console(width=80, color_system=None))
    monkeypatch.setattr(
        "librarian.storage.factory.get_metadata_store",
        lambda: SimpleNamespace(list_documents=lambda: [fake_document()]),
    )

    result = CliRunner().invoke(cli.app, ["docs", "search", "VGF", "--format", "table"])

    assert_table_preserves_long_path_reference(result)


def test_search_paths_outputs_complete_long_windows_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paths output remains the copyable full-path mode for search results."""
    monkeypatch.setattr(cli, "_get_config", lambda: {"ensure_directories": lambda: None})

    async def fake_search_library(**kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "score": 1.0,
                "document_path": LONG_WINDOWS_PATH,
                "content": "matched content",
                "heading_path": None,
            }
        ]

    monkeypatch.setattr("librarian.server.search_library", fake_search_library)

    result = CliRunner().invoke(
        cli.app,
        [
            "search",
            "VGF-56 UI editor Fabric Tweakpane Pixi Layout Pixi UI",
            "--format",
            "paths",
        ],
    )

    assert result.exit_code == 0
    assert result.output == f"{LONG_WINDOWS_PATH}\n"


def test_search_timeframe_uses_configured_metadata_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeframe filtering must read the active storage backend, not SQLite directly."""
    monkeypatch.setattr(cli, "_get_config", lambda: {"ensure_directories": lambda: None})
    monkeypatch.setattr(cli, "console", Console(width=500, color_system=None))

    async def fake_search_library(**kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "score": 1.0,
                "document_path": "/postgres/doc.md",
                "content": "matched content",
                "heading_path": None,
            }
        ]

    class FakeMetadata:
        def get_document_by_path(self, path: str) -> Any:
            assert path == "/postgres/doc.md"
            return SimpleNamespace(updated_at=datetime.now(timezone.utc))

    monkeypatch.setattr("librarian.server.search_library", fake_search_library)
    monkeypatch.setattr("librarian.storage.factory.get_metadata_store", lambda: FakeMetadata())

    result = CliRunner().invoke(
        cli.app,
        ["search", "postgres", "--timeframe", "today", "--format", "paths"],
    )

    assert result.exit_code == 0, result.output
    assert result.output == "/postgres/doc.md\n"


def test_health_json_outputs_machine_readable(
    clean_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = _storage(clean_db)
    _write(storage, _prepared_doc(tmp_path / "health.md", document_id="cli-health-json"))
    monkeypatch.setattr(cli, "SOURCES_FILE", tmp_path / "sources.json")

    result = CliRunner().invoke(cli.app, ["health", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["document_count"] == 1
    assert payload["chunk_count"] == 1
    assert payload["embedding_count"] == 1
    assert isinstance(payload["issues"], list)


def test_health_table_outputs_summary_sections(
    clean_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = _storage(clean_db)
    _write(storage, _prepared_doc(tmp_path / "health.md", document_id="cli-health-table"))
    monkeypatch.setattr(cli, "SOURCES_FILE", tmp_path / "sources.json")
    monkeypatch.setattr(cli, "console", Console(width=120, color_system=None))

    result = CliRunner().invoke(cli.app, ["health"])

    assert result.exit_code == 0, result.output
    assert "Library Health" in result.output
    assert "Asset Distribution" in result.output
    assert "Embedding Tables" in result.output
    assert "Issues" in result.output


def test_health_show_files_includes_issue_paths(
    clean_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from librarian.storage.write_models import PreparedChunk

    storage = _storage(clean_db)
    issue_path = tmp_path / "missing_embedding.md"
    _write(
        storage,
        _prepared_doc(
            issue_path,
            document_id="cli-health-show-files",
            chunks=[
                PreparedChunk(
                    chunk_id="cli-health-show-files-c1",
                    content="chunk without an embedding but with enough content",
                    chunk_index=0,
                    start_char=0,
                    end_char=48,
                    embedding=None,
                )
            ],
        ),
    )
    monkeypatch.setattr(cli, "SOURCES_FILE", tmp_path / "sources.json")
    monkeypatch.setattr(cli, "console", Console(width=500, color_system=None))

    result = CliRunner().invoke(cli.app, ["health", "--show-files"])

    assert result.exit_code == 0, result.output
    assert "missing_embeddings" in result.output
    assert str(issue_path) in result.output


def test_health_source_filter_limits_json_counts(
    clean_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = _storage(clean_db)
    docs = tmp_path / "docs"
    other = tmp_path / "other"
    docs.mkdir()
    other.mkdir()
    _write(storage, _prepared_doc(docs / "a.md", document_id="cli-health-source-a"))
    _write(storage, _prepared_doc(other / "b.md", document_id="cli-health-source-b"))

    config_dir = tmp_path / "config"
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cli, "SOURCES_FILE", config_dir / "sources.json")
    cli._save_sources([
        {
            "name": "docs",
            "path": str(docs),
            "type": "local",
            "is_file": False,
        }
    ])

    result = CliRunner().invoke(cli.app, ["health", "--json", "--source", "docs"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["source"] == "docs"
    assert payload["document_count"] == 1


class TestIndexRebuild:
    """Tests for the destructive ``libr index --rebuild`` remedy.

    Covers the data-safety contract: a backup is taken to ``<db>.v0-backup``;
    ``--no-backup`` skips it; a failing backup must NOT proceed to delete the
    original; an existing backup is never clobbered; the wipe yields a v0.14
    schema; and the wipe is gated behind an explicit confirmation.
    """

    @staticmethod
    def _canonical_backup(db_path: Path) -> Path:
        return Path(str(db_path) + ".v0-backup")

    def test_rebuild_creates_backup_and_recreates_v014(
        self, clean_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from librarian.storage.schema_version import schema_status_for_path

        db_path = clean_db
        db_path.write_bytes(b"old-db-bytes")
        monkeypatch.setattr(cli, "SOURCES_FILE", tmp_path / "no_sources.json")

        result = CliRunner().invoke(cli.app, ["index", "--rebuild", "--yes"])

        assert result.exit_code == 0, result.output
        backup = self._canonical_backup(db_path)
        assert backup.exists()
        assert backup.read_bytes() == b"old-db-bytes"  # original bytes preserved
        assert schema_status_for_path(db_path) == "v0.14"

    def test_rebuild_no_backup_skips_backup(
        self, clean_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from librarian.storage.schema_version import schema_status_for_path

        db_path = clean_db
        db_path.write_bytes(b"old-db-bytes")
        monkeypatch.setattr(cli, "SOURCES_FILE", tmp_path / "no_sources.json")

        result = CliRunner().invoke(cli.app, ["index", "--rebuild", "--no-backup", "--yes"])

        assert result.exit_code == 0, result.output
        assert not self._canonical_backup(db_path).exists()
        assert schema_status_for_path(db_path) == "v0.14"

    def test_rebuild_aborts_when_backup_fails(
        self, clean_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import shutil

        db_path = clean_db
        db_path.write_bytes(b"precious")
        monkeypatch.setattr(cli, "SOURCES_FILE", tmp_path / "no_sources.json")

        def boom(src: Any, dst: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(shutil, "copy2", boom)

        result = CliRunner().invoke(cli.app, ["index", "--rebuild", "--yes"])

        assert result.exit_code != 0
        # A failed backup must never proceed to unlink the original database.
        assert db_path.exists()
        assert db_path.read_bytes() == b"precious"
        assert not self._canonical_backup(db_path).exists()

    def test_rebuild_does_not_clobber_existing_backup(
        self, clean_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = clean_db
        db_path.write_bytes(b"current")
        canonical = self._canonical_backup(db_path)
        canonical.write_bytes(b"original-v013")
        monkeypatch.setattr(cli, "SOURCES_FILE", tmp_path / "no_sources.json")

        result = CliRunner().invoke(cli.app, ["index", "--rebuild", "--yes"])

        assert result.exit_code == 0, result.output
        # The only pre-v0.14 recovery copy survives untouched.
        assert canonical.read_bytes() == b"original-v013"
        siblings = list(db_path.parent.glob(db_path.name + ".v0-backup-*"))
        assert len(siblings) == 1
        assert siblings[0].read_bytes() == b"current"

    def test_rebuild_aborts_without_confirmation(
        self, clean_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = clean_db
        db_path.write_bytes(b"keep")
        monkeypatch.setattr(cli, "SOURCES_FILE", tmp_path / "no_sources.json")

        result = CliRunner().invoke(cli.app, ["index", "--rebuild"], input="n\n")

        assert result.exit_code == 0
        assert db_path.read_bytes() == b"keep"  # declined: nothing wiped
        assert not self._canonical_backup(db_path).exists()

    def test_rebuild_no_backup_requires_typed_confirmation(
        self, clean_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = clean_db
        db_path.write_bytes(b"keep")
        monkeypatch.setattr(cli, "SOURCES_FILE", tmp_path / "no_sources.json")

        result = CliRunner().invoke(
            cli.app, ["index", "--rebuild", "--no-backup"], input="not-rebuild\n"
        )

        assert result.exit_code == 0
        assert db_path.read_bytes() == b"keep"  # wrong phrase: nothing wiped
