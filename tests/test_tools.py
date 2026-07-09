"""Tests for MCP tools."""

from pathlib import Path
from typing import Any

import pytest
from arcade_tdk.errors import RetryableToolError

# Context can be None in tests since we don't use it
CTX: Any = None


class TestIngestionTools:
    """Tests for document ingestion tools."""

    @pytest.mark.asyncio
    async def test_index_directory_to_library(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test indexing documents from a directory into the library."""
        from librarian.server import index_directory_to_library

        result = await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
        )

        assert "error" not in result
        assert result["total_files"] >= 2
        assert result["indexed"] >= 0 or result["updated"] >= 0

    @pytest.mark.asyncio
    async def test_index_directory_uses_configured_storage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Directory ingest must use the selected storage bundle, not SQLite directly."""
        from librarian import server

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "note.md").write_text("# Note\n\nBody.", encoding="utf-8")

        class FakeMetadata:
            def get_document_by_path(self, path: str) -> None:
                return None

        class FakeStorage:
            metadata = FakeMetadata()

        storage = FakeStorage()

        class FakeOrchestrator:
            def __init__(self, storage: Any) -> None:
                assert storage is fake_storage

            def index_file(self, file_path: Path) -> dict[str, Any]:
                return {"path": str(file_path), "status": "created"}

        fake_storage = storage
        monkeypatch.setattr(server, "get_storage", lambda: fake_storage)
        monkeypatch.setattr("librarian.orchestrator.Orchestrator", FakeOrchestrator)

        result = await server.index_directory_to_library(context=CTX, directory=str(docs_dir))

        assert result["indexed"] == 1
        assert result["errors"] == []

    @pytest.mark.asyncio
    async def test_index_nonexistent_directory(self, clean_db: Path) -> None:
        """Test indexing from a nonexistent directory raises a retryable error."""
        from librarian.server import index_directory_to_library

        with pytest.raises(RetryableToolError) as exc_info:
            await index_directory_to_library(
                context=CTX,
                directory="/nonexistent/path",
            )

        assert "no directory at" in str(exc_info.value)
        assert exc_info.value.additional_prompt_content is not None

    @pytest.mark.asyncio
    async def test_index_change_detection(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test that unchanged files are skipped on re-index."""
        import time

        from librarian.server import index_directory_to_library

        # First index - should index all files
        result1 = await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
        )
        assert "error" not in result1
        initial_indexed = result1["indexed"]
        assert initial_indexed >= 2  # We have at least 2 test files

        # Second index without changes - should skip all
        result2 = await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
        )
        assert "error" not in result2
        assert result2["indexed"] == 0
        assert result2["updated"] == 0
        assert result2["skipped"] == result1["total_files"]

        # Modify one file (update mtime)
        time.sleep(0.1)  # Ensure mtime changes
        test_file = temp_docs_dir / "test1.md"
        original_content = test_file.read_text()
        test_file.write_text(original_content + "\n\nModified content.")

        # Third index - should only update the modified file
        result3 = await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
        )
        assert "error" not in result3
        assert result3["updated"] == 1
        assert result3["skipped"] == result1["total_files"] - 1

    @pytest.mark.asyncio
    async def test_add_to_library(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test adding new content to the library."""
        from librarian.server import add_to_library

        result = await add_to_library(
            context=CTX,
            content="# New Document\n\nThis is new content.",
            title="new_doc",
            directory=str(temp_docs_dir),
        )

        assert result.get("status") == "stored"
        assert "new_doc.md" in result.get("path", "")
        assert (temp_docs_dir / "new_doc.md").exists()

    @pytest.mark.asyncio
    async def test_add_to_library_no_embedding_fallback_uses_factory(
        self, temp_docs_dir: Path, clean_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When indexing fails, the no-embedding fallback stores the document
        through the storage factory (honoring STORAGE_BACKEND), not a direct
        SQLite call. The document row is persisted and retrievable even though
        no chunks/embeddings were written."""
        import librarian.server as server
        from librarian.storage.factory import get_metadata_store

        def _boom(_file_path: Path) -> dict:
            raise RuntimeError("embedding service unavailable")

        monkeypatch.setattr(server, "_process_and_index_file", _boom)

        result = await server.add_to_library(
            context=CTX,
            content="# Fallback Doc\n\nStored without embeddings.",
            title="fallback_doc",
            directory=str(temp_docs_dir),
        )

        assert result.get("status") == "stored_partial"
        assert result.get("indexed") is False

        # The document row landed through the factory and is retrievable.
        file_path = temp_docs_dir / "fallback_doc.md"
        doc = get_metadata_store().get_document_by_path(str(file_path))
        assert doc is not None
        assert "Stored without embeddings." in doc.content

    @pytest.mark.asyncio
    async def test_no_embedding_fallback_preserves_existing_index(
        self, temp_docs_dir: Path, clean_db: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fallback must not wipe an already-indexed document's chunks.

        If a file was indexed, then removed from disk out-of-band (its index rows
        survive) and re-added during an embedding outage, the zero-chunk fallback
        upsert would otherwise hard-delete the live chunks. The guard skips the
        write when the path is already indexed, leaving search intact.
        """
        import librarian.server as server
        from librarian.storage.factory import get_metadata_store, get_read_storage

        # 1. Index a document normally (it gets chunks).
        await server.add_to_library(
            context=CTX,
            content="# Kept Doc\n\nThis content stays searchable.",
            title="kept_doc",
            directory=str(temp_docs_dir),
        )
        file_path = temp_docs_dir / "kept_doc.md"
        doc = get_metadata_store().get_document_by_path(str(file_path))
        assert doc is not None

        def _count_chunks() -> int:
            with get_read_storage().database._connection() as conn:
                return conn.execute(
                    "SELECT COUNT(*) AS n FROM chunks WHERE document_id = ? AND deleted_at IS NULL",
                    (doc.id,),
                ).fetchone()["n"]

        chunks_before = _count_chunks()
        assert chunks_before > 0

        # 2. File removed out-of-band (index rows remain), then re-added during
        #    an embedding outage.
        file_path.unlink()

        def _boom(_file_path: Path) -> dict:
            raise RuntimeError("embedding service unavailable")

        monkeypatch.setattr(server, "_process_and_index_file", _boom)

        result = await server.add_to_library(
            context=CTX,
            content="# Kept Doc\n\nThis content stays searchable.",
            title="kept_doc",
            directory=str(temp_docs_dir),
        )

        assert result.get("status") == "stored_partial"
        # The pre-existing chunks survive -- the fallback left the index alone.
        assert _count_chunks() == chunks_before

    @pytest.mark.asyncio
    async def test_add_to_library_with_tags(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test adding content to the library with tags."""
        from librarian.server import add_to_library

        result = await add_to_library(
            context=CTX,
            content="Content here.",
            title="with_tags",
            directory=str(temp_docs_dir),
            tags=["test", "example"],
            metadata={"author": "tester"},
        )

        assert result.get("status") == "stored"
        content = (temp_docs_dir / "with_tags.md").read_text()
        assert "---" in content
        assert "tags:" in content

    @pytest.mark.asyncio
    async def test_add_to_library_already_exists(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Adding content with an existing title should raise a retryable error."""
        from librarian.server import add_to_library

        with pytest.raises(RetryableToolError) as exc_info:
            await add_to_library(
                context=CTX,
                content="New content.",
                title="test1",  # Already exists
                directory=str(temp_docs_dir),
            )

        assert "already lives at" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_library_doc(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test updating existing content in the library."""
        from librarian.server import update_library_doc

        file_path = temp_docs_dir / "test1.md"
        result = await update_library_doc(
            context=CTX,
            path=str(file_path),
            content="# Updated Title\n\nUpdated content.",
        )

        assert result.get("status") == "updated"
        new_content = file_path.read_text()
        assert "Updated" in new_content


class TestSearchTools:
    """Tests for search tools."""

    @pytest.mark.asyncio
    async def test_search_library_hybrid(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test hybrid search in the library (default mode)."""
        from librarian.server import index_directory_to_library, search_library

        await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
        )

        results = await search_library(context=CTX, query="test document", limit=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_library_returns_v014_shape(
        self, temp_docs_dir: Path, clean_db: Path
    ) -> None:
        """Search hits expose the v0.14 shape: str chunk_id + additive fields."""
        from librarian.server import index_directory_to_library, search_library
        from librarian.types import SearchMode

        await index_directory_to_library(context=CTX, directory=str(temp_docs_dir))

        results = await search_library(context=CTX, query="test", mode=SearchMode.KEYWORD, limit=5)
        assert results, "expected at least one keyword hit for indexed docs"

        hit = results[0]
        # chunk_id is the deterministic hash rendered as a hex string, not an int.
        assert isinstance(hit["chunk_id"], str)
        assert len(hit["chunk_id"]) == 64  # sha256 hexdigest
        # Additive v0.14 fields are present and populated for freshly ingested rows.
        assert "chunk_source_uri" in hit
        assert hit["chunk_source_uri"] and hit["chunk_source_uri"].startswith("file://")
        assert isinstance(hit["chunk_index"], int)
        assert isinstance(hit["document_size"], int)

    @pytest.mark.asyncio
    async def test_search_library_semantic(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test semantic search mode."""
        from librarian.server import index_directory_to_library, search_library
        from librarian.types import SearchMode

        await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
        )

        results = await search_library(
            context=CTX, query="document content", mode=SearchMode.SEMANTIC, limit=5
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_library_keyword(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test keyword search mode."""
        from librarian.server import index_directory_to_library, search_library
        from librarian.types import SearchMode

        await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
        )

        results = await search_library(context=CTX, query="test", mode=SearchMode.KEYWORD, limit=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_library_with_asset_type(
        self, temp_docs_dir: Path, clean_db: Path
    ) -> None:
        """Test search with asset type filter."""
        from librarian.server import index_directory_to_library, search_library
        from librarian.types import AssetType

        await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
        )

        results = await search_library(
            context=CTX, query="test", asset_type=AssetType.TEXT, limit=5
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_library_with_timeframe(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test search with timeframe filter."""
        from librarian.server import index_directory_to_library, search_library
        from librarian.utils.timeframe import Timeframe

        await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
        )

        results = await search_library(
            context=CTX, query="test", timeframe=Timeframe.THIS_YEAR, limit=5
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_library_empty_query(self, clean_db: Path) -> None:
        """Test search with empty query."""
        from librarian.server import search_library

        results = await search_library(context=CTX, query="", limit=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_library_include_deleted_opt_in(
        self, temp_docs_dir: Path, clean_db: Path
    ) -> None:
        """search_library hides soft-deleted chunks by default; opt-in surfaces them."""
        from librarian.server import index_directory_to_library, search_library
        from librarian.storage.factory import get_storage
        from librarian.types import SearchMode

        await index_directory_to_library(context=CTX, directory=str(temp_docs_dir))

        # Soft-delete every indexed document (tombstone, not hard delete).
        storage = get_storage()
        with storage.transaction() as conn:
            doc_ids = [
                row["document_id"]
                for row in conn.execute("SELECT document_id FROM documents").fetchall()
            ]
            for did in doc_ids:
                storage.soft_delete_document(conn, did, "test tombstone")
        assert doc_ids

        # Default: tombstoned content is excluded.
        default_hits = await search_library(
            context=CTX, query="test", mode=SearchMode.KEYWORD, limit=10
        )
        assert default_hits == []

        # Opt-in: the soft-deleted chunks come back.
        opted_in = await search_library(
            context=CTX,
            query="test",
            mode=SearchMode.KEYWORD,
            limit=10,
            include_deleted=True,
        )
        assert opted_in != []


class TestExpandContext:
    """Tests for the expand_context MCP tool."""

    @staticmethod
    def _multi_section_doc(tmp_path: Path) -> Path:
        docs = tmp_path / "docs"
        docs.mkdir()
        sections = [
            f"## Section {i}\n\n" + f"Body of section {i}. distinctword{i} " * 20 for i in range(5)
        ]
        (docs / "thread.md").write_text("# Conversation\n\n" + "\n\n".join(sections))
        return docs

    @pytest.mark.asyncio
    async def test_expand_context_returns_ordered_neighbors(
        self, tmp_path: Path, clean_db: Path
    ) -> None:
        from librarian.server import expand_context, index_directory_to_library
        from librarian.storage.factory import get_metadata_store

        docs = self._multi_section_doc(tmp_path)
        await index_directory_to_library(context=CTX, directory=str(docs))

        # Grab the ordered chunk ids straight from storage so we can anchor on a
        # known middle chunk and assert the exact window deterministically.
        db = get_metadata_store()
        with db._connection() as conn:  # type: ignore[attr-defined]
            ordered = [
                row["chunk_id"]
                for row in conn.execute(
                    "SELECT chunk_id FROM chunks ORDER BY chunk_index"
                ).fetchall()
            ]
        assert len(ordered) >= 5, "doc should chunk into several sections"

        middle = len(ordered) // 2
        anchor = ordered[middle]
        neighbors = await expand_context(context=CTX, chunk_id=anchor, before=2, after=2)

        returned_ids = [n["chunk_id"] for n in neighbors]
        assert anchor not in returned_ids  # anchor itself is not repeated
        assert len(neighbors) == 4
        # Neighbors come back in source order and carry the v0.14 shape.
        indices = [n["chunk_index"] for n in neighbors]
        assert indices == sorted(indices)
        for n in neighbors:
            assert isinstance(n["chunk_id"], str)
            assert n["chunk_source_uri"] and n["chunk_source_uri"].startswith("file://")

    @pytest.mark.asyncio
    async def test_expand_context_clips_at_boundary(self, tmp_path: Path, clean_db: Path) -> None:
        from librarian.server import expand_context, index_directory_to_library
        from librarian.storage.factory import get_metadata_store

        docs = self._multi_section_doc(tmp_path)
        await index_directory_to_library(context=CTX, directory=str(docs))

        db = get_metadata_store()
        with db._connection() as conn:  # type: ignore[attr-defined]
            first = conn.execute(
                "SELECT chunk_id FROM chunks ORDER BY chunk_index LIMIT 1"
            ).fetchone()["chunk_id"]

        neighbors = await expand_context(context=CTX, chunk_id=first, before=2, after=2)
        # No chunks precede the first, so only the following neighbors come back.
        assert 1 <= len(neighbors) <= 2
        assert all(n["chunk_index"] >= 1 for n in neighbors)

    @pytest.mark.asyncio
    async def test_expand_context_blank_chunk_id_raises(self, clean_db: Path) -> None:
        from librarian.server import expand_context

        with pytest.raises(RetryableToolError):
            await expand_context(context=CTX, chunk_id="  ")

    @pytest.mark.asyncio
    async def test_expand_context_unknown_chunk_raises(self, clean_db: Path) -> None:
        from librarian.server import expand_context
        from librarian.storage.factory import get_storage

        # Migrate the (empty) DB so the lookup reaches the v0.14 schema and the
        # unknown id resolves to "no chunk found" rather than a missing column.
        get_storage()

        with pytest.raises(RetryableToolError):
            await expand_context(context=CTX, chunk_id="no-such-chunk")

    @pytest.mark.asyncio
    async def test_expand_context_unmigrated_db_degrades_gracefully(self, clean_db: Path) -> None:
        """A serve-only process on a never-migrated DB gets a retryable error, not a crash.

        Without ingest running in-process the read path never adds
        ``chunks.chunk_id``; the tool must degrade like search_library rather
        than surfacing a non-retryable backend error.
        """
        from librarian.server import expand_context

        with pytest.raises(RetryableToolError):
            await expand_context(context=CTX, chunk_id="anything")

    @pytest.mark.asyncio
    async def test_expand_context_single_chunk_returns_empty(
        self, tmp_path: Path, clean_db: Path
    ) -> None:
        """A single-chunk document has no neighbors -> [] (not an error)."""
        from librarian.server import expand_context, index_directory_to_library
        from librarian.storage.factory import get_metadata_store

        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "solo.md").write_text("# Solo\n\nOne short paragraph.")
        await index_directory_to_library(context=CTX, directory=str(docs))

        db = get_metadata_store()
        with db._connection() as conn:  # type: ignore[attr-defined]
            rows = conn.execute("SELECT chunk_id FROM chunks").fetchall()
        assert len(rows) == 1
        assert await expand_context(context=CTX, chunk_id=rows[0]["chunk_id"]) == []

    @pytest.mark.asyncio
    async def test_expand_context_tombstoned_suggests_include_deleted(
        self, tmp_path: Path, clean_db: Path
    ) -> None:
        """Tombstoned neighbors under the default flag steer the agent to include_deleted."""
        from librarian.server import expand_context, index_directory_to_library
        from librarian.storage.factory import get_metadata_store, get_storage

        docs = self._multi_section_doc(tmp_path)
        await index_directory_to_library(context=CTX, directory=str(docs))

        db = get_metadata_store()
        with db._connection() as conn:  # type: ignore[attr-defined]
            ordered = [
                row["chunk_id"]
                for row in conn.execute(
                    "SELECT chunk_id FROM chunks ORDER BY chunk_index"
                ).fetchall()
            ]
        anchor = ordered[len(ordered) // 2]

        # Tombstone the whole document (soft-delete tombstones every chunk).
        storage = get_storage()
        with storage.transaction() as conn:
            for did in [
                row["document_id"]
                for row in conn.execute("SELECT document_id FROM documents").fetchall()
            ]:
                storage.soft_delete_document(conn, did, "test tombstone")

        # Default flag: the neighbors exist but are tombstoned -> guided to the flag.
        with pytest.raises(RetryableToolError) as exc:
            await expand_context(context=CTX, chunk_id=anchor, before=2, after=2)
        assert "include_deleted" in str(exc.value.additional_prompt_content)

        # Opt-in: neighbors come back, each flagged with its deleted_at tombstone.
        neighbors = await expand_context(
            context=CTX, chunk_id=anchor, before=2, after=2, include_deleted=True
        )
        assert neighbors
        assert all(n["deleted_at"] for n in neighbors)

    def test_expand_context_is_reexportable(self) -> None:
        """Consumers re-export the tool by importing it from the server module."""
        from librarian.server import expand_context

        assert callable(expand_context)


class TestDocumentManagementTools:
    """Tests for document management tools."""

    @pytest.mark.asyncio
    async def test_read_from_library(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test reading content from the library."""
        from librarian.server import index_directory_to_library, read_from_library

        await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
        )

        file_path = temp_docs_dir / "test1.md"
        result = await read_from_library(context=CTX, path=str(file_path))

        assert "error" not in result
        assert "content" in result
        assert "Test Document 1" in result["content"]

    @pytest.mark.asyncio
    async def test_read_nonexistent_from_library(self, clean_db: Path) -> None:
        """Reading a missing document should raise a retryable error."""
        from librarian.server import read_from_library

        with pytest.raises(RetryableToolError) as exc_info:
            await read_from_library(context=CTX, path="/nonexistent/file.md")

        assert "Nothing in the library" in str(exc_info.value)
        assert exc_info.value.additional_prompt_content is not None

    @pytest.mark.asyncio
    async def test_remove_from_library(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test removing content from the library."""
        from librarian.server import index_directory_to_library, remove_from_library

        await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
        )

        file_path = temp_docs_dir / "test1.md"
        result = await remove_from_library(context=CTX, path=str(file_path), delete_file=True)

        # Index removal routes through the storage factory (delete_document_by_path).
        assert result.get("removed_from_index") is True
        assert result.get("file_deleted") is True
        assert not file_path.exists()

        # The document is gone from the index: a second removal is a no-op.
        from librarian.storage.factory import get_metadata_store

        assert get_metadata_store().get_document_by_path(str(file_path)) is None

    @pytest.mark.asyncio
    async def test_list_library_contents(self, temp_docs_dir: Path, clean_db: Path) -> None:
        """Test listing all library contents."""
        from librarian.server import index_directory_to_library, list_library_contents

        await index_directory_to_library(
            context=CTX,
            directory=str(temp_docs_dir),
        )

        result = await list_library_contents(context=CTX, limit=100)
        assert isinstance(result, list)
        assert len(result) >= 2

        # A negative limit (agents' "-1 = no limit" convention) must not reach
        # SQL: it errors on Postgres and returns everything on SQLite. It's
        # clamped to "no bound" and returns the full list without raising.
        unbounded = await list_library_contents(context=CTX, limit=-1)
        assert len(unbounded) == len(result)

    @pytest.mark.asyncio
    async def test_get_library_overview_stats(self, clean_db: Path) -> None:
        """STATS view returns library totals + the active config block."""
        from librarian.server import get_library_overview
        from librarian.types import LibraryView

        result = await get_library_overview(context=CTX, view=LibraryView.STATS)
        assert result["view"] == "stats"
        assert "document_count" in result
        assert "chunk_count" in result
        assert "config" in result

    @pytest.mark.asyncio
    async def test_get_library_overview_sections(self, clean_db: Path) -> None:
        """SECTIONS view returns the per-source section list (may be empty)."""
        from librarian.server import get_library_overview
        from librarian.types import LibraryView

        result = await get_library_overview(context=CTX, view=LibraryView.SECTIONS)
        assert result["view"] == "sections"
        assert "sections" in result
        assert isinstance(result["sections"], list)

    @pytest.mark.asyncio
    async def test_get_library_overview_tree(self, clean_db: Path) -> None:
        """TREE view returns recursive structure per source (may be empty)."""
        from librarian.server import get_library_overview
        from librarian.types import LibraryView

        result = await get_library_overview(context=CTX, view=LibraryView.TREE, depth=2)
        assert result["view"] == "tree"
        assert "sources" in result
        assert isinstance(result["sources"], list)
