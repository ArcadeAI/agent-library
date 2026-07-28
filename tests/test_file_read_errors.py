"""Tests for file reading error handling across parsers.

Verifies that parsers handle timeout, permission, and I/O errors
gracefully, especially for cloud-synced filesystems (iCloud, Dropbox).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from librarian.processing.parsers.base import (
    FileReadError,
    FileReadTimeoutError,
    safe_read_bytes,
    safe_read_text,
)
from librarian.processing.parsers.md import MarkdownParser


class TestSafeReadText:
    """Tests for the safe_read_text helper."""

    def test_read_existing_file(self, tmp_path: Path) -> None:
        """Test reading a normal file."""
        test_file = tmp_path / "test.md"
        test_file.write_text("hello world", encoding="utf-8")

        content = safe_read_text(test_file)
        assert content == "hello world"

    def test_read_existing_file_without_sigalrm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test reading text on platforms without POSIX SIGALRM."""
        import signal

        test_file = tmp_path / "test.md"
        test_file.write_text("# Hello\nworld", encoding="utf-8")

        monkeypatch.delattr(signal, "SIGALRM", raising=False)

        content = safe_read_text(test_file)
        assert content == "# Hello\nworld"

    def test_read_from_worker_thread(self, tmp_path: Path) -> None:
        """Reads must work off the main thread.

        ``signal.signal`` raises ValueError outside the main thread, so callers
        indexing from a thread pool would otherwise fail on every file.
        """
        import concurrent.futures

        test_file = tmp_path / "threaded.md"
        test_file.write_text("# Threaded\nbody", encoding="utf-8")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            content = pool.submit(safe_read_text, test_file).result()

        assert content == "# Threaded\nbody"

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test FileNotFoundError for missing files."""
        missing = tmp_path / "nonexistent.md"
        with pytest.raises(FileNotFoundError):
            safe_read_text(missing)

    def test_unicode_fallback(self, tmp_path: Path) -> None:
        """Test fallback to latin-1 for non-UTF-8 files."""
        test_file = tmp_path / "latin.md"
        test_file.write_bytes(b"caf\xe9")  # latin-1 encoded

        content = safe_read_text(test_file)
        assert "caf" in content

    def test_unicode_no_fallback(self, tmp_path: Path) -> None:
        """Test UnicodeDecodeError raised when no fallback encoding."""
        test_file = tmp_path / "bad.md"
        test_file.write_bytes(b"\x80\x81\x82")

        with pytest.raises(UnicodeDecodeError):
            safe_read_text(test_file, fallback_encoding=None)

    def test_timeout_on_read(self, tmp_path: Path) -> None:
        """Test that timeout triggers FileReadTimeoutError."""
        test_file = tmp_path / "slow.md"
        test_file.write_text("content", encoding="utf-8")

        def mock_read_text(*args, **kwargs):
            raise FileReadTimeoutError("Simulated timeout")

        with (
            patch.object(Path, "read_text", side_effect=mock_read_text),
            pytest.raises(FileReadTimeoutError),
        ):
            safe_read_text(test_file, timeout=1)

    def test_permission_error(self, tmp_path: Path) -> None:
        """Test FileReadError for permission denied."""
        test_file = tmp_path / "noperm.md"
        test_file.write_text("secret", encoding="utf-8")

        with (
            patch.object(Path, "read_text", side_effect=PermissionError("Permission denied")),
            pytest.raises(FileReadError, match="Permission denied"),
        ):
            safe_read_text(test_file)

    def test_os_error(self, tmp_path: Path) -> None:
        """Test FileReadError for generic OS errors."""
        test_file = tmp_path / "broken.md"
        test_file.write_text("data", encoding="utf-8")

        with (
            patch.object(Path, "read_text", side_effect=OSError("Disk error")),
            pytest.raises(FileReadError, match="Cannot read"),
        ):
            safe_read_text(test_file)


class TestSafeReadBytes:
    """Tests for the safe_read_bytes helper."""

    def test_read_existing_file(self, tmp_path: Path) -> None:
        """Test reading a normal binary file."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02")

        content = safe_read_bytes(test_file)
        assert content == b"\x00\x01\x02"

    def test_read_existing_file_without_sigalrm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test reading bytes on platforms without POSIX SIGALRM."""
        import signal

        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02")

        monkeypatch.delattr(signal, "SIGALRM", raising=False)

        content = safe_read_bytes(test_file)
        assert content == b"\x00\x01\x02"

    def test_read_from_worker_thread(self, tmp_path: Path) -> None:
        """Binary reads must work off the main thread too."""
        import concurrent.futures

        test_file = tmp_path / "threaded.bin"
        test_file.write_bytes(b"\x00\x01\x02")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            content = pool.submit(safe_read_bytes, test_file).result()

        assert content == b"\x00\x01\x02"

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test FileNotFoundError for missing files."""
        missing = tmp_path / "nonexistent.bin"
        with pytest.raises(FileNotFoundError):
            safe_read_bytes(missing)

    def test_permission_error(self, tmp_path: Path) -> None:
        """Test FileReadError for permission denied."""
        test_file = tmp_path / "noperm.bin"
        test_file.write_bytes(b"secret")

        with (
            patch.object(Path, "read_bytes", side_effect=PermissionError("Permission denied")),
            pytest.raises(FileReadError, match="Permission denied"),
        ):
            safe_read_bytes(test_file)


class TestMarkdownParserErrorHandling:
    """Tests for error handling in MarkdownParser.parse_file()."""

    def test_parse_file_not_found(self, tmp_path: Path) -> None:
        """Test that missing file raises FileNotFoundError."""
        parser = MarkdownParser()
        missing = tmp_path / "missing.md"

        with pytest.raises(FileNotFoundError):
            parser.parse_file(missing)

    def test_parse_file_timeout(self, tmp_path: Path) -> None:
        """Test that timeout raises FileReadTimeoutError."""
        parser = MarkdownParser()
        test_file = tmp_path / "slow.md"
        test_file.write_text("# Title\nContent", encoding="utf-8")

        with (
            patch(
                "librarian.processing.parsers.md.safe_read_text",
                side_effect=FileReadTimeoutError("Timed out"),
            ),
            pytest.raises(FileReadTimeoutError),
        ):
            parser.parse_file(test_file)

    def test_parse_file_permission_error(self, tmp_path: Path) -> None:
        """Test that permission error raises FileReadError."""
        parser = MarkdownParser()
        test_file = tmp_path / "noperm.md"
        test_file.write_text("content", encoding="utf-8")

        with (
            patch(
                "librarian.processing.parsers.md.safe_read_text",
                side_effect=FileReadError("Permission denied"),
            ),
            pytest.raises(FileReadError),
        ):
            parser.parse_file(test_file)

    def test_parse_valid_file(self, tmp_path: Path) -> None:
        """Test parsing a valid markdown file."""
        parser = MarkdownParser()
        test_file = tmp_path / "good.md"
        test_file.write_text("# Hello\nWorld", encoding="utf-8")

        result = parser.parse_file(test_file)
        assert result.title == "Hello"
        assert "World" in result.content


class TestCodeParserErrorHandling:
    """Tests for error handling in CodeParser.parse_file()."""

    def test_parse_file_timeout(self, tmp_path: Path) -> None:
        """Test that timeout raises FileReadTimeoutError."""
        from librarian.processing.parsers.code import CodeParser

        parser = CodeParser()
        test_file = tmp_path / "slow.py"
        test_file.write_text("def foo(): pass", encoding="utf-8")

        with (
            patch(
                "librarian.processing.parsers.code.safe_read_text",
                side_effect=FileReadTimeoutError("Timed out"),
            ),
            pytest.raises(FileReadTimeoutError),
        ):
            parser.parse_file(test_file)

    def test_parse_valid_file(self, tmp_path: Path) -> None:
        """Test parsing a valid code file."""
        from librarian.processing.parsers.code import CodeParser

        parser = CodeParser()
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    pass\n", encoding="utf-8")

        result = parser.parse_file(test_file)
        assert result.asset_type.value == "code"
        assert "hello" in result.content


class TestIndexingServiceErrorHandling:
    """Tests for error handling in IndexingService."""

    def test_index_file_timeout_on_stat(self, clean_db, tmp_path: Path) -> None:
        """Test that stat() timeout is handled gracefully."""
        from librarian.indexing import IndexingService
        from librarian.storage.database import get_database

        # Initialize database first to avoid patching issues
        get_database()

        service = IndexingService()
        test_file = tmp_path / "slow.md"
        test_file.write_text("content", encoding="utf-8")

        # Patch Path.stat directly with new= to work across Python versions.
        # Using new= preserves self-binding (functions are descriptors).
        original_stat = Path.stat

        def patched_stat(self, *args, **kwargs):
            if str(self) == str(test_file):
                raise TimeoutError("stat timeout")
            return original_stat(self, *args, **kwargs)

        with (
            patch.object(Path, "stat", new=patched_stat),
            pytest.raises(FileReadTimeoutError),
        ):
            service.index_file(test_file)

    def test_index_file_stat_os_error(self, clean_db, tmp_path: Path) -> None:
        """Test that stat() OS errors are handled."""
        from librarian.indexing import IndexingService
        from librarian.storage.database import get_database

        # Initialize database first to avoid patching issues
        get_database()

        service = IndexingService()
        test_file = tmp_path / "broken.md"
        test_file.write_text("content", encoding="utf-8")

        original_stat = Path.stat

        def patched_stat(self, *args, **kwargs):
            if str(self) == str(test_file):
                raise OSError("Disk error")
            return original_stat(self, *args, **kwargs)

        with (
            patch.object(Path, "stat", new=patched_stat),
            pytest.raises(FileReadError),
        ):
            service.index_file(test_file)
