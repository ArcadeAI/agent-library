"""Tests for read-only library health diagnostics."""

from __future__ import annotations

from pathlib import Path

from librarian.health import (
    HealthSourceFilter,
    IssueCode,
    scan_library_health,
)
from librarian.storage.database import Database
from librarian.storage.sqlite_storage import SQLiteStorage
from librarian.storage.write_models import PreparedChunk, PreparedDocument
from librarian.types import AssetType, EmbeddingModality


def _storage(db_path: Path) -> SQLiteStorage:
    db = Database(str(db_path))
    storage = SQLiteStorage(database=db)
    storage.migrate()
    return storage


def _embedding() -> list[float]:
    return [0.01] * 384


def _prepared_doc(
    path: Path | str,
    *,
    document_id: str,
    content: str = "This is a document with enough extracted text for health checks.",
    asset_type: AssetType = AssetType.TEXT,
    metadata: dict | None = None,
    chunks: list[PreparedChunk] | None = None,
    document_source_uri: str | None = None,
    file_mtime: float | None = None,
) -> PreparedDocument:
    path_str = str(path)
    if chunks is None:
        chunks = [
            PreparedChunk(
                chunk_id=f"{document_id}-chunk-1",
                content=content,
                chunk_index=0,
                start_char=0,
                end_char=len(content),
                asset_type=asset_type,
                modality=EmbeddingModality.TEXT,
                embedding=_embedding(),
                model_version="test",
            )
        ]
    return PreparedDocument(
        document_id=document_id,
        path=path_str,
        title=Path(path_str).name,
        content=content,
        metadata=metadata or {},
        asset_type=asset_type,
        document_source_uri=document_source_uri,
        file_mtime=file_mtime,
        chunks=chunks,
    )


def _write(storage: SQLiteStorage, document: PreparedDocument) -> None:
    with storage.transaction() as conn:
        storage.write_upsert(conn, document)


def _issue_codes(report) -> set[str]:  # type: ignore[no-untyped-def]
    return {issue.code.value for issue in report.issues}


def test_health_report_counts_documents_chunks_and_embeddings(tmp_path: Path) -> None:
    db_path = tmp_path / "health.db"
    storage = _storage(db_path)
    doc_path = tmp_path / "docs" / "note.md"
    doc_path.parent.mkdir()
    doc_path.write_text("health note", encoding="utf-8")

    chunks = [
        PreparedChunk(
            chunk_id="doc-counts-c1",
            content="alpha content that is long enough",
            chunk_index=0,
            start_char=0,
            end_char=33,
            embedding=_embedding(),
            model_version="test",
        ),
        PreparedChunk(
            chunk_id="doc-counts-c2",
            content="beta content that is long enough",
            chunk_index=1,
            start_char=34,
            end_char=66,
            embedding=_embedding(),
            model_version="test",
        ),
    ]
    _write(storage, _prepared_doc(doc_path, document_id="doc-counts", chunks=chunks))

    report = scan_library_health(database_path=str(db_path), sources=[])

    assert report.document_count == 1
    assert report.chunk_count == 2
    assert report.embedding_count == 2
    assert report.fts_count == 2
    assert report.embedding_coverage == 1.0
    assert report.fts_coverage == 1.0


def test_health_detects_missing_embeddings(tmp_path: Path) -> None:
    db_path = tmp_path / "health.db"
    storage = _storage(db_path)
    chunks = [
        PreparedChunk(
            chunk_id="missing-embedding-c1",
            content="chunk without an embedding but with enough text",
            chunk_index=0,
            start_char=0,
            end_char=43,
            embedding=None,
        )
    ]
    _write(
        storage, _prepared_doc("/virtual/doc.md", document_id="missing-embedding", chunks=chunks)
    )

    report = scan_library_health(database_path=str(db_path), sources=[])

    assert IssueCode.MISSING_EMBEDDINGS.value in _issue_codes(report)


def test_health_detects_short_pdf_extraction(tmp_path: Path) -> None:
    db_path = tmp_path / "health.db"
    storage = _storage(db_path)
    _write(
        storage,
        _prepared_doc(
            "/virtual/scan.pdf",
            document_id="short-pdf",
            content="tiny",
            asset_type=AssetType.PDF,
            metadata={"page_count": 4},
        ),
    )

    report = scan_library_health(database_path=str(db_path), sources=[])

    assert IssueCode.SHORT_PDF_TEXT.value in _issue_codes(report)


def test_health_detects_image_without_ocr_text(tmp_path: Path) -> None:
    db_path = tmp_path / "health.db"
    storage = _storage(db_path)
    _write(
        storage,
        _prepared_doc(
            "/virtual/image.png",
            document_id="image-no-ocr",
            content="Image: image.png\nFormat: PNG\nDimensions: 20x20",
            asset_type=AssetType.IMAGE,
            metadata={"width": 20, "height": 20},
        ),
    )

    report = scan_library_health(database_path=str(db_path), sources=[])

    assert IssueCode.IMAGE_WITHOUT_OCR.value in _issue_codes(report)


def test_health_accepts_image_ocr_text_in_searchable_content(tmp_path: Path) -> None:
    db_path = tmp_path / "health.db"
    storage = _storage(db_path)
    _write(
        storage,
        _prepared_doc(
            "/virtual/image-with-ocr.png",
            document_id="image-with-ocr",
            content=(
                "Image: image-with-ocr.png\n"
                "Format: PNG\n"
                "Dimensions: 20x20\n\n"
                "Text extracted from image:\nInvoice total due"
            ),
            asset_type=AssetType.IMAGE,
            metadata={"width": 20, "height": 20},
        ),
    )

    report = scan_library_health(database_path=str(db_path), sources=[])

    assert IssueCode.IMAGE_WITHOUT_OCR.value not in _issue_codes(report)


def test_health_detects_small_chunks(tmp_path: Path) -> None:
    db_path = tmp_path / "health.db"
    storage = _storage(db_path)
    _write(
        storage,
        _prepared_doc(
            "/virtual/small.md",
            document_id="small-chunk",
            chunks=[
                PreparedChunk(
                    chunk_id="small-chunk-c1",
                    content="tiny",
                    chunk_index=0,
                    start_char=0,
                    end_char=4,
                    embedding=_embedding(),
                    model_version="test",
                )
            ],
        ),
    )

    report = scan_library_health(database_path=str(db_path), sources=[])

    assert IssueCode.SMALL_CHUNKS.value in _issue_codes(report)


def test_health_ignores_soft_deleted_chunks_for_missing_embeddings(tmp_path: Path) -> None:
    db_path = tmp_path / "health.db"
    storage = _storage(db_path)
    document = _prepared_doc(
        "/virtual/deleted.md",
        document_id="deleted-doc",
        chunks=[
            PreparedChunk(
                chunk_id="deleted-doc-c1",
                content="deleted chunk with no embedding",
                chunk_index=0,
                start_char=0,
                end_char=31,
                embedding=None,
            )
        ],
    )
    _write(storage, document)
    with storage.transaction() as conn:
        storage.soft_delete_document(conn, "deleted-doc", "gone")

    report = scan_library_health(database_path=str(db_path), sources=[])

    assert report.chunk_count == 0
    assert IssueCode.MISSING_EMBEDDINGS.value not in _issue_codes(report)


def test_health_detects_missing_registered_source_path(tmp_path: Path) -> None:
    db_path = tmp_path / "health.db"
    _storage(db_path)
    missing_source = tmp_path / "missing"

    report = scan_library_health(
        database_path=str(db_path),
        sources=[{"name": "missing", "path": str(missing_source), "type": "local"}],
    )

    assert IssueCode.SOURCE_PATH_MISSING.value in _issue_codes(report)


def test_health_does_not_stat_non_file_connector_paths(tmp_path: Path) -> None:
    db_path = tmp_path / "health.db"
    storage = _storage(db_path)
    _write(
        storage,
        _prepared_doc(
            "/slack/channel/message",
            document_id="slack-doc",
            document_source_uri="slack://channel/message",
        ),
    )

    report = scan_library_health(database_path=str(db_path), sources=[])

    assert IssueCode.INDEXED_FILE_MISSING.value not in _issue_codes(report)


def test_health_source_filter_does_not_match_sibling_prefix(tmp_path: Path) -> None:
    db_path = tmp_path / "health.db"
    storage = _storage(db_path)
    docs = tmp_path / "docs"
    archive = tmp_path / "docs-archive"
    docs.mkdir()
    archive.mkdir()
    _write(storage, _prepared_doc(docs / "a.md", document_id="docs-a"))
    _write(storage, _prepared_doc(archive / "b.md", document_id="archive-b"))

    report = scan_library_health(
        database_path=str(db_path),
        source_filter=HealthSourceFilter(name="docs", path=str(docs), is_file=False),
        sources=[],
    )

    assert report.document_count == 1
    assert report.document_asset_counts == {"text": 1}
