"""
Vision pipeline tests (issue #53, Slice 4).

Cover the VLM-text image path, the failure/reprocess loop, PDF OCR wiring, and
the CLIP-retirement (strategy "Option C / Hybrid") behaviour:

* ``IMAGE_GENERATE_CAPTIONS=true`` -> the image chunk's content is the VLM's
  description + transcribed text;
* flag off -> a metadata-only chunk (the VLM is never called);
* a simulated VLM failure -> content ``"[image, processing failed]"`` with
  ``processing_status='failed'``, still searchable;
* ``libr reprocess`` retries failed chunks and updates content + status;
* a PDF with image-only pages (``PDF_OCR_ENABLED=true``) -> page chunks carrying
  OCR'd text;
* ``ENABLE_VISION_EMBEDDINGS`` is deprecated and a no-op (images embed as TEXT).

The hosted VLM is always injected as a fake, so these run without network,
Pillow, or the provider SDKs.
"""

import importlib
from pathlib import Path

import pytest

from librarian import config
from librarian.orchestrator import Orchestrator
from librarian.processing.vision import (
    BaseVisionDescriber,
    LLMVisionDescriber,
    VisionError,
    VisionResult,
    parse_vlm_response,
)
from librarian.storage.database import Database
from librarian.storage.sqlite_storage import SQLiteStorage
from librarian.types import AssetType, EmbeddingModality

from .conftest import FakeEmbedder

# A real 1x1 transparent PNG -- valid enough for Pillow to open in the
# captions-off path, and arbitrary bytes for the (PIL-free) VLM path.
_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6360000002000154a24f5f0000000049454e44ae426082"
)


# =============================================================================
# Fakes
# =============================================================================


class FakeVisionDescriber(BaseVisionDescriber):
    """Returns a canned :class:`VisionResult` and counts invocations."""

    def __init__(self, result: VisionResult | None = None) -> None:
        self._result = result or VisionResult(
            description="A red square diagram", transcribed_text="HELLO 123"
        )
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake-vlm"

    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> VisionResult:
        self.calls += 1
        return self._result


class FailingVisionDescriber(BaseVisionDescriber):
    """Always raises, to exercise the failure path."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake-vlm"

    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> VisionResult:
        self.calls += 1
        raise VisionError("simulated VLM failure")


# =============================================================================
# Fixtures / helpers
# =============================================================================


@pytest.fixture
def storage(tmp_path: Path) -> SQLiteStorage:
    db = Database(str(tmp_path / "vision.db"))
    st = SQLiteStorage(database=db)
    st.migrate()
    return st


def _write_png(tmp_path: Path, name: str = "diagram.png") -> Path:
    img = tmp_path / name
    img.write_bytes(_PNG_1x1)
    return img


def _chunks(storage: SQLiteStorage) -> list[dict]:
    conn = storage.database._get_connection()
    rows = conn.execute(
        "SELECT content, asset_type, modality, modality_data FROM chunks "
        "WHERE deleted_at IS NULL ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]


# =============================================================================
# Vision module unit tests (no deps, no network)
# =============================================================================


def test_vision_result_content_combines_description_and_text() -> None:
    result = VisionResult(description="A chart", transcribed_text="Q1 revenue")
    assert "A chart" in result.content
    assert "Transcribed text:" in result.content
    assert "Q1 revenue" in result.content


def test_vision_result_content_description_only() -> None:
    result = VisionResult(description="A chart", transcribed_text="")
    assert result.content == "A chart"
    assert "Transcribed text" not in result.content


def test_parse_vlm_response_plain_json() -> None:
    result = parse_vlm_response('{"description": "a cat", "text": "meow"}')
    assert result.description == "a cat"
    assert result.transcribed_text == "meow"


def test_parse_vlm_response_fenced_json() -> None:
    raw = '```json\n{"description": "a dog", "text": ""}\n```'
    result = parse_vlm_response(raw)
    assert result.description == "a dog"
    assert result.transcribed_text == ""


def test_parse_vlm_response_non_json_falls_back_to_description() -> None:
    result = parse_vlm_response("Just a plain sentence describing the image.")
    assert result.description == "Just a plain sentence describing the image."
    assert result.transcribed_text == ""


def test_parse_vlm_response_empty_raises() -> None:
    with pytest.raises(VisionError):
        parse_vlm_response("   ")


def test_llm_describer_unknown_provider_raises() -> None:
    with pytest.raises(VisionError):
        LLMVisionDescriber(provider="bananas")


def test_llm_describer_openai_with_injected_client() -> None:
    class _Msg:
        content = '{"description": "a logo", "text": "ACME"}'

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = (_Choice(),)

    class _Completions:
        def create(self, **kwargs: object) -> _Resp:
            # The image must be sent as a base64 data URL.
            content = kwargs["messages"][1]["content"]  # type: ignore[index]
            assert any(part.get("type") == "image_url" for part in content)
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    describer = LLMVisionDescriber(provider="openai", model="gpt-4o", client=_Client())
    result = describer.describe(b"\x89PNG...", "image/png")
    assert result.description == "a logo"
    assert result.transcribed_text == "ACME"


def test_llm_describer_wraps_client_errors_as_vision_error() -> None:
    class _Boom:
        def __getattr__(self, name: str) -> object:
            raise RuntimeError("network down")

    describer = LLMVisionDescriber(provider="openai", client=_Boom())
    with pytest.raises(VisionError):
        describer.describe(b"bytes", "image/png")


# =============================================================================
# AC: caption on -> VLM description + transcribed text becomes the chunk content
# =============================================================================


def test_caption_on_produces_vlm_content(
    storage: SQLiteStorage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "IMAGE_GENERATE_CAPTIONS", True)
    img = _write_png(tmp_path)
    describer = FakeVisionDescriber()
    orch = Orchestrator(storage=storage, embedder=FakeEmbedder(), vision_describer=describer)

    result = orch.index_file(img)

    assert result["chunks"] == 1
    assert describer.calls == 1
    chunks = _chunks(storage)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["asset_type"] == AssetType.IMAGE.value
    # Image content is the VLM description + transcribed text...
    assert "A red square diagram" in chunk["content"]
    assert "HELLO 123" in chunk["content"]
    # ...embedded in the TEXT space (CLIP VISION modality retired).
    assert chunk["modality"] == EmbeddingModality.TEXT.value
    assert '"processing_status": "ok"' in chunk["modality_data"]


# =============================================================================
# AC: flag off -> metadata-only chunk; the VLM is never called
# =============================================================================


def test_caption_off_produces_metadata_only_chunk(
    storage: SQLiteStorage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "IMAGE_GENERATE_CAPTIONS", False)
    img = _write_png(tmp_path)
    describer = FakeVisionDescriber()
    orch = Orchestrator(storage=storage, embedder=FakeEmbedder(), vision_describer=describer)

    orch.index_file(img)

    assert describer.calls == 0, "VLM must not be called with captions disabled"
    chunks = _chunks(storage)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["asset_type"] == AssetType.IMAGE.value
    # No VLM description and no failure placeholder -- just metadata.
    assert "A red square diagram" not in chunk["content"]
    assert "[image, processing failed]" not in chunk["content"]


# =============================================================================
# AC: simulated VLM failure -> placeholder content + failed status, searchable
# =============================================================================


def test_caption_failure_records_failed_status(
    storage: SQLiteStorage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "IMAGE_GENERATE_CAPTIONS", True)
    img = _write_png(tmp_path)
    orch = Orchestrator(
        storage=storage, embedder=FakeEmbedder(), vision_describer=FailingVisionDescriber()
    )

    orch.index_file(img)

    chunks = _chunks(storage)
    assert len(chunks) == 1
    assert chunks[0]["content"] == "[image, processing failed]"
    assert '"processing_status": "failed"' in chunks[0]["modality_data"]
    # The failed chunk is discoverable by the reprocess query.
    targets = storage.documents_to_reprocess(AssetType.IMAGE, "processing_status", "failed")
    assert str(img) in targets


def test_failed_image_chunk_is_searchable(
    clean_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "IMAGE_GENERATE_CAPTIONS", True)
    from librarian.processing.embed import get_embedder
    from librarian.retrieval.search import HybridSearcher
    from librarian.storage.factory import get_storage

    img = _write_png(tmp_path)
    orch = Orchestrator(
        storage=get_storage(), embedder=FakeEmbedder(), vision_describer=FailingVisionDescriber()
    )
    orch.index_file(img)

    searcher = HybridSearcher(embedder=get_embedder())
    results = searcher.keyword_search("processing failed", limit=10)
    assert any("[image, processing failed]" in r.content for r in results)


# =============================================================================
# AC: libr reprocess retries failed chunks and updates content + status
# =============================================================================


def test_reprocess_updates_failed_chunk(
    storage: SQLiteStorage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "IMAGE_GENERATE_CAPTIONS", True)
    img = _write_png(tmp_path)

    # First pass: the VLM fails -> a failed chunk.
    Orchestrator(
        storage=storage, embedder=FakeEmbedder(), vision_describer=FailingVisionDescriber()
    ).index_file(img)
    failed = storage.documents_to_reprocess(AssetType.IMAGE, "processing_status", "failed")
    assert str(img) in failed

    # Reprocess: re-ingest the failed document with a now-working VLM.
    good = FakeVisionDescriber(VisionResult(description="recovered caption", transcribed_text="OK"))
    orch = Orchestrator(storage=storage, embedder=FakeEmbedder(), vision_describer=good)
    for path in failed:
        orch.index_file(Path(path))

    chunks = _chunks(storage)
    assert len(chunks) == 1, "reprocess must replace, not duplicate, the chunk"
    assert "recovered caption" in chunks[0]["content"]
    assert '"processing_status": "ok"' in chunks[0]["modality_data"]
    # Nothing left to reprocess.
    assert storage.documents_to_reprocess(AssetType.IMAGE, "processing_status", "failed") == []


def test_documents_to_reprocess_rejects_bad_key(storage: SQLiteStorage) -> None:
    with pytest.raises(ValueError):
        storage.documents_to_reprocess(AssetType.IMAGE, "bad key; DROP", "failed")


# =============================================================================
# AC: PDF with image pages (PDF_OCR_ENABLED=true) -> page chunks with OCR text
# =============================================================================


def test_pdf_ocr_produces_page_chunks_with_ocr_text(
    storage: SQLiteStorage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pypdf = pytest.importorskip("pypdf")
    from librarian.processing.parsers import pdf as pdf_module

    # Build a PDF with a single blank (image-only-style) page: extract_text()
    # returns "", which is exactly what triggers the OCR fallback.
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    pdf_path = tmp_path / "scan.pdf"
    with pdf_path.open("wb") as fh:
        writer.write(fh)

    # OCR deps may be absent; force the parser to take the OCR branch and stub
    # the actual OCR call so the test is hermetic.
    monkeypatch.setenv("PDF_OCR_ENABLED", "true")
    monkeypatch.setattr(pdf_module, "OCR_AVAILABLE", True, raising=False)
    monkeypatch.setattr(
        pdf_module.PDFParser, "_ocr_page", lambda self, p, n: "OCR EXTRACTED TEXT", raising=True
    )

    orch = Orchestrator(storage=storage, embedder=FakeEmbedder())
    result = orch.index_file(pdf_path)

    assert result["chunks"] >= 1
    chunks = _chunks(storage)
    assert all(c["asset_type"] == AssetType.PDF.value for c in chunks)
    assert any("OCR EXTRACTED TEXT" in c["content"] for c in chunks)


def test_registry_threads_pdf_ocr_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pypdf")
    from librarian.processing.parsers import pdf as pdf_module
    from librarian.processing.parsers.registry import ParserRegistry

    # Isolate "registry threads the flag" from OCR-dependency availability:
    # pretend the OCR deps are present so PDFParser(enable_ocr=True) constructs.
    monkeypatch.setenv("PDF_OCR_ENABLED", "true")
    monkeypatch.setattr(pdf_module, "OCR_AVAILABLE", True, raising=False)

    parser, asset_type = ParserRegistry().get_parser(Path("doc.pdf"))
    assert asset_type == AssetType.PDF
    assert getattr(parser, "enable_ocr", False) is True


# =============================================================================
# AC: ENABLE_VISION_EMBEDDINGS retired -> images embed as TEXT; flag warns
# =============================================================================


def test_images_never_use_vision_modality(storage: SQLiteStorage) -> None:
    orch = Orchestrator(storage=storage, embedder=FakeEmbedder())
    assert orch._determine_modality(AssetType.IMAGE) == EmbeddingModality.TEXT


# =============================================================================
# CLI: libr reprocess
# =============================================================================


def _run_reprocess(args: list[str]):
    from typer.testing import CliRunner

    from librarian import cli

    return CliRunner().invoke(cli.app, ["reprocess", *args])


def test_cli_reprocess_rejects_bad_where(clean_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from librarian import cli

    monkeypatch.setattr(cli, "_guard_schema_or_exit", lambda: None)
    result = _run_reprocess(["--where", "nokeyvalue"])
    assert result.exit_code == 1
    assert "key=value" in result.output


def test_cli_reprocess_unknown_asset_type(clean_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from librarian import cli

    monkeypatch.setattr(cli, "_guard_schema_or_exit", lambda: None)
    result = _run_reprocess(["--asset-type", "hologram"])
    assert result.exit_code == 1
    assert "Unknown asset type" in result.output


def test_cli_reprocess_nothing_to_do(clean_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from librarian import cli

    monkeypatch.setattr(cli, "_guard_schema_or_exit", lambda: None)
    result = _run_reprocess([])
    assert result.exit_code == 0
    assert "Nothing to reprocess" in result.output


def test_cli_reprocess_retries_failed_image(
    clean_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from librarian import cli
    from librarian.orchestrator import orchestrator as orch_module
    from librarian.storage.factory import get_storage

    monkeypatch.setattr(config, "IMAGE_GENERATE_CAPTIONS", True)
    monkeypatch.setattr(cli, "_guard_schema_or_exit", lambda: None)

    # Seed a failed image chunk on the global storage the CLI uses.
    img = _write_png(tmp_path)
    Orchestrator(
        storage=get_storage(), embedder=FakeEmbedder(), vision_describer=FailingVisionDescriber()
    ).index_file(img)
    assert get_storage().documents_to_reprocess(AssetType.IMAGE, "processing_status", "failed")

    # Make the file-ingest path the CLI drives use a now-working VLM.
    good = FakeVisionDescriber(VisionResult(description="recovered", transcribed_text=""))
    monkeypatch.setattr(orch_module, "get_vision_describer", lambda: good)

    result = _run_reprocess([])
    assert result.exit_code == 0
    assert "Reprocessed 1" in result.output
    assert (
        get_storage().documents_to_reprocess(AssetType.IMAGE, "processing_status", "failed") == []
    )


def test_enable_vision_embeddings_is_deprecated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_VISION_EMBEDDINGS", "true")
    try:
        with pytest.warns(DeprecationWarning, match="ENABLE_VISION_EMBEDDINGS"):
            importlib.reload(config)
        # Even when explicitly set, the flag is a no-op for the active path.
        assert config.ENABLE_VISION_EMBEDDINGS is True
    finally:
        # Restore the module to its env-clean state for the rest of the session.
        monkeypatch.delenv("ENABLE_VISION_EMBEDDINGS", raising=False)
        importlib.reload(config)
