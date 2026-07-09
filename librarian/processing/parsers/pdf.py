"""
PDF parser for extracting text from PDF documents.

Uses pypdf for text extraction with optional OCR support via pytesseract.
"""

import logging
from pathlib import Path
from typing import Any

from librarian.processing.parsers.base import BaseParser, safe_read_bytes
from librarian.types import AssetType, ParsedDocument, Section

logger = logging.getLogger(__name__)

try:
    from pypdf import PdfReader

    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import pytesseract
    from pdf2image import convert_from_path

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class PDFParser(BaseParser):
    """Parser for PDF documents."""

    def __init__(self, enable_ocr: bool = False) -> None:
        """
        Initialize PDF parser.

        Args:
            enable_ocr: Whether to use OCR for image-based PDFs.
        """
        if not PYPDF_AVAILABLE:
            raise ImportError("pypdf is required for PDF parsing. Install with: pip install pypdf")

        # OCR was requested vs. actually available: when the OCR deps are absent
        # we still parse the PDF's text (rather than raising, which would drop
        # the whole document) but record ``ocr_unavailable`` so a scanned PDF that
        # produced no text is retryable via ``libr reprocess`` once deps land.
        self.ocr_requested = enable_ocr
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        self.ocr_unavailable = enable_ocr and not OCR_AVAILABLE

    def parse_content(self, content: str, path: str = "") -> ParsedDocument:
        """
        Parse PDF content from a string.

        Note: PDFs require binary data and cannot be parsed from string content.
        This method is provided to satisfy the BaseParser interface.

        Args:
            content: The content string (not supported for PDFs).
            path: Optional path for reference.

        Raises:
            NotImplementedError: PDFs must be parsed from file path.
        """
        raise NotImplementedError(
            "PDF parser requires a file path. Use parse_file() instead of parse_content()."
        )

    def parse_file(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ParsedDocument with extracted text and metadata.

        Raises:
            FileNotFoundError: If file doesn't exist.
            FileReadTimeoutError: If file read times out (e.g., iCloud).
        """
        # Convert to Path if string
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # Read bytes with timeout protection (handles cloud/network filesystems)
        pdf_bytes = safe_read_bytes(file_path)

        import io

        reader = PdfReader(io.BytesIO(pdf_bytes))

        # Extract metadata
        metadata: dict[str, Any] = {
            "file_type": "pdf",
            "page_count": len(reader.pages),
        }

        # Try to extract PDF metadata
        if reader.metadata:
            pdf_meta = reader.metadata
            if pdf_meta.title:
                metadata["pdf_title"] = pdf_meta.title
            if pdf_meta.author:
                metadata["pdf_author"] = pdf_meta.author
            if pdf_meta.subject:
                metadata["pdf_subject"] = pdf_meta.subject
            if pdf_meta.creator:
                metadata["pdf_creator"] = pdf_meta.creator

        # Extract text from all pages
        pages_text = []
        sections = []

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text()

                # If page is empty and OCR is enabled, try OCR
                if not page_text.strip() and self.enable_ocr:
                    page_text = self._ocr_page(file_path, page_num)

                pages_text.append(page_text)

                # Create section for each page
                if page_text.strip():
                    sections.append(
                        Section(
                            title=f"Page {page_num}",
                            level=1,
                            content=page_text,
                            start_pos=(page_num - 1) * 4000,  # Rough estimate
                            end_pos=page_num * 4000,
                        )
                    )

            except Exception as e:
                # Log error but continue processing other pages
                import logging

                logging.warning(f"Error extracting text from page {page_num}: {e}")
                pages_text.append("")

        # Combine all pages
        full_text = "\n\n".join(pages_text)

        # Determine title (use PDF metadata or filename)
        title = metadata.get("pdf_title") or file_path.stem

        modality_data: dict[str, Any] = {"page_count": len(reader.pages)}
        # Record a retryable processing_status so a scanned/image-only PDF that
        # yielded no text isn't silently indexed empty -- ``libr reprocess`` can
        # pick it up once OCR deps are installed (ocr_unavailable) or the source
        # is fixed (failed). ``ProcessingStatus`` values are used as bare strings
        # here to avoid a parser -> types cross-layer import churn.
        if not full_text.strip():
            modality_data["processing_status"] = (
                "ocr_unavailable" if self.ocr_unavailable else "failed"
            )
        elif self.ocr_unavailable:
            # Text extracted, but OCR was requested and unavailable: any image-only
            # pages were skipped, so flag for a later OCR pass.
            modality_data["processing_status"] = "ocr_unavailable"

        return ParsedDocument(
            path=str(file_path),
            title=title,
            content=full_text,
            metadata=metadata,
            sections=sections,
            raw_content=full_text,
            asset_type=AssetType.PDF,
            modality_data=modality_data,
        )

    def _ocr_page(self, pdf_path: Path, page_num: int) -> str:
        """
        Perform OCR on a single page.

        Args:
            pdf_path: Path to PDF file.
            page_num: Page number (1-indexed).

        Returns:
            Extracted text via OCR.
        """
        if not OCR_AVAILABLE:
            return ""

        try:
            # Convert page to image
            images = convert_from_path(
                str(pdf_path),
                first_page=page_num,
                last_page=page_num,
                dpi=200,
            )

            if not images:
                return ""

            # Perform OCR
            return str(pytesseract.image_to_string(images[0]))

        except Exception as e:
            import logging

            logging.warning(f"OCR failed for page {page_num}: {e}")
            return ""
