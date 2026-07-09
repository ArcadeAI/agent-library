"""
Parser registry for automatic parser selection.

Routes files to the appropriate parser based on file extension.
Supports text, code, PDF, and image formats.
"""

from pathlib import Path

from librarian.processing.parsers.base import BaseParser
from librarian.processing.parsers.md import MarkdownParser
from librarian.processing.parsers.obsidian import ObsidianParser
from librarian.types import AssetType, ProgrammingLanguage


class ParserRegistry:
    """Registry for mapping file extensions to parsers."""

    def __init__(self) -> None:
        """Initialize the parser registry with default parsers."""
        self._parsers: dict[str, tuple[type[BaseParser], AssetType]] = {}
        self._language_map: dict[str, ProgrammingLanguage] = {}
        self._register_default_parsers()

    def _register_default_parsers(self) -> None:
        """Register default parsers for common file types."""
        # Text formats
        self.register(".md", MarkdownParser, AssetType.TEXT)
        self.register(".markdown", MarkdownParser, AssetType.TEXT)
        self.register(".txt", MarkdownParser, AssetType.TEXT)

        # Obsidian markdown
        # Note: Obsidian files are detected based on vault structure,
        # but we provide this for explicit registration
        self.register(".obsidian.md", ObsidianParser, AssetType.TEXT)

        # Code formats - will be handled by CodeParser when implemented
        # For now, we'll import and register the CodeParser later
        code_extensions = {
            ".py": ProgrammingLanguage.PYTHON,
            ".js": ProgrammingLanguage.JAVASCRIPT,
            ".ts": ProgrammingLanguage.TYPESCRIPT,
            ".jsx": ProgrammingLanguage.JAVASCRIPT,
            ".tsx": ProgrammingLanguage.TYPESCRIPT,
            ".go": ProgrammingLanguage.GO,
            ".rs": ProgrammingLanguage.RUST,
            ".java": ProgrammingLanguage.JAVA,
            ".cpp": ProgrammingLanguage.CPP,
            ".cc": ProgrammingLanguage.CPP,
            ".cxx": ProgrammingLanguage.CPP,
            ".c": ProgrammingLanguage.C,
            ".h": ProgrammingLanguage.C,
            ".hpp": ProgrammingLanguage.CPP,
            ".rb": ProgrammingLanguage.RUBY,
            ".php": ProgrammingLanguage.PHP,
            ".swift": ProgrammingLanguage.SWIFT,
            ".kt": ProgrammingLanguage.KOTLIN,
            ".cs": ProgrammingLanguage.CSHARP,
            ".html": ProgrammingLanguage.HTML,
            ".css": ProgrammingLanguage.CSS,
            ".sql": ProgrammingLanguage.SQL,
            ".sh": ProgrammingLanguage.SHELL,
            ".bash": ProgrammingLanguage.SHELL,
            ".zsh": ProgrammingLanguage.SHELL,
        }
        self._language_map = code_extensions

        # PDF formats
        try:
            from librarian.processing.parsers.pdf import PDFParser

            self.register(".pdf", PDFParser, AssetType.PDF)  # type: ignore[type-abstract]
        except ImportError:
            # PDF dependencies not installed
            pass

        # Image formats
        try:
            from librarian.processing.parsers.image import ImageParser

            image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"]
            for ext in image_extensions:
                self.register(ext, ImageParser, AssetType.IMAGE)  # type: ignore[type-abstract]
        except ImportError:
            # PIL not installed
            pass

    def register(
        self,
        extension: str,
        parser_class: type[BaseParser],
        asset_type: AssetType,
    ) -> None:
        """
        Register a parser for a file extension.

        Args:
            extension: File extension (including leading dot).
            parser_class: Parser class to use.
            asset_type: Asset type the parser handles.
        """
        self._parsers[extension.lower()] = (parser_class, asset_type)

    def get_parser(self, file_path: Path) -> tuple[BaseParser | None, AssetType]:
        """
        Get the appropriate parser for a file.

        Args:
            file_path: Path to the file.

        Returns:
            Tuple of (parser instance, asset type), or (None, AssetType.TEXT) if no parser found.
        """
        # Get file extension
        extension = file_path.suffix.lower()

        # Check for Obsidian vault
        if self._is_obsidian_vault(file_path):
            return ObsidianParser(), AssetType.TEXT

        # Check registry
        if extension in self._parsers:
            parser_class, asset_type = self._parsers[extension]

            # Special handling for ImageParser to pass OCR config.
            # Re-evaluate environment variables to support dynamic configuration.
            if asset_type == AssetType.IMAGE:
                import os

                from librarian.config import safe_bool

                enable_ocr = safe_bool(os.getenv("ENABLE_OCR"), False)
                try:
                    return parser_class(enable_ocr=enable_ocr), asset_type  # type: ignore[call-arg]
                except ImportError:
                    # Pillow not installed: the VLM caption path (which reads raw
                    # bytes) can still handle the image, so preserve the asset
                    # type and let the orchestrator decide.
                    return None, AssetType.IMAGE

            # PDFs: thread the PDF_OCR_ENABLED flag through so image-only pages
            # get OCR'd. Re-read the env so tests/config changes take effect.
            # PDFParser degrades to text-only (recording ``ocr_unavailable``) when
            # OCR deps are missing rather than raising, so an ImportError here
            # means pypdf itself is absent -> genuinely unsupported.
            if asset_type == AssetType.PDF:
                import os

                from librarian.config import safe_bool

                enable_ocr = safe_bool(os.getenv("PDF_OCR_ENABLED"), False)
                try:
                    return parser_class(enable_ocr=enable_ocr), asset_type  # type: ignore[call-arg]
                except ImportError:
                    return None, AssetType.PDF

            return parser_class(), asset_type

        # Check if it's a code file we recognize
        if extension in self._language_map:
            # Import CodeParser here to avoid circular import
            try:
                from librarian.processing.parsers.code import CodeParser

                return CodeParser(), AssetType.CODE  # type: ignore[abstract]
            except ImportError:
                # CodeParser not yet implemented, treat as text
                return MarkdownParser(), AssetType.TEXT

        # Default to markdown parser for unknown text files
        return None, AssetType.TEXT

    def get_language_for_extension(self, extension: str) -> ProgrammingLanguage | None:
        """
        Get the programming language for a file extension.

        Args:
            extension: File extension (including leading dot).

        Returns:
            Programming language enum, or None if not recognized.
        """
        return self._language_map.get(extension.lower())

    def get_asset_type(self, file_path: Path) -> AssetType:
        """
        Determine the asset type for a file.

        Args:
            file_path: Path to the file.

        Returns:
            Asset type enum.
        """
        _, asset_type = self.get_parser(file_path)
        return asset_type

    def _is_obsidian_vault(self, file_path: Path) -> bool:
        """
        Check if a file is part of an Obsidian vault.

        Args:
            file_path: Path to the file.

        Returns:
            True if the file is in an Obsidian vault.
        """
        # Check if there's a .obsidian directory in any parent
        current = file_path.parent
        while current != current.parent:
            if (current / ".obsidian").is_dir():
                return True
            current = current.parent
        return False

    def supports_file(self, file_path: Path) -> bool:
        """
        Check if a file is supported by any registered parser.

        Args:
            file_path: Path to the file.

        Returns:
            True if the file can be parsed.
        """
        parser, _ = self.get_parser(file_path)
        return parser is not None

    def get_supported_extensions(self) -> set[str]:
        """
        Get all file extensions supported by registered parsers.

        Returns:
            Set of file extensions (including leading dot).
        """
        extensions = set(self._parsers.keys())
        # Add language extensions (code files)
        extensions.update(self._language_map.keys())
        return extensions


# Global registry instance
_registry: ParserRegistry | None = None


def get_registry() -> ParserRegistry:
    """
    Get the global parser registry instance.

    Returns:
        The global ParserRegistry instance.
    """
    global _registry
    if _registry is None:
        _registry = ParserRegistry()
    return _registry


def get_parser_for_file(file_path: Path) -> tuple[BaseParser | None, AssetType]:
    """
    Get the appropriate parser for a file using the global registry.

    Args:
        file_path: Path to the file.

    Returns:
        Tuple of (parser instance, asset type).
    """
    return get_registry().get_parser(file_path)


def register_parser(
    extension: str,
    parser_class: type[BaseParser],
    asset_type: AssetType,
) -> None:
    """
    Register a custom parser in the global registry.

    Args:
        extension: File extension (including leading dot).
        parser_class: Parser class to use.
        asset_type: Asset type the parser handles.
    """
    get_registry().register(extension, parser_class, asset_type)
