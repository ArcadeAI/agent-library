# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Agent Library** (package name: `agent-library`, Python module: `librarian`) is a multi-modal knowledge library for AI agents built on Arcade for the Model Context Protocol (MCP). It provides persistent storage with semantic and keyword search for text, code, images, and PDFs.

**Naming note for contributors:** The product is "Agent Library." The Python module path is `librarian` (don't rename — it's the import surface). The MCP server identifies itself as `Librarian` (`MCPApp(name="Librarian")` in `server.py`), which is why exposed tools carry the `Librarian_` prefix. Keep that distinction in mind: prose says "Agent Library"; code-level identifiers say "librarian"/"Librarian".

### Key Technologies
- SQLite with `sqlite-vec` for vector search
- FTS5 with BM25 ranking for full-text search
- Hybrid search combining both approaches with Max Marginal Relevance (MMR)
- Configurable embedding models (local sentence-transformers or OpenAI-compatible API)
- Support for multi-modal assets (text, code, images, PDFs)

### Current Status
- **Multi-Modal Support Complete:** Code, PDF, and image parsing with asset type preservation
- **Parser Registry:** Automatic parser selection based on file extension
- **Database Schema:** Multi-modal columns (asset_type, modality_data) fully implemented
- **Search Integration:** All search tools return asset_type to AI agents
- See `IMPLEMENTATION_STATUS.md` for detailed progress tracking
- See `MULTI_MODAL_LIBRARIAN_DESIGN.md` for complete design specification

## Development Commands

### Setup & Installation
```bash
./setup.sh              # Initial setup
make install            # Install base dependencies
make sync               # Sync dependencies from pyproject.toml

# Install optional multi-modal dependencies
uv pip install -e ".[pdf]"      # PDF processing (pypdf)
uv pip install -e ".[vision]"   # Image processing (Pillow)
uv pip install -e ".[all]"      # All multi-modal features
```

**Optional Dependencies:**
- **PDF Support**: Install `pypdf` to enable PDF parsing
  ```bash
  uv pip install -e ".[pdf]"
  ```
  Without this, PDF files will be skipped during indexing.

- **Image Support**: Install `Pillow` to enable image metadata extraction
  ```bash
  uv pip install -e ".[vision]"
  ```
  Without this, image files will be skipped during indexing.

- **Code Parsing**: Works out-of-the-box with regex-based symbol extraction
  - No additional dependencies required
  - Supports 18+ programming languages

**Testing Multi-Modal Features:**
After installing optional dependencies, verify they work:
```bash
# Test PDF parsing
uv run pytest tests/test_multimodal.py::TestPDFParser -v

# Test image parsing
uv run pytest tests/test_multimodal.py::TestImageParser -v

# Test all multi-modal features
uv run pytest tests/test_multimodal.py -v
```

### Testing
```bash
make test               # Run all tests with coverage
make test-fast          # Run tests without coverage
uv run pytest tests/test_file.py  # Run specific test file
uv run pytest tests/test_file.py::TestClass::test_method  # Run specific test
uv run pytest -m slow   # Run slow tests (loads real embedding models)
```

### Code Quality
```bash
make check              # Run all checks (lint + format-check + typecheck)
make lint               # Run ruff linting
make lint-fix           # Auto-fix linting issues
make format             # Format code with ruff
make typecheck          # Run mypy type checking
```

### Building & Evaluation
```bash
make build              # Build wheel distribution
make evals              # Run Arcade tool evaluations
```

### CLI Usage
```bash
# Multi-modal indexing (auto-detects file types)
libr add ~/notes        # Index markdown files (AssetType.TEXT)
libr add ~/code         # Index source code (AssetType.CODE)
libr add ~/docs         # Index PDFs, images, code, markdown - all types
libr search "query"     # Search across all asset types

# Search results include asset_type
# Example: {"asset_type": "code", "document_path": "main.py", ...}

# Library management
libr list               # List all sources
libr index              # Show library statistics
libr serve stdio        # Start MCP server (stdio)
libr serve http --port 8000  # Start MCP server (HTTP)
```

## Architecture Overview

### Core Pipeline
**File → Parser → Chunker → Embedder → Database → Search**

### Type System
All components use strongly-typed enums:
- `AssetType`: TEXT, CODE, IMAGE, PDF, MULTIMODAL
- `SourceType`: DOCUMENTS, CODEBASE, KNOWLEDGE_BASE, ASSETS, MIXED
- `ChunkingStrategy`: HEADERS, PARAGRAPHS, SENTENCES, FIXED, CODE_BLOCKS, PAGES
- `EmbeddingModality`: TEXT, CODE, VISION, HYBRID
- `ProgrammingLanguage`: 18 supported languages (PYTHON, JAVASCRIPT, etc.)
- `CodeSymbolType`: FUNCTION, CLASS, METHOD, VARIABLE, etc.
- `SearchMode`: HYBRID, SEMANTIC, KEYWORD

### Document Processing (`librarian/processing/`)
1. **Parsers** (`parsers/`):
   - `base.py`: Base parser interface with parse_file() and parse_content()
   - `md.py`: Markdown parser with frontmatter (AssetType.TEXT)
   - `obsidian.py`: Obsidian-flavored markdown with wikilinks
   - `code.py`: Source code parser with symbol extraction (AssetType.CODE)
     - Supports: Python, JavaScript, TypeScript, Go, Rust, Java, C++, etc.
     - Extracts: Classes, functions, methods, variables
   - `pdf.py`: PDF text extraction with page-based chunking (AssetType.PDF)
     - Requires: `pypdf` (install with `uv pip install -e ".[pdf]"`)
   - `image.py`: Image metadata and EXIF extraction (AssetType.IMAGE)
     - Requires: `Pillow` (install with `uv pip install -e ".[vision]"`)
   - `registry.py`: Automatic parser selection by file extension

2. **Chunking** (`transform/`):
   - `chunker.py`: Text chunking with multiple strategies
   - `code.py`: Code-aware chunking by symbols (functions, classes)
   - `pdf.py`: Page-based chunking for PDF documents

3. **Embeddings** (`embed/`):
   - `base.py`: Embedding provider interface
   - `local.py`: Sentence-transformers (local models)
   - `openai.py`: OpenAI-compatible API
   - Note: Currently all assets use TEXT embeddings (all-MiniLM-L6-v2)
   - Future: CodeBERT for code, CLIP for images

### Storage Layer (`librarian/storage/`)
- `protocols.py`: The substrate-agnostic `Storage` protocol bundle (`MetadataStore` / `VectorStore` / `FTSStore` / `StateStore` + `migrate`/`transaction`/write paths)
- `factory.py`: Backend-neutral resolution — `get_storage()` (write paths, migrated), `get_read_storage()` (search), `get_metadata_store()`. Selects the backend from `STORAGE_BACKEND`; SQLite is the default.
- `database.py`: Core SQLite operations with thread-safe connection management
- `vector_store.py`: Vector similarity search using sqlite-vec (preserves asset_type)
- `fts_store.py`: Full-text search using FTS5 with BM25 ranking (preserves asset_type)
- `sqlite_storage.py`: `SQLiteStorage` — the default concrete `Storage` bundle
- `postgres/`: `PostgresStorage` — pgvector-backed `Storage` bundle (DEV-472). Mirrors the SQLite v0.14 schema (pgvector `vector(dim)` columns + a `tsvector` generated column for FTS + `sync_state`). Selected via `STORAGE_BACKEND=postgres`; requires the `postgres` extra (`uv pip install -e ".[postgres]"`) and `POSTGRES_DSN`. Same observable `libr add`/`libr search` behavior as SQLite; atomic content+cursor writes stay single-substrate.

Multi-Modal Schema (Complete):
- `documents` table: Stores file metadata with `asset_type` and `modality_data` columns
- `chunks` table: Stores chunks with `asset_type` column for filtering
- `chunk_embeddings` virtual table: sqlite-vec index for vector search
- `chunks_fts` virtual table: FTS5 index for full-text search

Asset Type Preservation:
- All search operations (vector, FTS, hybrid) preserve and return asset_type
- Search results include asset_type for filtering and display
- Database queries JOIN documents table to retrieve asset_type

### Retrieval (`librarian/retrieval/`)
- `search.py`: Hybrid search combining vector + FTS with configurable weighting
- MMR algorithm for diverse result selection

### Indexing (`librarian/indexing.py`)
Coordinates the full pipeline: parsing → chunking → embedding → storage
Handles modification time tracking for incremental updates

### Interfaces
- `server.py`: MCP server with tools for agents
- `cli.py`: Command-line interface wrapping server functions

## Configuration

All settings in `librarian/config.py` can be overridden via environment variables:

### Core Settings
- `DATABASE_PATH`: SQLite database location (default: `~/.librarian/index.db`)
- `SOURCES_CONFIG_PATH`: Sources configuration (default: `~/.librarian/sources.json`)
- `DOCUMENTS_PATH`: Default document directory (default: `./documents`)

### Storage Backend (DEV-472)
- `STORAGE_BACKEND`: `sqlite` (default) or `postgres`
- `POSTGRES_DSN`: Postgres connection string (libpq DSN or `postgres://` URL); `DATABASE_URL` is accepted as a fallback. Required when `STORAGE_BACKEND=postgres`.
- `POSTGRES_SCHEMA`: schema for librarian-owned tables (default: `public`)
- Tests: set `TEST_POSTGRES_DSN` to run the parameterized storage suite (`tests/test_storage_contract.py`) against Postgres; it's skipped when unset. The pgvector extension is pinned to `public`. Quick local DB: `docker run -d -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=librarian -p 5432:5432 pgvector/pgvector:pg16`.

### Text Embedding
- `EMBEDDING_PROVIDER`: "local" or "openai"
- `EMBEDDING_MODEL`: Local model name (default: `all-MiniLM-L6-v2`)
- `OPENAI_API_BASE`: OpenAI-compatible API endpoint
- `OPENAI_EMBEDDING_MODEL`: API model name

### Code Processing (Optional)
- `ENABLE_CODE_EMBEDDINGS`: Enable code-specific embeddings
- `CODE_EMBEDDING_MODEL`: CodeBERT model
- `CODE_CHUNK_STRATEGY`: "code_blocks" or "fixed"
- `CODE_CONTEXT_LINES`: Lines of context around code

### Vision Processing (Optional)
The v0.14 vision pipeline is **VLM-text only** (issue #53, strategy "Option C / Hybrid"):
a hosted vision-language model describes/transcribes an image in one call, and that
text is embedded in the ordinary TEXT space. CLIP image embeddings are retired.
- `IMAGE_GENERATE_CAPTIONS`: Describe/transcribe images with a VLM (default off).
  When off, images are indexed as metadata-only chunks tagged
  `processing_status='uncaptioned'` (back-fillable later — see `libr reprocess`).
- `VLM_PROVIDER`: `"openai"` (default) or `"anthropic"` (case-insensitive).
- `VLM_MODEL`: Vision model id (default `gpt-4o` / `claude-3-5-sonnet-*` per provider).
- `VLM_MAX_TOKENS`: Max tokens for the VLM response (default 1024).
- `ENABLE_VISION_EMBEDDINGS`: **DEPRECATED / no-op.** Retired CLIP flag; setting it
  emits a `DeprecationWarning` and changes nothing (the `vec_chunks_vision` table /
  `EmbeddingModality.VISION` remain dormant for a possible v1.x re-introduction).
- Requires the `vlm` extra: `uv pip install -e ".[vlm]"`.

### PDF Processing (Optional)
- `ENABLE_PDF_PROCESSING`: Enable PDF parsing
- `PDF_OCR_ENABLED`: Enable OCR for image-based PDF pages (needs the `ocr` extra:
  `pytesseract` + `pdf2image` + a system `tesseract`/`poppler`). When requested but
  the deps are absent, PDFs still index their text and are tagged
  `processing_status='ocr_unavailable'` for later retry.
- `PDF_CHUNK_STRATEGY`: "pages" or "sections"

### Reprocessing failed/uncaptioned chunks
`libr reprocess --asset-type <type> --where <key>=<value>` re-ingests documents whose
chunks carry a matching `modality_data` status (default `processing_status=failed`),
updating content + status on success. Examples:
- `libr reprocess` — retry images whose VLM caption failed.
- `libr reprocess --where processing_status=uncaptioned` — back-fill captions after
  enabling `IMAGE_GENERATE_CAPTIONS`.
- `libr reprocess --asset-type pdf --where processing_status=ocr_unavailable` — retry
  scanned PDFs after installing the OCR deps.
`processing_status` values are the `ProcessingStatus` enum in `librarian/types.py`
(`ok`, `failed`, `uncaptioned`, `ocr_unavailable`, `unsupported`).

### Search
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: Text chunking parameters
- `HYBRID_ALPHA`: Vector vs keyword weight (0=keyword only, 1=vector only)
- `MMR_LAMBDA`: Diversity parameter (0=max diversity, 1=max relevance)
- `SEARCH_LIMIT`: Default results limit

### Codebase Management
- `CODEBASE_AUTO_DETECT`: Auto-detect languages and framework
- `CODEBASE_INDEX_TESTS`: Include test files when indexing
- `CODEBASE_MAX_FILE_SIZE_KB`: Max file size to index

## Key Design Patterns

1. **Singleton Pattern**: Global instances for database, embedder, indexing service
   - `get_database()`, `get_embedder()`, `get_indexing_service()`

2. **Provider Pattern**: Swappable embedding backends
   - `EmbeddingProvider` base class
   - `LocalEmbeddingProvider` and `OpenAIEmbeddingProvider` implementations

3. **Registry Pattern**: Automatic parser selection (planned)
   - `ParserRegistry` routes files to appropriate parser by extension

4. **Thread Safety**: Database connections use thread-local storage
   - Each thread gets its own connection via `_local` attribute
   - Operations protected with `threading.RLock()`

5. **Enum-First Design**: All parameters use typed enums for safety
   - Prevents invalid values at runtime
   - Better IDE autocomplete and type checking

## Multi-Modal Architecture

### Parser Registry
The parser registry automatically selects the appropriate parser based on file extension:

```python
from librarian.processing.parsers.registry import get_parser_for_file
from pathlib import Path

# Automatic parser selection
parser, asset_type = get_parser_for_file(Path("example.py"))
# Returns: (CodeParser(language=ProgrammingLanguage.PYTHON), AssetType.CODE)

parser, asset_type = get_parser_for_file(Path("document.pdf"))
# Returns: (PDFParser(), AssetType.PDF)

parser, asset_type = get_parser_for_file(Path("diagram.png"))
# Returns: (ImageParser(), AssetType.IMAGE)
```

Supported file types:
- **Code**: .py, .js, .ts, .jsx, .tsx, .go, .rs, .java, .cpp, .c, .h, .cs, .rb, .php, .swift, .kt, .scala
- **PDF**: .pdf (requires `pypdf`)
- **Images**: .png, .jpg, .jpeg, .gif, .webp (requires `Pillow`)
- **Text**: .md, .txt

### Using Parsers Directly
```python
from librarian.processing.parsers.code import CodeParser
from librarian.types import ProgrammingLanguage

# Parse Python code
parser = CodeParser(language=ProgrammingLanguage.PYTHON)
parsed = parser.parse_file(Path("script.py"))

print(f"Asset Type: {parsed.asset_type}")  # AssetType.CODE
print(f"Symbols: {len(parsed.metadata['symbols'])}")  # Classes, functions, methods

# Symbols include:
for symbol in parsed.metadata['symbols']:
    print(f"  {symbol['type']}: {symbol['name']} (line {symbol['line_start']})")
```

### Multi-Modal Indexing
The indexing service automatically detects file types and routes to the correct parser:

```python
from librarian.indexing import get_indexing_service

service = get_indexing_service()

# Index any file type - parser auto-detected
result = service.index_file(Path("example.py"))
# Returns: {"status": "created", "chunks": 6, "asset_type": "code"}

result = service.index_file(Path("document.pdf"))
# Returns: {"status": "created", "chunks": 10, "asset_type": "pdf"}

result = service.index_file(Path("diagram.png"))
# Returns: {"status": "created", "chunks": 1, "asset_type": "image"}
```

### Multi-Modal Search
All search tools return asset_type in results:

```python
from librarian.retrieval.search import HybridSearcher
from librarian.processing.embed import get_embedder

searcher = HybridSearcher(embedder=get_embedder())

# Hybrid search (vector + keyword)
results = searcher.search("calculator", limit=10)
for r in results:
    print(f"[{r.asset_type.value}] {Path(r.document_path).name}")
    # [code] example.py
    # [pdf] math_tutorial.pdf
    # [text] calculator_notes.md

# Vector-only search (semantic similarity)
results = searcher.vector_search("authentication code", limit=5)

# Keyword-only search (exact matching)
results = searcher.keyword_search("Calculator", limit=5)
```

### MCP Tool Integration
AI agents receive asset_type in all search results via the unified `search_library` tool:

```python
# search_library MCP tool response (all modes: hybrid, semantic, keyword)
{
    "chunk_id": 42,
    "document_path": "/path/to/example.py",
    "content": "class Calculator...",
    "asset_type": "code",  # ← AI agents see this
    "score": 0.87
}

# search_library with mode="semantic" and asset_type="pdf"
{
    "asset_type": "pdf",
    "document_path": "/path/to/document.pdf",
    ...
}

# search_library with mode="keyword"
{
    "asset_type": "image",
    "document_path": "/path/to/diagram.png",
    ...
}
```

## Important Notes

### Type Checking
- mypy is configured to exclude `cli.py` (wraps Arcade-decorated functions)
- Type ignores used for optional provider dependencies
- All new code should use enums from `librarian/types.py`

### Testing
- Tests marked `@pytest.mark.slow` load real embedding models (skip by default)
- Use `pytest -m slow` to run them explicitly
- Test fixtures in `conftest.py` provide mock database and embedder
- All tests must pass before merging features

### Linting
- UP007/UP045 ignored for typer CLI compatibility (use `Optional[]` not `| None`)
- Security warnings S603/S607 ignored in `cli.py` for controlled subprocess calls

### Database Schema
Multi-modal schema is fully implemented:
- `documents.asset_type`: Stores file type (text, code, pdf, image)
- `documents.modality_data`: JSON field for type-specific metadata (language, dimensions, etc.)
- All search queries JOIN documents table to preserve asset_type
- Vector and FTS search results include asset_type from documents table

Schema ensures asset type is preserved throughout the entire search pipeline.

## Extending the System

### Adding a New Parser
1. Create `librarian/processing/parsers/myparser.py`:
```python
from librarian.processing.parsers.base import BaseParser
from librarian.types import AssetType, ParsedDocument

class MyParser(BaseParser):
    asset_type = AssetType.MY_TYPE
    supported_extensions = {".ext"}

    def parse_file(self, file_path: Path) -> ParsedDocument:
        # Parse file and return ParsedDocument with asset_type set
        pass
```

2. Register in `ParserRegistry` (when implemented)
3. Add tests in `tests/test_myparser.py`

### Adding a New MCP Tool
1. Add to `librarian/server.py`:
```python
from arcade_mcp_server import Context, MCPApp

@app.tool
async def my_tool(
    context: Context,
    param: Annotated[AssetType, "Description"],
) -> dict[str, Any]:
    """Tool description for AI."""
    await context.log.info(f"Starting my_tool with {param}")

    try:
        await context.progress(0.0, "Initializing...")
        # Do work
        await context.progress(1.0, "Complete")
        return {"result": "success"}
    except Exception as e:
        await context.log.error(f"Error: {e}")
        return {"error": str(e)}
```

2. Add tests in `tests/test_tools.py`
3. Use Context for logging and progress reporting

### Adding a New Enum
1. Add to `librarian/types.py` in appropriate section
2. Use `str, Enum` base for string enums
3. Document in this file under Type System section

## Development Workflow

1. **Feature Development**:
   - Check `IMPLEMENTATION_STATUS.md` for what's needed
   - Review `MULTI_MODAL_LIBRARIAN_DESIGN.md` for design
   - Write tests first (TDD)
   - Implement feature
   - Run `make check` and `make test`

2. **Code Review Checklist**:
   - All tests passing
   - Type hints on all functions
   - Enums used instead of strings
   - Logging via Context in tools
   - Documentation updated
   - No emojis in code

3. **Testing Strategy**:
   - Unit tests for parsers, chunkers, embedders
   - Integration tests for tools
   - Use fixtures from `conftest.py`
   - Mock expensive operations (embedding models)

## Project Structure

```
librarian/
├── types.py           # All enums and data classes
├── config.py          # Configuration management
├── indexing.py        # Document indexing service
├── storage/
│   ├── database.py    # SQLite operations
│   ├── vector_store.py  # Vector search
│   └── fts_store.py     # Full-text search
├── processing/
│   ├── embed/         # Embedding providers
│   ├── parsers/       # Document parsers
│   └── transform/     # Text chunking
├── retrieval/
│   └── search.py      # Hybrid search + MMR
├── sources/           # Source management (planned)
│   ├── codebase.py    # Codebase management
│   └── ignore.py      # .librarianignore support
├── server.py          # MCP server and tool definitions
├── cli.py             # Command-line interface
└── utils/
    └── timeframe.py   # Time filter utilities
```

## Resources

- [Arcade.dev](https://arcade.dev) - MCP server framework
- [Arcade Documentation](https://docs.arcade.dev) - Integration guides
- [MULTI_MODAL_LIBRARIAN_DESIGN.md](./MULTI_MODAL_LIBRARIAN_DESIGN.md) - Complete design
- [IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md) - Current progress

## Quick Reference

### Common Tasks
```bash
# Run tests for specific component
uv run pytest tests/test_parser.py -v

# Type check specific file
uv run mypy librarian/processing/parsers/md.py

# Format and lint
make format && make lint

# Run single test with output
uv run pytest tests/test_tools.py::TestSearchTools::test_search_library -v -s
```

### Environment Variables for Testing
```bash
export DATABASE_PATH=":memory:"  # Use in-memory DB for tests
export EMBEDDING_PROVIDER="local"
export CHUNK_SIZE="256"
```

### Debugging
- Use `await context.log.debug()` in tools for detailed logging
- Check `~/.librarian/index.db` with SQLite browser for database inspection
- Use `-v -s` flags with pytest for verbose output

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Contact

- Email: <contact@arcade.dev>
- Website: [arcade.dev](https://arcade.dev)
