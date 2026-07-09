"""
Configuration management for librarian multi-modal support.

Centralized configuration with environment variable overrides and
sensible defaults for all system parameters.
"""

import os
from pathlib import Path

from librarian.types import AssetType, ChunkingStrategy, EmbeddingModality


def safe_int(value: str | None, default: int) -> int:
    """Convert value to int, returning default on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: str | None, default: float) -> float:
    """Convert value to float, returning default on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_bool(value: str | None, default: bool) -> bool:
    """Convert value to bool, returning default on failure."""
    if value is None:
        return default
    return str(value).lower() in ("true", "1", "yes", "on")


# =============================================================================
# Path Configuration
# =============================================================================

DOCUMENTS_PATH = os.path.abspath(os.path.expanduser(os.getenv("DOCUMENTS_PATH", "./documents")))

# =============================================================================
# Indexing Skip Defaults
# =============================================================================

# Directories that are skipped during indexing unless explicitly overridden
# (via --force-include or a .librariantrack entry). Override the default set
# with INDEX_SKIP_DIRS as a comma-separated list.
_DEFAULT_INDEX_SKIP_DIRS = (
    "__pycache__,.git,.svn,.hg,node_modules,.venv,venv,"
    ".pytest_cache,.mypy_cache,.ruff_cache,__MACOSX,.DS_Store"
)
INDEX_SKIP_DIRS: frozenset[str] = frozenset(
    d.strip()
    for d in os.getenv("INDEX_SKIP_DIRS", _DEFAULT_INDEX_SKIP_DIRS).split(",")
    if d.strip()
)

# File extensions that are skipped during indexing (binary / archive / media).
# Override with INDEX_SKIP_EXTENSIONS as a comma-separated list (include the dot).
_DEFAULT_INDEX_SKIP_EXTENSIONS = (
    ".exe,.bin,.dll,.so,.dylib,.a,.o,"
    ".dmg,.iso,.img,.app,.pkg,"
    ".zip,.tar,.gz,.bz2,.xz,.7z,.rar,"
    ".pyc,.pyo,.pyd,"
    ".lock,.log,.tmp,.temp,.cache,"
    ".mp4,.mp3,.wav,.avi,.mov,.flac,"
    ".ttf,.otf,.woff,.woff2"
)
INDEX_SKIP_EXTENSIONS: frozenset[str] = frozenset(
    e.strip().lower()
    for e in os.getenv("INDEX_SKIP_EXTENSIONS", _DEFAULT_INDEX_SKIP_EXTENSIONS).split(",")
    if e.strip()
)

DATABASE_PATH = os.path.abspath(
    os.path.expanduser(os.getenv("DATABASE_PATH", "~/.librarian/index.db"))
)

SOURCES_CONFIG_PATH = os.path.abspath(
    os.path.expanduser(os.getenv("SOURCES_CONFIG_PATH", "~/.librarian/sources.json"))
)

# =============================================================================
# Storage Backend Configuration
# =============================================================================

# Which storage substrate the Storage bundle resolves to. "sqlite" is the
# default for OSS / local use (zero external dependencies); "postgres" selects
# the pgvector-backed PostgresStorage (requires the `postgres` extra).
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "sqlite").strip().lower()
if STORAGE_BACKEND not in {"sqlite", "postgres"}:
    raise ValueError(
        f"Invalid STORAGE_BACKEND {STORAGE_BACKEND!r}; expected one of: postgres, sqlite."
    )

# Postgres connection string (libpq DSN or postgres:// URL). DATABASE_URL is
# accepted as a fallback so the backend works out of the box on common PaaS.
POSTGRES_DSN = os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL")

# Schema the librarian-owned tables live in. Overridable so test runs (and
# multi-tenant deployments) can isolate their tables on a shared database.
POSTGRES_SCHEMA = os.getenv("POSTGRES_SCHEMA", "public").strip()

# Connection / query timeouts (Postgres only). These keep an unreachable host or
# a pathological query from hanging a worker thread or pinning a server
# connection indefinitely. ``0`` disables the corresponding server-side timeout.
POSTGRES_CONNECT_TIMEOUT = safe_int(os.getenv("POSTGRES_CONNECT_TIMEOUT"), 10)
POSTGRES_STATEMENT_TIMEOUT_MS = safe_int(os.getenv("POSTGRES_STATEMENT_TIMEOUT_MS"), 30_000)
POSTGRES_IDLE_TX_TIMEOUT_MS = safe_int(os.getenv("POSTGRES_IDLE_TX_TIMEOUT_MS"), 60_000)

# Bounded connection pool (Postgres only). A threaded MCP/HTTP server checks
# connections out of this pool instead of opening one per thread, so concurrent
# request fan-out can't drift toward the server's ``max_connections``.
# ``POSTGRES_POOL_MAX_SIZE`` caps live connections; ``POSTGRES_POOL_MIN_SIZE``
# keeps a warm floor; ``POSTGRES_POOL_TIMEOUT`` bounds how long a checkout waits
# for a free connection before raising (so an exhausted pool fails fast rather
# than blocking forever).
POSTGRES_POOL_MIN_SIZE = safe_int(os.getenv("POSTGRES_POOL_MIN_SIZE"), 1)
POSTGRES_POOL_MAX_SIZE = safe_int(os.getenv("POSTGRES_POOL_MAX_SIZE"), 10)
POSTGRES_POOL_TIMEOUT = safe_int(os.getenv("POSTGRES_POOL_TIMEOUT"), 30)

# Text-search configuration (regconfig) used for the generated ``content_tsv``
# column and the FTS query/headline functions. Pinned at table-creation time;
# changing it on a populated database requires a schema rebuild.
POSTGRES_FTS_LANGUAGE = os.getenv("POSTGRES_FTS_LANGUAGE", "english").strip()

# =============================================================================
# Text Embedding Configuration
# =============================================================================

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = safe_int(os.getenv("EMBEDDING_DIMENSION"), 384)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:7171/v1")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "qwen3-embedding-06b")
OPENAI_EMBEDDING_DIMENSION = safe_int(os.getenv("OPENAI_EMBEDDING_DIMENSION"), 1024)
OPENAI_EMBEDDING_BATCH_SIZE = safe_int(os.getenv("OPENAI_EMBEDDING_BATCH_SIZE"), 64)

EMBEDDING_QUERY_INSTRUCTION = os.getenv(
    "EMBEDDING_QUERY_INSTRUCTION",
    "Given a query, return relevant information from documents.",
)


def get_effective_embedding_dimension() -> int:
    """Get the text embedding dimension based on the configured provider."""
    if EMBEDDING_PROVIDER == "openai":
        return OPENAI_EMBEDDING_DIMENSION
    return EMBEDDING_DIMENSION


# =============================================================================
# Code Embedding Configuration
# =============================================================================

# Code embeddings enabled by default for better code search using CodeBERT
ENABLE_CODE_EMBEDDINGS = safe_bool(os.getenv("ENABLE_CODE_EMBEDDINGS"), True)
CODE_EMBEDDING_MODEL = os.getenv("CODE_EMBEDDING_MODEL", "microsoft/codebert-base")
CODE_EMBEDDING_DIMENSION = safe_int(os.getenv("CODE_EMBEDDING_DIMENSION"), 768)
CODE_EMBEDDING_PROVIDER = os.getenv("CODE_EMBEDDING_PROVIDER", "local")

# =============================================================================
# Vision Embedding Configuration
# =============================================================================

# Vision embeddings enabled by default for image search using CLIP
ENABLE_VISION_EMBEDDINGS = safe_bool(os.getenv("ENABLE_VISION_EMBEDDINGS"), True)
# Use sentence-transformers CLIP model name (not HuggingFace format)
VISION_EMBEDDING_MODEL = os.getenv("VISION_EMBEDDING_MODEL", "clip-ViT-B-32")
VISION_EMBEDDING_DIMENSION = safe_int(os.getenv("VISION_EMBEDDING_DIMENSION"), 512)

# =============================================================================
# OCR Configuration
# =============================================================================

ENABLE_OCR = safe_bool(os.getenv("ENABLE_OCR"), True)
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")
OCR_CONFIG = os.getenv("OCR_CONFIG", "--psm 3")  # Page segmentation mode
OCR_MIN_CONFIDENCE = safe_int(os.getenv("OCR_MIN_CONFIDENCE"), 0)  # 0-100, 0 = no filtering

# =============================================================================
# Asset Type Configuration
# =============================================================================

DEFAULT_ASSET_TYPES_STR = os.getenv("DEFAULT_ASSET_TYPES", "text,code")
DEFAULT_ASSET_TYPES = [
    AssetType(t.strip()) for t in DEFAULT_ASSET_TYPES_STR.split(",") if t.strip()
]

# =============================================================================
# Chunking Configuration
# =============================================================================

CHUNK_SIZE = safe_int(os.getenv("CHUNK_SIZE"), 512)
CHUNK_OVERLAP = safe_int(os.getenv("CHUNK_OVERLAP"), 50)
MIN_CHUNK_SIZE = safe_int(os.getenv("MIN_CHUNK_SIZE"), 50)

# Code-specific chunking
CODE_CHUNK_STRATEGY_STR = os.getenv("CODE_CHUNK_STRATEGY", "code_blocks")
CODE_CHUNK_STRATEGY = ChunkingStrategy(CODE_CHUNK_STRATEGY_STR)
CODE_INCLUDE_CONTEXT = safe_bool(os.getenv("CODE_INCLUDE_CONTEXT"), True)
CODE_CONTEXT_LINES = safe_int(os.getenv("CODE_CONTEXT_LINES"), 5)

# PDF-specific chunking
PDF_CHUNK_STRATEGY_STR = os.getenv("PDF_CHUNK_STRATEGY", "pages")
PDF_CHUNK_STRATEGY = ChunkingStrategy(PDF_CHUNK_STRATEGY_STR)

# =============================================================================
# Search Configuration
# =============================================================================

SEARCH_LIMIT = safe_int(os.getenv("SEARCH_LIMIT"), 10)
MMR_LAMBDA = safe_float(os.getenv("MMR_LAMBDA"), 0.7)
HYBRID_ALPHA = safe_float(os.getenv("HYBRID_ALPHA"), 0.7)

# Cross-modal search
ENABLE_CROSS_MODAL_SEARCH = safe_bool(os.getenv("ENABLE_CROSS_MODAL_SEARCH"), True)
CROSS_MODAL_SIMILARITY_THRESHOLD = safe_float(os.getenv("CROSS_MODAL_SIMILARITY_THRESHOLD"), 0.7)

# Modality weights (equal by default for fair contribution)
MODALITY_WEIGHT_TEXT = safe_float(os.getenv("MODALITY_WEIGHT_TEXT"), 1.0)
MODALITY_WEIGHT_CODE = safe_float(os.getenv("MODALITY_WEIGHT_CODE"), 1.0)
MODALITY_WEIGHT_VISION = safe_float(os.getenv("MODALITY_WEIGHT_VISION"), 1.0)
MODALITY_WEIGHT_FTS = safe_float(os.getenv("MODALITY_WEIGHT_FTS"), 1.0)

MODALITY_WEIGHTS = {
    EmbeddingModality.TEXT: MODALITY_WEIGHT_TEXT,
    EmbeddingModality.CODE: MODALITY_WEIGHT_CODE,
    EmbeddingModality.VISION: MODALITY_WEIGHT_VISION,
}

# =============================================================================
# Codebase Management Configuration
# =============================================================================

CODEBASE_AUTO_DETECT = safe_bool(os.getenv("CODEBASE_AUTO_DETECT"), True)
CODEBASE_INDEX_TESTS = safe_bool(os.getenv("CODEBASE_INDEX_TESTS"), True)
CODEBASE_MAX_FILE_SIZE_KB = safe_int(os.getenv("CODEBASE_MAX_FILE_SIZE_KB"), 500)

# =============================================================================
# PDF Processing Configuration
# =============================================================================

ENABLE_PDF_PROCESSING = safe_bool(os.getenv("ENABLE_PDF_PROCESSING"), True)
PDF_OCR_ENABLED = safe_bool(os.getenv("PDF_OCR_ENABLED"), False)

# =============================================================================
# Image Processing Configuration
# =============================================================================

IMAGE_GENERATE_CAPTIONS = safe_bool(os.getenv("IMAGE_GENERATE_CAPTIONS"), False)
IMAGE_CAPTION_MODEL = os.getenv("IMAGE_CAPTION_MODEL", "blip-base")

# =============================================================================
# Tool Behavior Configuration
# =============================================================================

TOOL_SEARCH_DEFAULT_LIMIT = safe_int(os.getenv("TOOL_SEARCH_DEFAULT_LIMIT"), 10)
TOOL_MAX_CONTEXT_LINES = safe_int(os.getenv("TOOL_MAX_CONTEXT_LINES"), 10)
CODE_MAX_DEPENDENCY_DEPTH = safe_int(os.getenv("CODE_MAX_DEPENDENCY_DEPTH"), 3)
CODE_MAX_REFERENCES = safe_int(os.getenv("CODE_MAX_REFERENCES"), 50)

# =============================================================================
# Background Processing Configuration
# =============================================================================

INDEX_POLL_INTERVAL = safe_float(os.getenv("INDEX_POLL_INTERVAL"), 60.0)
INDEX_START_DELAY = safe_float(os.getenv("INDEX_START_DELAY"), 5.0)


# =============================================================================
# Server Configuration
# =============================================================================

SERVER_HOST = os.getenv("LIBRARIAN_HOST", "127.0.0.1")
SERVER_PORT = safe_int(os.getenv("LIBRARIAN_PORT"), 8000)

# =============================================================================
# Optional Tools Configuration
# =============================================================================

ENABLE_OPTIONAL_TOOLS = safe_bool(os.getenv("LIBRARIAN_ENABLE_OPTIONAL_TOOLS"), True)


def ensure_directories() -> None:
    """Ensure required directories exist."""
    Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(SOURCES_CONFIG_PATH).parent.mkdir(parents=True, exist_ok=True)
