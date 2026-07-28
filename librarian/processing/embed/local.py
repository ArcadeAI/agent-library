"""
Local embedding provider using sentence-transformers.

Provides embedding generation using locally-run sentence transformer models.
Supports lazy loading to avoid startup overhead when embeddings aren't needed.

Supports both text and image inputs for CLIP-based models.
"""

import importlib.util
import logging
import signal
import threading
import warnings
from typing import TYPE_CHECKING, Any, Union

import numpy as np

from librarian.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL
from librarian.processing.embed.base import EmbeddingProvider

# Timeout for model loading (seconds). Prevents indefinite hangs if model
# download stalls or disk I/O is slow. Set to 0 to disable.
MODEL_LOAD_TIMEOUT = 120

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ModelLoadTimeoutError(TimeoutError):
    """Raised when model loading exceeds the configured timeout."""

    pass


class _TimeoutHandler:
    """Context manager for setting a timeout on model loading (Unix only).

    Becomes a no-op when the POSIX alarm is unusable: on platforms without
    ``SIGALRM``, and on any non-main thread, where ``signal.signal`` raises
    ``ValueError``. Callers that load the model from a thread pool would
    otherwise fail outright rather than simply loading without a timeout.
    """

    def __init__(self, seconds: int, model_name: str):
        self.seconds = seconds
        self.model_name = model_name
        self._old_handler: Any = None
        self._armed = False

    def _handler(self, signum: int, frame: Any) -> None:
        raise ModelLoadTimeoutError(
            f"Loading model '{self.model_name}' timed out after {self.seconds}s. "
            f"This may indicate a network issue downloading the model or disk I/O problems. "
            f"Try running again or pre-download the model."
        )

    def _can_arm(self) -> bool:
        return (
            self.seconds > 0
            and hasattr(signal, "SIGALRM")
            and threading.current_thread() is threading.main_thread()
        )

    def __enter__(self) -> "_TimeoutHandler":
        if self._can_arm():
            self._old_handler = signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
            self._armed = True
        return self

    def __exit__(self, *args: Any) -> None:
        if self._armed:
            signal.alarm(0)
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)
            self._armed = False


# Suppress sentence-transformers and transformers info/warning messages globally
# (e.g., "No sentence-transformers model found...", "Using a slow image processor...")
for _st_logger_name in [
    "sentence_transformers",
    "sentence_transformers.SentenceTransformer",
    "transformers",
    "transformers.image_processing_utils",
]:
    logging.getLogger(_st_logger_name).setLevel(logging.ERROR)

# Suppress transformers FutureWarning about slow image processor
warnings.filterwarnings("ignore", message=".*slow image processor.*", category=FutureWarning)

# =============================================================================
# Module-level dependency availability flags
# (following the pattern used in parsers/image.py and parsers/pdf.py)
# =============================================================================


def _is_module_available(module_name: str) -> bool:
    """Return True when a module can be imported without importing it."""
    return importlib.util.find_spec(module_name) is not None


PILLOW_AVAILABLE = _is_module_available("PIL")
TRANSFORMERS_AVAILABLE = _is_module_available("torch") and _is_module_available("transformers")

# Type alias for content that can be embedded (text or image)
EmbeddableContent = Union[str, "PILImage"]  # type: ignore[no-any-unimported]


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using local sentence-transformers models.

    Lazily loads the model on first use to avoid startup costs.
    Thread-safe for concurrent embedding requests.

    Models are automatically downloaded from HuggingFace Hub if not cached.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        """
        Initialize the local embedding provider.

        Args:
            model_name: Sentence transformer model name (default: from config).
            device: Device to run on ('cpu', 'cuda', 'mps', etc.).
        """
        self._model_name = model_name or EMBEDDING_MODEL
        self._device = device
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def model(self) -> "SentenceTransformer":
        """
        Lazily load and return the sentence transformer model.

        Thread-safe double-checked locking pattern.
        Models are downloaded from HuggingFace Hub on first access if not cached.
        """
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._load_model()
        return self._model  # type: ignore[return-value]

    def _is_clip_model(self) -> bool:
        """Check if this is a CLIP-based model that supports images."""
        model_lower = self._model_name.lower()
        return "clip" in model_lower or "siglip" in model_lower

    def _is_codebert_model(self) -> bool:
        """Check if this is a CodeBERT-based model for code embeddings."""
        model_lower = self._model_name.lower()
        return "codebert" in model_lower or "codellama" in model_lower

    def _check_dependencies(self) -> None:
        """
        Check that required dependencies are installed for this model type.

        Raises:
            ImportError: With a specific, actionable error message if deps are missing.
        """
        if self._is_clip_model() and not PILLOW_AVAILABLE:
            msg = (
                f"CLIP model '{self._model_name}' requires Pillow. "
                f"Install with: pip install Pillow>=10.0.0\n"
                f"Or install all vision dependencies: pip install -e '.[vision]'"
            )
            raise ImportError(msg)
        elif self._is_codebert_model() and not TRANSFORMERS_AVAILABLE:
            msg = (
                f"CodeBERT model '{self._model_name}' requires transformers and torch. "
                f"Install with: pip install transformers>=4.30.0 torch>=2.0.0\n"
                f"Or install all code dependencies: pip install -e '.[code]'"
            )
            raise ImportError(msg)

    def validate(self) -> bool:
        """
        Validate that this provider's dependencies are installed.

        Does NOT load the model -- only checks that required imports will succeed.
        Fast enough to call eagerly on embedder creation.

        Returns:
            True if all dependencies are available.

        Raises:
            ImportError: If required dependencies are missing.
        """
        self._check_dependencies()
        return True

    def _load_model(self) -> None:
        """
        Load the sentence transformer model.

        Downloads the model from HuggingFace Hub if not cached locally.
        Suppresses noisy progress bars from weight materialization.

        Raises:
            ImportError: If sentence-transformers or model dependencies are missing.
            ModelLoadTimeoutError: If model loading exceeds MODEL_LOAD_TIMEOUT.
            RuntimeError: If model loading fails for other reasons.
        """
        import os

        # Check model-specific dependencies FIRST with clear error messages
        self._check_dependencies()

        # Suppress verbose logging and progress bars
        for logger_name in [
            "sentence_transformers",
            "sentence_transformers.SentenceTransformer",
            "transformers",
        ]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*use_fast.*")
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        # Suppress tqdm progress bars (weight materialization, tokenizer loading, etc.)
        old_tqdm_disable = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            msg = (
                "sentence-transformers is required for local embedding. "
                "Install with: pip install sentence-transformers"
            )
            raise ImportError(msg) from e

        logger.info(
            "Loading local embedding model: %s (will download if not cached)",
            self._model_name,
        )

        try:
            # Use timeout to prevent indefinite hangs during model loading
            with _TimeoutHandler(MODEL_LOAD_TIMEOUT, self._model_name):
                if self._device:
                    self._model = SentenceTransformer(self._model_name, device=self._device)
                else:
                    self._model = SentenceTransformer(self._model_name)
        except ModelLoadTimeoutError:
            # Re-raise timeout errors with clear message
            raise
        except ImportError as e:
            # Model loaded but a sub-dependency is missing (e.g., Pillow for CLIP)
            error_str = str(e).lower()
            if "pillow" in error_str or "pil" in error_str or "vision" in error_str:
                msg = (
                    f"Model '{self._model_name}' requires Pillow for image processing. "
                    f"Install with: pip install Pillow>=10.0.0\n"
                    f"Or install all vision dependencies: pip install -e '.[vision]'"
                )
                raise ImportError(msg) from e
            raise
        except Exception as e:
            # Wrap other errors with context about what we were trying to do
            error_type = type(e).__name__
            raise RuntimeError(
                f"Failed to load embedding model '{self._model_name}': {error_type}: {e}"
            ) from e
        finally:
            # Restore TQDM_DISABLE
            if old_tqdm_disable is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = old_tqdm_disable

        # Some models (e.g., CLIP) return None for get_sentence_embedding_dimension()
        # or may throw an exception, so wrap in try/except
        try:
            raw_dim = self._model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.warning(
                "Failed to get embedding dimension for %s: %s. Using default.",
                self._model_name,
                e,
            )
            raw_dim = None

        if raw_dim is not None:
            self._dimension = raw_dim
            logger.info("Loaded model %s with dimension %d", self._model_name, self._dimension)
        else:
            # Fall back to config default
            self._dimension = EMBEDDING_DIMENSION
            logger.info(
                "Loaded model %s (dimension not reported, using default %d)",
                self._model_name,
                self._dimension,
            )

    @property
    def dimension(self) -> int:
        """Return the embedding dimension, loading model if needed."""
        if self._dimension is None:
            _ = self.model  # Trigger load
        return self._dimension or EMBEDDING_DIMENSION

    def _is_image(self, content: Any) -> bool:
        """Check if content is a PIL Image."""
        try:
            from PIL.Image import Image as PILImage

            return isinstance(content, PILImage)
        except ImportError:
            return False

    def embed(self, content: EmbeddableContent) -> list[float]:
        """
        Generate embedding for a single piece of content.

        Args:
            content: Text string or PIL Image to embed.

        Returns:
            Embedding vector as list of floats.

        Raises:
            ValueError: If image passed to non-CLIP model.
        """
        if self._is_image(content):
            if not self._is_clip_model():
                msg = f"Model {self._model_name} does not support image embeddings"
                raise ValueError(msg)
            logger.debug("Generating image embedding with CLIP model")

        # sentence-transformers CLIP models accept PIL Images via encode()
        embedding = self.model.encode(content, convert_to_numpy=True)  # type: ignore[arg-type]
        result: list[float] = embedding.tolist()
        return result

    def embed_batch(  # type: ignore[override]
        self, contents: list[EmbeddableContent], batch_size: int = 32
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple content items.

        Supports mixed batches of text and images for CLIP models.

        Args:
            contents: List of texts or PIL Images to embed.
            batch_size: Batch size for processing.

        Returns:
            List of embedding vectors.

        Raises:
            ValueError: If images passed to non-CLIP model.
        """
        if not contents:
            return []

        # Check if any images in the batch
        has_images = any(self._is_image(c) for c in contents)
        if has_images and not self._is_clip_model():
            msg = f"Model {self._model_name} does not support image embeddings"
            raise ValueError(msg)

        if has_images:
            logger.debug(
                "Generating batch embeddings with %d items (includes images)", len(contents)
            )

        # sentence-transformers CLIP models accept PIL Images via encode()
        embeddings = self.model.encode(
            contents,  # type: ignore[arg-type]
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [emb.tolist() for emb in embeddings]

    def embed_image(self, image: "PILImage") -> list[float]:  # type: ignore[no-any-unimported]
        """
        Generate embedding for a PIL Image.

        Args:
            image: PIL Image object to embed.

        Returns:
            Embedding vector as list of floats.

        Raises:
            ValueError: If model doesn't support images.
        """
        if not self._is_clip_model():
            msg = f"Model {self._model_name} does not support image embeddings"
            raise ValueError(msg)

        logger.debug("Generating embedding for PIL Image")
        return self.embed(image)

    def embed_query(self, query: str) -> list[float]:
        """
        Generate query-optimized embedding.

        Handles model-specific query prefixes (E5, Instructor models).
        """
        # E5 models expect "query:" prefix
        if "e5" in self._model_name.lower():
            query = f"query: {query}"
        # Instructor models use instruction prefix
        elif "instructor" in self._model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"

        return self.embed(query)

    def embed_document(self, document: str) -> list[float]:
        """
        Generate document-optimized embedding.

        Handles model-specific document prefixes.
        """
        if "e5" in self._model_name.lower():
            document = f"passage: {document}"

        return self.embed(document)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Generate document-optimized embeddings for multiple documents.

        Handles model-specific document prefixes.
        """
        # Apply prefixes for models that need them
        if "e5" in self._model_name.lower():
            documents = [f"passage: {doc}" for doc in documents]

        # Cast to EmbeddableContent list for type compatibility
        contents: list[EmbeddableContent] = list(documents)
        return self.embed_batch(contents)

    def similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score (0-1).
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
