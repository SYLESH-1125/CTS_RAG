"""
Shared embedder - single model instance for chunking and vector store.
Avoids loading SentenceTransformer twice (~2x faster pipeline).
"""
import logging
from threading import Lock

from config import get_settings

logger = logging.getLogger("graph_rag.services.embedder")

_lock = Lock()
_instance = None


def get_embedder():
    """Singleton SentenceTransformer - shared across services."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                from sentence_transformers import SentenceTransformer
                settings = get_settings()
                _instance = SentenceTransformer(settings.embedding_model)
                logger.info("Embedder loaded (shared)")
    return _instance
