"""Active PDF session: last completed upload scopes query + FAISS to one document."""
from threading import Lock

_lock = Lock()
_active_document_id: str | None = None


def set_active_document(document_id: str) -> None:
    with _lock:
        global _active_document_id
        _active_document_id = document_id or None


def get_active_document() -> str | None:
    with _lock:
        return _active_document_id
