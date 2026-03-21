"""
Per-session event buffer for real-time SSE/WebSocket-style streaming.
session_id == upload job_id.
Thread-safe; safe to call from background threads (sync pipeline).
"""
from __future__ import annotations

import time
from threading import Lock
from typing import Any

_MAX_EVENTS = 8000
_TRIM_TO = 5000


class StreamHub:
    _buffers: dict[str, list[dict[str, Any]]] = {}
    _lock = Lock()

    @classmethod
    def ensure_session(cls, session_id: str) -> None:
        with cls._lock:
            if session_id not in cls._buffers:
                cls._buffers[session_id] = []

    @classmethod
    def push(cls, session_id: str, event_type: str, data: dict[str, Any]) -> None:
        ev = {"type": event_type, "data": dict(data), "ts": time.time()}
        with cls._lock:
            buf = cls._buffers.setdefault(session_id, [])
            buf.append(ev)
            if len(buf) > _MAX_EVENTS:
                del buf[: len(buf) - _TRIM_TO]

    @classmethod
    def count(cls, session_id: str) -> int:
        with cls._lock:
            return len(cls._buffers.get(session_id, []))

    @classmethod
    def get_from(cls, session_id: str, start_index: int) -> tuple[list[dict[str, Any]], int]:
        """Return events[start_index:] and total length."""
        with cls._lock:
            buf = cls._buffers.get(session_id, [])
            total = len(buf)
            if start_index >= total:
                return [], total
            return buf[start_index:], total

    @classmethod
    def clear_session(cls, session_id: str) -> None:
        with cls._lock:
            cls._buffers.pop(session_id, None)
