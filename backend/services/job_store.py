"""
In-memory job store for pipeline status.
Used for SSE streaming and UI updates.
"""
import logging
from typing import Any
from threading import Lock

logger = logging.getLogger("graph_rag.services.job_store")

# Single source of truth for pipeline phases - frontend consumes dynamically
PIPELINE_PHASES = ["extraction", "chunking", "graph_build", "vector_store", "completed"]


class JobStore:
    """Thread-safe in-memory job status store."""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, "_jobs"):
            self._jobs: dict[str, dict] = {}
            self._job_lock = Lock()
    
    def init_job(self, job_id: str):
        with self._job_lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "processing",
                "phase": "init",
                "phases": PIPELINE_PHASES,
                "extraction": None,
                "chunks": None,
                "graph": None,
                "logs": [],
                "error": None,
                "live_graph": None,
                "progress": {
                    "extraction": {"step": "", "current": 0, "total": 0, "detail": ""},
                    "chunking": {"current": 0, "total": 0, "stage": ""},
                    "graph": {"current": 0, "total": 0, "entities": 0, "relations": 0},
                },
            }
    
    def update(self, job_id: str, key: str, value: Any):
        with self._job_lock:
            if job_id in self._jobs:
                self._jobs[job_id][key] = value
    
    def append_log(self, job_id: str, phase: str, message: str):
        with self._job_lock:
            if job_id in self._jobs:
                self._jobs[job_id]["logs"].append({"phase": phase, "message": message})

    def patch_progress(self, job_id: str, section: str, data: dict):
        """Merge into progress[section] for live UI (extraction | chunking | graph)."""
        with self._job_lock:
            if job_id in self._jobs and "progress" in self._jobs[job_id]:
                base = self._jobs[job_id]["progress"].get(section) or {}
                base = {**base, **data}
                self._jobs[job_id]["progress"][section] = base
    
    def get(self, job_id: str) -> dict | None:
        with self._job_lock:
            return self._jobs.get(job_id)
