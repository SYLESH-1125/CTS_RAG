"""
API routes for Graph RAG.
"""
from fastapi import APIRouter

from api.endpoints import upload, query, status, graph, stream

router = APIRouter()

router.include_router(upload.router, prefix="/upload", tags=["upload"])
router.include_router(query.router, prefix="/query", tags=["query"])
router.include_router(status.router, prefix="/status", tags=["status"])
router.include_router(graph.router, prefix="/graph", tags=["graph"])
router.include_router(stream.router, prefix="/stream", tags=["stream"])
