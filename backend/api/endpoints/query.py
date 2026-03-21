"""
Query API - Graph-first RAG with optional FAISS fallback.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.query_engine import QueryEngine

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    document_id: str | None = None  # Scope to PDF (job_id); None + search_all=False → last upload
    search_all: bool = False  # If True, search across all documents in Neo4j/FAISS (cumulative)


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict] = Field(default_factory=list)  # [{chunk_id, page, source, text}]
    graph_trace: dict = Field(default_factory=dict)  # {nodes, edges, used_node_ids}
    graph_nodes: list[dict] = Field(default_factory=list)
    graph_edges: list[dict] = Field(default_factory=list)
    retrieval_graph: dict = Field(default_factory=dict)
    chunks_used: list[dict] = Field(default_factory=list)
    reasoning_steps: list[dict] = Field(default_factory=list)
    confidence: float = 0.0
    cypher_query: str | None = None
    processing_timeline: list[dict] = Field(default_factory=list)


@router.post("/", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    """
    Graph RAG: document_id scopes to one PDF; search_all=True queries across all cumulative uploads.
    """
    from services.current_document import get_active_document

    doc_id: str | None = None
    if req.search_all:
        doc_id = None  # Query across all documents in Neo4j + FAISS
    else:
        doc_id = req.document_id or get_active_document()
    try:
        from services.query_engine import QueryEngine
        result = QueryEngine().query(req.query, document_id=doc_id)
        return QueryResponse(**result)
    except Exception as err:
        raise HTTPException(500, str(err))
