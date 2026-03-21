"""Export knowledge graph for a document (force-graph UI)."""
from fastapi import APIRouter, HTTPException

from config import get_settings

router = APIRouter()


@router.get("/document/{document_id}")
def get_document_graph(document_id: str):
    """Return nodes + links for one PDF (document_id = job_id from upload)."""
    s = get_settings()
    if not s.neo4j_uri:
        return {"nodes": [], "links": []}
    try:
        from neo4j import GraphDatabase
        from services.graph_builder import GraphBuilderService

        driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user_resolved, s.neo4j_password))
        with driver.session() as session:
            snap = GraphBuilderService()._snapshot_document_graph(session, document_id)
        return snap
    except Exception as e:
        raise HTTPException(500, str(e))
