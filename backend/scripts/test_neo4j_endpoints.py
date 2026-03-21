"""
Test Neo4j endpoints: neo4j-stats, graph build, verify new uploads increment counts.
Requires backend running: uvicorn main:app --port 8000
Run: python backend/scripts/test_neo4j_endpoints.py
"""
import sys
import uuid
from pathlib import Path

# Ensure backend root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

BASE = "http://localhost:8000/api"


def get_neo4j_stats():
    """GET /api/upload/neo4j-stats"""
    r = requests.get(f"{BASE}/upload/neo4j-stats", timeout=10)
    if r.status_code != 200:
        print(f"  ERROR: neo4j-stats returned {r.status_code}: {r.text[:200]}")
        return None
    return r.json()


def add_test_document_to_neo4j():
    """Add test chunks to Neo4j via GraphBuilderService (same as pipeline does)."""
    from services.graph_builder import GraphBuilderService

    job_id = "test_e2e_" + uuid.uuid4().hex[:8]
    chunks = [
        {
            "chunk_id": f"{job_id}_c1",
            "text": "Revenue increased from 80 in 2021 to 100 in 2022. Costs were 50.",
            "source": "page_1",
            "page": 1,
        },
        {
            "chunk_id": f"{job_id}_c2",
            "text": "Machine learning is used for NLP. Transformers are popular.",
            "source": "page_2",
            "page": 2,
        },
    ]
    gb = GraphBuilderService()
    if not gb._get_driver():
        print("  ERROR: Neo4j driver not available")
        return None, None
    result = gb.build_graph(
        chunks,
        document_id=job_id,
        on_progress=None,
        on_graph_snapshot=None,
        session_id=None,
    )
    return job_id, result


def cleanup(job_id: str):
    """Remove test document from Neo4j."""
    from neo4j import GraphDatabase
    from config import get_settings

    s = get_settings()
    if not s.neo4j_uri:
        return
    driver = GraphDatabase.driver(
        s.neo4j_uri,
        auth=(s.neo4j_user_resolved, s.neo4j_password),
        connection_timeout=5,
    )
    with driver.session() as session:
        session.run(
            "MATCH (c:Chunk) WHERE c.document_id = $d OR c.doc_id = $d DETACH DELETE c",
            d=job_id,
        )
        session.run(
            "MATCH (e:Entity) WHERE e.document_id = $d OR e.doc_id = $d DETACH DELETE e",
            d=job_id,
        )
    driver.close()


def test_push_neo4j_via_api():
    """Test push-neo4j endpoint using FastAPI TestClient (same process = shared JobStore)."""
    from fastapi.testclient import TestClient
    from main import app

    job_id = "test_push_" + uuid.uuid4().hex[:8]
    chunks = [
        {"chunk_id": f"{job_id}_c1", "text": "Sales grew 20% in Q3.", "source": "p1", "page": 1},
        {"chunk_id": f"{job_id}_c2", "text": "AI drives innovation.", "source": "p2", "page": 2},
    ]
    from services.job_store import JobStore
    JobStore().init_job(job_id)
    JobStore().update(job_id, "chunks", chunks)

    with TestClient(app) as client:
        r = client.post(f"/api/upload/push-neo4j/{job_id}")
        if r.status_code != 200:
            print(f"  push-neo4j returned {r.status_code}: {r.json()}")
            return None, None
        body = r.json()
        return job_id, body


def main():
    print("=" * 50)
    print("Neo4j endpoints end-to-end test")
    print("=" * 50)

    # 0. Test push-neo4j (in-process)
    print("\n0. POST /api/upload/push-neo4j (TestClient, in-process)")
    result = test_push_neo4j_via_api()
    if result[0] is None:
        print("  SKIP or FAIL: push-neo4j test")
    else:
        job_id, body = result
        print(f"  OK: chunks={body.get('chunks_processed')}, entities={body.get('entities_created')}, stored={body.get('stored_in_neo4j')}")
        cleanup(job_id)

    # 1. Baseline stats
    print("\n1. GET /api/upload/neo4j-stats (baseline)")
    stats_before = get_neo4j_stats()
    if not stats_before:
        print("  Aborting: could not fetch neo4j-stats (is backend running on :8000?)")
        sys.exit(1)
    if not stats_before.get("connected"):
        print("  Neo4j not connected:", stats_before.get("error", "unknown"))
        sys.exit(1)
    c_before = stats_before.get("chunks", 0)
    e_before = stats_before.get("entities", 0)
    r_before = stats_before.get("relationships", 0)
    print(f"  Chunks: {c_before}, Entities: {e_before}, Relationships: {r_before}")

    # 2. Add test document to Neo4j (via GraphBuilder - same as pipeline)
    print("\n2. Add test document to Neo4j (GraphBuilder.build_graph)")
    result = add_test_document_to_neo4j()
    if result[0] is None:
        print("  Aborting: could not add test data")
        sys.exit(1)
    job_id, build_result = result
    print(f"  Job ID: {job_id}")
    print(f"  build_graph: chunks={build_result['chunks_processed']}, entities={build_result['entities_created']}, rels={build_result['relations_created']}")

    # 3. Verify counts increased
    print("\n3. GET /api/upload/neo4j-stats (after upload)")
    stats_after = get_neo4j_stats()
    if not stats_after:
        print("  ERROR: could not fetch neo4j-stats after upload")
        cleanup(job_id)
        sys.exit(1)
    c_after = stats_after.get("chunks", 0)
    e_after = stats_after.get("entities", 0)
    r_after = stats_after.get("relationships", 0)
    print(f"  Chunks: {c_after}, Entities: {e_after}, Relationships: {r_after}")

    # 4. Assert increase
    c_diff = c_after - c_before
    e_diff = e_after - e_before
    r_diff = r_after - r_before
    ok = c_diff >= 2 and e_diff >= 1  # we added 2 chunks, at least a few entities
    print(f"\n4. Delta: +{c_diff} Chunks, +{e_diff} Entities, +{r_diff} Relationships")
    if not ok:
        print("  FAIL: Expected at least +2 chunks and +1 entity")
    else:
        print("  OK: New upload reflected in Neo4j counts")

    # 5. Cleanup (optional - comment out to keep test data)
    cleanup(job_id)
    print(f"\n5. Cleaned up test document {job_id}")

    print("\n" + "=" * 50)
    print("PASS" if ok else "FAIL")
    print("\nTo test with live server: ensure backend runs (uvicorn), then run this script.")
    print("neo4j-stats is fetched from http://localhost:8000")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
