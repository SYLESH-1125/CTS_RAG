"""
Test Neo4j upload: verify connection, graph build, and relationship persistence.
Run from backend: python scripts/test_neo4j_upload.py
"""
import sys
from pathlib import Path

# Ensure backend root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def test_neo4j_config():
    """Test 1: Neo4j configuration and connectivity."""
    from config import get_settings
    s = get_settings()
    print("\n=== Test 1: Neo4j config ===")
    print(f"  NEO4J_URI: {'(set)' if s.neo4j_uri else '(empty)'}")
    print(f"  NEO4J_USERNAME: {s.neo4j_user_resolved or '(default)'}")
    if not s.neo4j_uri:
        print("  FAIL: NEO4J_URI not set in .env")
        return False
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            s.neo4j_uri,
            auth=(s.neo4j_user_resolved, s.neo4j_password),
            connection_timeout=5,
        )
        driver.verify_connectivity()
        driver.close()
        print("  OK: Connected to Neo4j")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_graph_build_and_persist():
    """Test 2: Build graph with sample chunks and verify in Neo4j."""
    from services.graph_builder import GraphBuilderService
    from services.job_store import JobStore

    print("\n=== Test 2: Graph build + persist ===")

    # Create fake job with sample chunks
    job_id = "test_neo4j_" + __import__("uuid").uuid4().hex[:8]
    chunks = [
        {
            "chunk_id": f"{job_id}_c1",
            "text": "Revenue was 80 in 2021 and increased to 100 in 2022. Costs were 50 in 2021.",
            "source": "page_1",
            "page": 1,
        },
        {
            "chunk_id": f"{job_id}_c2",
            "text": "CNN is used for image recognition. Transformers are used in NLP.",
            "source": "page_2",
            "page": 2,
        },
    ]

    JobStore().init_job(job_id)
    JobStore().update(job_id, "chunks", chunks)

    gb = GraphBuilderService()
    driver = gb._get_driver()
    if not driver:
        print("  FAIL: Cannot get Neo4j driver")
        return False

    try:
        result = gb.build_graph(
            chunks,
            document_id=job_id,
            on_progress=None,
            on_graph_snapshot=None,
            session_id=None,
        )
        print(f"  build_graph returned: chunks={result['chunks_processed']}, "
              f"entities={result['entities_created']}, rels={result['relations_created']}")
    except Exception as e:
        print(f"  FAIL build_graph: {e}")
        return False

    return job_id, result


def test_verify_in_neo4j(job_id: str):
    """Test 3: Query Neo4j to verify chunks, entities, and relationships."""
    from neo4j import GraphDatabase
    from config import get_settings

    print("\n=== Test 3: Verify data in Neo4j ===")
    s = get_settings()
    driver = GraphDatabase.driver(
        s.neo4j_uri,
        auth=(s.neo4j_user_resolved, s.neo4j_password),
    )

    with driver.session() as session:
        # Chunks
        r = session.run(
            "MATCH (c:Chunk) WHERE c.document_id = $d OR c.doc_id = $d RETURN count(c) AS n",
            d=job_id,
        ).single()
        nc = (r or {}).get("n", 0)
        print(f"  Chunks in Neo4j: {nc}")

        # Entities
        r = session.run(
            "MATCH (e:Entity) WHERE e.document_id = $d OR e.doc_id = $d RETURN count(e) AS n",
            d=job_id,
        ).single()
        ne = (r or {}).get("n", 0)
        print(f"  Entities in Neo4j: {ne}")

        # MENTIONS (Chunk->Entity)
        r = session.run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE (c.document_id = $d OR c.doc_id = $d)
              AND (e.document_id = $d OR e.doc_id = $d)
            RETURN count(*) AS n
            """,
            d=job_id,
        ).single()
        nment = (r or {}).get("n", 0)
        print(f"  MENTIONS edges: {nment}")

        # Entity-Entity relationships
        r = session.run(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE (a.document_id = $d OR a.doc_id = $d)
              AND (b.document_id = $d OR b.doc_id = $d)
            RETURN count(r) AS n, collect(type(r)) AS types
            """,
            d=job_id,
        ).single()
        nrel = (r or {}).get("n", 0)
        types = (r or {}).get("types", [])
        print(f"  Entity-Entity relationships: {nrel}")
        if types:
            from collections import Counter
            print(f"  Relationship types: {dict(Counter(types))}")

    driver.close()

    ok = nc > 0 and ne > 0
    if not ok:
        print("  FAIL: No chunks or entities found")
    elif nrel == 0 and ne > 1:
        print("  WARN: Entities exist but no Entity-Entity relationships (may be expected for some docs)")
    else:
        print("  OK: Data verified in Neo4j")

    return ok


def cleanup_test(job_id: str):
    """Remove test data from Neo4j."""
    from neo4j import GraphDatabase
    from config import get_settings

    s = get_settings()
    driver = GraphDatabase.driver(
        s.neo4j_uri,
        auth=(s.neo4j_user_resolved, s.neo4j_password),
    )
    with driver.session() as session:
        session.run("MATCH (c:Chunk) WHERE c.document_id = $d OR c.doc_id = $d DETACH DELETE c", d=job_id)
        session.run("MATCH (e:Entity) WHERE e.document_id = $d OR e.doc_id = $d DETACH DELETE e", d=job_id)
    driver.close()
    print(f"\n  Cleaned up test document {job_id}")


def main():
    print("Graph RAG - Neo4j upload test")
    if not test_neo4j_config():
        sys.exit(1)

    result = test_graph_build_and_persist()
    if result is False:
        sys.exit(1)
    if isinstance(result, tuple):
        job_id, _ = result
    else:
        sys.exit(1)

    ok = test_verify_in_neo4j(job_id)
    cleanup_test(job_id)

    print("\n=== Result ===")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
