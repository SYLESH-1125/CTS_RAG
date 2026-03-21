"""
Upload API - PDF ingestion with real-time extraction logs.
"""
import copy
import uuid
import asyncio
from pathlib import Path

import fitz


def _minimal_extract(file_path: str, filename: str) -> dict:
    """Fallback: text-only extraction when full extraction times out."""
    text_units = []
    with fitz.open(file_path) as doc:
        num_pages = len(doc)
        for i in range(num_pages):
            text = doc[i].get_text().strip()
            if text:
                text_units.append({"type": "text", "original": text, "translated": text, "page": i + 1, "source": f"page_{i + 1}"})
    return {
        "text_units": text_units,
        "table_units": [],
        "image_units": [],
        "logs": [{"step": "fallback", "message": "Minimal text extraction (timeout)"}],
        "metadata": {"filename": filename, "pages": num_pages},
    }
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import json

from config import get_settings
from services.extraction import ExtractionService
from services.chunking import ChunkingService
from services.graph_builder import GraphBuilderService
from services.vector_store import VectorStoreService

router = APIRouter()
settings = get_settings()


@router.post("/pdf")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload PDF and start extraction pipeline.
    Returns job_id for SSE status polling.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "PDF file required")

    job_id = str(uuid.uuid4())
    upload_path = Path(settings.upload_dir) / job_id
    upload_path.mkdir(parents=True, exist_ok=True)
    file_path = upload_path / file.filename

    # Save file
    content = await file.read()
    file_path.write_bytes(content)

    # Create job immediately so SSE stream finds it when frontend connects
    from services.job_store import JobStore
    from services.stream_hub import StreamHub

    JobStore().init_job(job_id)
    StreamHub.ensure_session(job_id)

    # Start processing in background
    background_tasks.add_task(
        process_pdf_pipeline,
        job_id=job_id,
        file_path=str(file_path),
    )

    return {"job_id": job_id, "status": "processing", "message": "Extraction started"}


async def process_pdf_pipeline(job_id: str, file_path: str):
    """
    Full pipeline: Extract → Chunk → Store (FAISS + Neo4j).
    Updates job status for SSE consumption.
    Job is pre-created in upload_pdf so SSE stream finds it immediately.
    """
    from services.job_store import JobStore
    from services.stream_hub import StreamHub

    store = JobStore()

    def stream_push(event_type: str, data: dict) -> None:
        StreamHub.push(job_id, event_type, data)

    def on_extraction_log(step: str, message: str):
        store.append_log(job_id, "extraction", f"[{step}] {message}")

    try:
        # Phase 1: Extraction (timeout 5 min - avoid hangs)
        store.update(job_id, "phase", "extraction")
        store.append_log(job_id, "extraction", "Extraction started")
        extractor = ExtractionService()

        def on_extraction_progress(step: str, payload: dict):
            store.patch_progress(job_id, "extraction", {"step": step, **payload})
            stream_push("progress_update", {"phase": "extraction", "step": step, **payload})

        try:
            extracted = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: extractor.extract_pdf(
                        file_path,
                        on_log=on_extraction_log,
                        on_progress=on_extraction_progress,
                        on_stream=stream_push,
                    )
                ),
                timeout=300.0,
            )
        except asyncio.TimeoutError:
            store.append_log(job_id, "extraction", "Extraction timed out (5 min) - fallback text only")
            path = Path(file_path)
            extracted = await asyncio.to_thread(_minimal_extract, file_path, path.name)
        store.update(job_id, "extraction", extracted)
        store.append_log(job_id, "extraction", "Extraction completed")

        # Phase 2: Chunking (live progress + growing chunk list in UI)
        store.update(job_id, "phase", "chunking")
        store.append_log(job_id, "chunking", "Chunking started")
        chunker = ChunkingService()

        def on_chunk_progress(cur: int, tot: int, stage: str, partial: list):
            store.patch_progress(job_id, "chunking", {"current": cur, "total": tot, "stage": stage})
            stream_push(
                "progress_update",
                {"phase": "chunking", "stage": stage, "current": cur, "total": tot},
            )
            if partial:
                store.update(job_id, "chunks", partial)

        chunks = await asyncio.to_thread(
            lambda: chunker.chunk(extracted, on_progress=on_chunk_progress, on_stream=stream_push)
        )
        store.update(job_id, "chunks", chunks)
        store.append_log(job_id, "chunking", f"Created {len(chunks)} chunks")

        # Phase 3: Graph build (required - each PDF = isolated graph, clears previous)
        store.update(job_id, "phase", "graph_build")
        store.append_log(job_id, "graph_build", "Building knowledge graph...")
        graph_result = {"chunks_processed": 0, "entities_created": 0, "relations_created": 0}
        try:
            graph_builder = GraphBuilderService()

            def on_graph_progress(cur: int, tot: int, ent: int, rel: int):
                pct = int(round(100 * cur / tot)) if tot else 0
                store.patch_progress(
                    job_id,
                    "graph",
                    {"current": cur, "total": tot, "entities": ent, "relations": rel, "percent": pct},
                )
                stream_push(
                    "progress_update",
                    {
                        "phase": "graph_build",
                        "current": cur,
                        "total": tot,
                        "entities": ent,
                        "relations": rel,
                        "percent": pct,
                    },
                )

            def on_live_graph(snap: dict):
                store.update(job_id, "live_graph", snap)

            graph_result = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: graph_builder.build_graph(
                        list(chunks),
                        job_id,
                        on_progress=on_graph_progress,
                        on_graph_snapshot=on_live_graph,
                        session_id=job_id,
                    )
                ),
                timeout=600.0,
            )
        except asyncio.TimeoutError:
            store.append_log(job_id, "graph_build", "Graph build timed out (10 min)")
        except Exception as e:
            store.append_log(job_id, "graph_build", f"Graph build error: {e}")
            import traceback
            traceback.print_exc()

        store.update(job_id, "graph", graph_result)
        store.append_log(job_id, "graph_build", f"Graph built: {graph_result['chunks_processed']} chunks, {graph_result['entities_created']} entities")

        # Phase 4: Vector store (cumulative - append to Neo4j + FAISS, queries search across all uploads)
        store.update(job_id, "phase", "vector_store")
        store.append_log(job_id, "vector_store", "Creating embeddings...")
        try:
            for c in chunks:
                c["document_id"] = job_id
            vector_store = VectorStoreService()
            await asyncio.to_thread(lambda: vector_store.add_chunks(chunks, document_id=job_id))
            store.append_log(job_id, "vector_store", f"FAISS index: added {len(chunks)} chunks (cumulative)")
        except Exception as e:
            store.append_log(job_id, "vector_store", f"Vector store failed: {e}")
            raise

        store.update(job_id, "status", "completed")
        store.update(job_id, "phase", "completed")
        store.append_log(job_id, "complete", "Pipeline completed successfully")
        StreamHub.push(job_id, "done", {"status": "completed"})
        from services.current_document import set_active_document
        set_active_document(job_id)
        
    except Exception as e:
        store.update(job_id, "status", "failed")
        store.append_log(job_id, "error", str(e))
        store.update(job_id, "error", str(e))
        try:
            from services.stream_hub import StreamHub as _SH

            _SH.push(job_id, "done", {"status": "failed", "error": str(e)})
        except Exception:
            pass


@router.get("/status/{job_id}/stream")
async def stream_job_status(job_id: str):
    """
    Server-Sent Events stream for real-time job status.
    """
    from services.job_store import JobStore
    
    async def event_generator():
        store = JobStore()
        last_data = None
        while True:
            data = store.get(job_id)
            if data is None:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                return
            
            # Send when data changes (deep copy to detect mutations)
            if data != last_data:
                last_data = copy.deepcopy(data)
                yield f"data: {json.dumps(data)}\n\n"
            
            if data.get("status") in ("completed", "failed"):
                return
            
            await asyncio.sleep(0.2)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/status/{job_id}")
def get_job_status(job_id: str):
    """Get current job status (polling fallback)."""
    from services.job_store import JobStore

    store = JobStore()
    data = store.get(job_id)
    if data is None:
        raise HTTPException(404, "Job not found")
    return data


@router.post("/push-neo4j/{job_id}")
def push_to_neo4j(job_id: str):
    """
    Explicitly push job's graph (chunks + entities + relations) to Neo4j.
    Used by classic upload after graph build. Fails if Neo4j is not configured.
    """
    from services.job_store import JobStore

    settings = get_settings()
    if not settings.neo4j_uri:
        raise HTTPException(503, "Neo4j not configured. Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in backend .env")

    store = JobStore()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    chunks = job.get("chunks")
    if not chunks:
        raise HTTPException(400, "No chunks to push (run extraction + chunking first)")

    graph_builder = GraphBuilderService()
    driver = graph_builder._get_driver()
    if not driver:
        raise HTTPException(503, "Neo4j connection failed. Check NEO4J_URI and credentials in backend .env")

    try:
        # skip_clear=True: pipeline already wrote; this re-push only upserts, avoids clearing-then-fail data loss
        result = graph_builder.build_graph(
            list(chunks),
            document_id=job_id,
            on_progress=None,
            on_graph_snapshot=None,
            session_id=None,
            skip_clear=True,
        )
        # Verify data was written to Neo4j
        with driver.session() as session:
            r = session.run(
                """
                MATCH (c:Chunk) WHERE c.document_id = $d OR c.doc_id = $d
                WITH count(c) AS nc
                MATCH (e:Entity) WHERE e.document_id = $d OR e.doc_id = $d
                WITH nc, count(e) AS ne
                OPTIONAL MATCH (a:Entity)-[r]->(b:Entity)
                WHERE (a.document_id = $d OR a.doc_id = $d) AND (b.document_id = $d OR b.doc_id = $d)
                RETURN nc, ne, count(r) AS nrel
                """,
                d=job_id,
            ).single()
        rec = r or {}
        stored = (rec.get("nc") or 0, rec.get("ne") or 0, rec.get("nrel") or 0)
        return {
            "status": "ok",
            "message": "Pushed to Neo4j",
            "chunks_processed": result["chunks_processed"],
            "entities_created": result["entities_created"],
            "relations_created": result["relations_created"],
            "stored_in_neo4j": {"chunks": stored[0], "entities": stored[1], "relations": stored[2]},
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/neo4j-stats")
def get_neo4j_stats():
    """Return total Chunks, Entities, and Relationships in Neo4j (for verification)."""
    settings = get_settings()
    if not settings.neo4j_uri:
        return {"connected": False, "chunks": 0, "entities": 0, "relationships": 0, "by_document": []}

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user_resolved, settings.neo4j_password),
            connection_timeout=5,
        )
        driver.verify_connectivity()
    except Exception as e:
        return {"connected": False, "error": str(e), "chunks": 0, "entities": 0, "relationships": 0}

    try:
        with driver.session() as session:
            r = session.run(
                """
                OPTIONAL MATCH (c:Chunk) WITH count(c) AS nc
                OPTIONAL MATCH (e:Entity) WITH nc, count(e) AS ne
                OPTIONAL MATCH (a:Entity)-[r]->(b:Entity) WITH nc, ne, count(r) AS nrel
                RETURN nc, ne, nrel
                """
            ).single()
            rec = r or {}
            nc, ne, nrel = rec.get("nc") or 0, rec.get("ne") or 0, rec.get("nrel") or 0

            # Per-document breakdown
            by_doc = []
            for row in session.run(
                """
                MATCH (c:Chunk)
                WHERE c.document_id IS NOT NULL AND c.document_id <> ''
                RETURN c.document_id AS doc, count(c) AS chunks
                """
            ):
                by_doc.append({"document_id": row["doc"], "chunks": row["chunks"]})

        driver.close()
        return {
            "connected": True,
            "chunks": nc,
            "entities": ne,
            "relationships": nrel,
            "by_document": by_doc[:20],
        }
    except Exception as e:
        return {"connected": True, "error": str(e), "chunks": 0, "entities": 0, "relationships": 0}
