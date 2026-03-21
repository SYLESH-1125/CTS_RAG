"""
Server-Sent Events: granular pipeline events for session_id (same as job_id).
"""
import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter()


@router.get("/{session_id}")
async def stream_pipeline_events(session_id: str):
    """
    SSE stream with named events:
      extraction_text, extraction_table, extraction_image, chunk_created,
      entity_created, relationship_created, graph_update, progress_update, done, error
    """
    from services.job_store import JobStore
    from services.stream_hub import StreamHub

    store = JobStore()
    if store.get(session_id) is None:
        raise HTTPException(404, "Unknown session_id / job not started")

    StreamHub.ensure_session(session_id)

    async def event_generator():
        index = 0
        idle_rounds = 0
        try:
            while True:
                events, _total = StreamHub.get_from(session_id, index)
                if events:
                    idle_rounds = 0
                    for ev in events:
                        typ = ev.get("type", "message")
                        data = ev.get("data", {})
                        yield f"event: {typ}\ndata: {json.dumps(data, default=str)}\n\n"
                        index += 1
                else:
                    idle_rounds += 1

                job = store.get(session_id)
                status = (job or {}).get("status")
                if status in ("completed", "failed") and idle_rounds >= 8:
                    yield f"event: done\ndata: {json.dumps({'status': status})}\n\n"
                    return

                await asyncio.sleep(0.08)
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
