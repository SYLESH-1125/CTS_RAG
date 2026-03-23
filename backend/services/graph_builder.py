"""
Graph builder: hybrid extraction per chunk, two-stage Neo4j writes (entities first, rels batched),
document isolation, optional communities, live streaming snapshots.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Callable

from config import get_settings

from services.chunk_processor import (
    ALLOWED_RELATIONSHIP_TYPES,
    ProcessedChunk,
    RelRow,
    extract_chunk_sync,
)

logger = logging.getLogger("graph_rag.services.graph_builder")


def _union_find_communities(keys: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    parent = {k: k for k in keys}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        if a in parent and b in parent:
            union(a, b)

    root_to_cid: dict[str, int] = {}
    out: dict[str, int] = {}
    next_id = 0
    for k in keys:
        r = find(k)
        if r not in root_to_cid:
            root_to_cid[r] = next_id
            next_id += 1
        out[k] = root_to_cid[r]
    return out


class GraphBuilderService:
    """Isolated graph per document: async parallel hybrid extract + staged Neo4j."""

    def __init__(self):
        self.settings = get_settings()
        self._driver = None

    def _get_driver(self):
        if self._driver is None:
            if not self.settings.neo4j_uri:
                return None
            try:
                from neo4j import GraphDatabase

                self._driver = GraphDatabase.driver(
                    self.settings.neo4j_uri,
                    auth=(self.settings.neo4j_user_resolved, self.settings.neo4j_password),
                    connection_timeout=10,
                )
                self._driver.verify_connectivity()
            except Exception as e:
                logger.warning("Neo4j connection failed: %s", e)
                return None
        return self._driver

    def clear_graph_for_document(self, document_id: str) -> bool:
        """Remove Chunk + Entity subgraph for this document only (multi-PDF safe)."""
        driver = self._get_driver()
        if not driver or not document_id:
            return False
        try:
            with driver.session() as session:
                session.run(
                    """
                    MATCH (c:Chunk)
                    WHERE c.document_id = $d OR c.doc_id = $d
                    DETACH DELETE c
                    """,
                    d=document_id,
                )
                session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.document_id = $d OR e.doc_id = $d
                    DETACH DELETE e
                    """,
                    d=document_id,
                )
            logger.info("Cleared graph for document %s…", document_id[:8])
            return True
        except Exception as e:
            logger.warning("Graph clear failed: %s", e)
            return False

    def clear_graph(self) -> bool:
        """Legacy: clear entire DB (single-tenant). Prefer clear_graph_for_document."""
        driver = self._get_driver()
        if not driver:
            return False
        try:
            with driver.session() as session:
                session.run("MATCH (c:Chunk) DETACH DELETE c")
                session.run("MATCH (e:Entity) DETACH DELETE e")
            return True
        except Exception as e:
            logger.warning("Graph clear failed: %s", e)
            return False

    def _snapshot_document_graph(self, session, document_id: str) -> dict[str, Any]:
        nodes: list[dict] = []
        links: list[dict] = []
        seen: set[str] = set()

        for rec in session.run(
            """
            MATCH (c:Chunk)
            WHERE c.document_id = $d OR c.doc_id = $d
            RETURN c.id AS cid, c.source AS src
            """,
            d=document_id,
        ):
            cid = f"chunk:{rec['cid']}"
            if cid not in seen:
                seen.add(cid)
                nodes.append({
                    "id": cid,
                    "name": (rec["src"] or "chunk")[:40],
                    "group": 1,
                    "entity_type": "chunk",
                    "community_id": -1,
                    "val": 1,
                })

        for rec in session.run(
            """
            MATCH (e:Entity)
            WHERE e.document_id = $d OR e.doc_id = $d
            OPTIONAL MATCH (e)-[r]-()
            WITH e, count(r) AS deg
            RETURN e.graph_key AS gid, e.name AS label, e.entity_type AS et,
                   coalesce(e.community_id, -1) AS comm, deg
            """,
            d=document_id,
        ):
            gid = rec["gid"]
            if gid and gid not in seen:
                seen.add(gid)
                label = (rec["label"] or "")[:60] or (
                    str(gid).split("::")[-1] if "::" in str(gid) else str(gid)
                )[:60]
                nodes.append({
                    "id": gid,
                    "name": label,
                    "group": 2,
                    "entity_type": rec["et"] or "concept",
                    "community_id": int(rec["comm"]) if rec["comm"] is not None else -1,
                    "val": max(1, int(rec["deg"] or 0)),
                })

        for rec in session.run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE (c.document_id = $d OR c.doc_id = $d)
              AND (e.document_id = $d OR e.doc_id = $d)
            RETURN c.id AS cid, e.graph_key AS gid
            """,
            d=document_id,
        ):
            cid, gid = rec["cid"], rec["gid"]
            if cid and gid:
                links.append({"source": f"chunk:{cid}", "target": gid, "type": "MENTIONS"})

        for rec in session.run(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE (a.document_id = $d OR a.doc_id = $d)
              AND (b.document_id = $d OR b.doc_id = $d)
            RETURN a.graph_key AS s, b.graph_key AS t, type(r) AS rel
            """,
            d=document_id,
        ):
            s, t = rec["s"], rec["t"]
            if s and t:
                links.append({"source": s, "target": t, "type": rec["rel"]})

        return {"nodes": nodes, "links": links}

    def _snapshot_with_driver(self, driver, document_id: str) -> dict[str, Any]:
        with driver.session() as session:
            return self._snapshot_document_graph(session, document_id)

    def _persist_stage1_entities(
        self,
        driver,
        document_id: str,
        pc: ProcessedChunk,
        push_stream: Callable[[str, dict[str, Any]], None],
    ) -> int:
        """Stage 1: Chunk + Entity nodes + MENTIONS only."""
        n_ent = 0
        with driver.session() as session:
            if pc.mentions:
                mentions = [asdict(m) for m in pc.mentions]
                session.run(
                    """
                    UNWIND $mentions AS m
                    MERGE (c:Chunk {id: m.chunk_id})
                    SET c.document_id = $document_id,
                        c.doc_id = $document_id,
                        c.text = m.text,
                        c.source = m.source,
                        c.page = m.page
                    MERGE (e:Entity {graph_key: m.graph_key})
                    SET e.document_id = $document_id,
                        e.doc_id = $document_id,
                        e.name = m.display_name,
                        e.entity_type = coalesce(m.entity_type, 'concept')
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    mentions=mentions,
                    document_id=document_id,
                )
                n_ent = len(pc.mentions)
            elif pc.chunk_id:
                session.run(
                    """
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.document_id = $document_id,
                        c.doc_id = $document_id,
                        c.text = $text,
                        c.source = $source,
                        c.page = $page
                    """,
                    chunk_id=pc.chunk_id,
                    document_id=document_id,
                    text=pc.text[:10000],
                    source=pc.source,
                    page=pc.page,
                )

        push_stream(
            "progress_update",
            {
                "phase": "graph_build",
                "step": "stage1_entities",
                "chunk_id": pc.chunk_id,
                "chunk_index": pc.chunk_index,
            },
        )

        for m in pc.mentions:
            push_stream(
                "entity_created",
                {
                    "name": m.display_name,
                    "graph_key": m.graph_key,
                    "chunk_id": m.chunk_id,
                    "entity_type": m.entity_type,
                    "group": 2,
                },
            )

        return n_ent

    def _persist_stage2_relationships(
        self,
        driver,
        document_id: str,
        relationships: list[RelRow],
        push_stream: Callable[[str, dict[str, Any]], None],
    ) -> int:
        """Stage 2: batched entity–entity relationships (valid types only)."""
        if not relationships:
            return 0
        n_rel = 0
        by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
        for r in relationships:
            if r.rel_type not in ALLOWED_RELATIONSHIP_TYPES:
                continue
            by_type[r.rel_type].append({
                "from_key": r.from_key,
                "to_key": r.to_key,
                "from_name": r.from_name,
                "to_name": r.to_name,
            })

        with driver.session() as session:
            for rel_type, rows in by_type.items():
                if not rows:
                    continue
                cypher = f"""
                UNWIND $rows AS r
                MERGE (a:Entity {{graph_key: r.from_key}})
                SET a.document_id = $document_id, a.doc_id = $document_id, a.name = r.from_name
                MERGE (b:Entity {{graph_key: r.to_key}})
                SET b.document_id = $document_id, b.doc_id = $document_id, b.name = r.to_name
                MERGE (a)-[:`{rel_type}`]->(b)
                """
                session.run(cypher, rows=rows, document_id=document_id)
                n_rel += len(rows)
                for row in rows:
                    push_stream(
                        "relationship_created",
                        {
                            "subject": row["from_name"],
                            "object": row["to_name"],
                            "predicate": rel_type,
                            "source_key": row["from_key"],
                            "target_key": row["to_key"],
                        },
                    )

        push_stream(
            "progress_update",
            {
                "phase": "graph_build",
                "step": "stage2_relationships",
                "relations_written": n_rel,
            },
        )
        return n_rel

    def _apply_communities(
        self,
        driver,
        document_id: str,
        relationships: list[RelRow],
        all_graph_keys: set[str],
    ) -> None:
        if not self.settings.graph_enable_communities:
            return
        edges = [(r.from_key, r.to_key) for r in relationships if r.rel_type in ALLOWED_RELATIONSHIP_TYPES]
        keys = sorted(all_graph_keys)
        if not keys:
            return
        comm = _union_find_communities(keys, edges)
        rows = [{"key": k, "community": comm[k]} for k in keys if k in comm]
        if not rows:
            return
        try:
            with driver.session() as session:
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (e:Entity {graph_key: row.key})
                    WHERE e.document_id = $d OR e.doc_id = $d
                    SET e.community_id = row.community
                    """,
                    rows=rows,
                    d=document_id,
                )
        except Exception as e:
            logger.warning("Community assignment failed: %s", e)

    async def _build_graph_async(
        self,
        chunks: list[dict],
        document_id: str,
        on_progress: Callable[[int, int, int, int], None] | None,
        on_graph_snapshot: Callable[[dict], None] | None,
        session_id: str | None,
    ) -> dict[str, Any]:
        doc_id = document_id
        n = len(chunks)
        stats = {"chunks_processed": 0, "entities_created": 0, "relations_created": 0}
        cache: dict[str, dict[str, Any]] = {}
        cache_max = max(256, self.settings.graph_llm_cache_max_entries)
        concurrency = max(1, self.settings.graph_extract_concurrency)
        if self.settings.llm_provider.lower() == "ollama":
            concurrency = min(concurrency, 2)  # Ollama struggles with 5+ concurrent requests

        def push_stream(event_type: str, data: dict[str, Any]) -> None:
            if not session_id:
                return
            try:
                from services.stream_hub import StreamHub

                StreamHub.push(session_id, event_type, {**data, "document_id": doc_id})
            except Exception:
                pass

        driver = self._get_driver()
        entities_acc = 0
        rels_acc = 0
        all_relationships: list[RelRow] = []
        all_keys: set[str] = set()

        if not driver:
            logger.info("Neo4j not configured — running extraction only for stats")
            for i, ch in enumerate(chunks):
                pc = extract_chunk_sync(i, ch, doc_id, cache, cache_max)
                stats["chunks_processed"] += 1
                stats["entities_created"] += len(pc.mentions)
                stats["relations_created"] += len(pc.relationships)
            return stats

        sem = asyncio.Semaphore(concurrency)

        async def extract_safe(idx: int, ch: dict) -> ProcessedChunk:
            async with sem:
                try:
                    return await asyncio.to_thread(
                        extract_chunk_sync,
                        idx,
                        ch,
                        doc_id,
                        cache,
                        cache_max,
                    )
                except Exception as e:
                    logger.warning("Chunk %s extract failed: %s", idx, e)
                    cid = str(ch.get("chunk_id") or uuid.uuid4())
                    return ProcessedChunk(
                        chunk_index=idx,
                        chunk_id=cid,
                        text=(ch.get("text") or "")[:10000],
                        source=str(ch.get("source") or ""),
                        page=int(ch.get("page") or 0),
                        mentions=[],
                        relationships=[],
                    )

        tasks = [asyncio.create_task(extract_safe(i, c)) for i, c in enumerate(chunks)]
        completed = 0
        processed_chunks: list[ProcessedChunk] = []

        try:
            for fut in asyncio.as_completed(tasks):
                pc = await fut
                processed_chunks.append(pc)
                for m in pc.mentions:
                    all_keys.add(m.graph_key)
                for r in pc.relationships:
                    all_keys.add(r.from_key)
                    all_keys.add(r.to_key)

                ne = await asyncio.to_thread(
                    self._persist_stage1_entities,
                    driver,
                    doc_id,
                    pc,
                    push_stream,
                )
                entities_acc += ne
                stats["entities_created"] += ne
                stats["chunks_processed"] += 1
                completed += 1
                if on_progress:
                    on_progress(completed, n, entities_acc, rels_acc)

                snap = await asyncio.to_thread(self._snapshot_with_driver, driver, doc_id)
                push_stream("graph_update", snap)
                if on_graph_snapshot:
                    on_graph_snapshot(snap)

            for pc in processed_chunks:
                all_relationships.extend(pc.relationships)

            seen_rel: set[tuple[str, str, str]] = set()
            deduped_rels: list[RelRow] = []
            for r in all_relationships:
                k = (r.from_key, r.to_key, r.rel_type)
                if k in seen_rel:
                    continue
                seen_rel.add(k)
                deduped_rels.append(r)
            all_relationships = deduped_rels

            nr = await asyncio.to_thread(
                self._persist_stage2_relationships,
                driver,
                doc_id,
                all_relationships,
                push_stream,
            )
            rels_acc += nr
            stats["relations_created"] += nr

            await asyncio.to_thread(
                self._apply_communities,
                driver,
                doc_id,
                all_relationships,
                all_keys,
            )

            snap = await asyncio.to_thread(self._snapshot_with_driver, driver, doc_id)
            push_stream("graph_update", snap)
            if on_graph_snapshot:
                on_graph_snapshot(snap)

        except Exception as e:
            logger.warning("Graph build async pipeline failed: %s", e)
            push_stream("error", {"phase": "graph_build", "message": str(e)})
            for t in tasks:
                if not t.done():
                    t.cancel()
            raise

        logger.info(
            "Graph built for doc %s: chunks=%s, entity rows=%s, rel rows=%s",
            doc_id[:8],
            stats["chunks_processed"],
            stats["entities_created"],
            stats["relations_created"],
        )
        return stats

    def build_graph(
        self,
        chunks: list[dict],
        document_id: str = "",
        on_progress: Callable[[int, int, int, int], None] | None = None,
        on_graph_snapshot: Callable[[dict], None] | None = None,
        session_id: str | None = None,
        skip_clear: bool = False,
    ) -> dict[str, Any]:
        doc_id = document_id or str(uuid.uuid4())
        if not skip_clear:
            self.clear_graph_for_document(doc_id)

        if not chunks:
            logger.warning("build_graph called with no chunks")
            return {"chunks_processed": 0, "entities_created": 0, "relations_created": 0}

        for c in chunks:
            if not c.get("chunk_id"):
                c["chunk_id"] = str(uuid.uuid4())

        concurrency = max(1, self.settings.graph_extract_concurrency)
        if self.settings.llm_provider.lower() == "ollama":
            concurrency = min(concurrency, 2)
        logger.info(
            "Building graph: %s chunks, concurrency=%s, doc=%s…",
            len(chunks),
            concurrency,
            doc_id[:8],
        )

        try:
            return asyncio.run(
                self._build_graph_async(
                    chunks,
                    doc_id,
                    on_progress,
                    on_graph_snapshot,
                    session_id,
                )
            )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(
                    self._build_graph_async(
                        chunks,
                        doc_id,
                        on_progress,
                        on_graph_snapshot,
                        session_id,
                    )
                )
            finally:
                loop.close()
