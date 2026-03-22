"""
Production-grade Graph RAG Query Engine.
Always synthesize, never just retrieve. Graph-aware reasoning, real confidence, explainability.
"""
import logging
import re
import time
from typing import Any

import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator

from config import get_settings

from services.answer_generator import generate_answer

logger = logging.getLogger("graph_rag.services.query_engine")

CITATION_MAX = 5
CITATION_DEDUP_SIM_THRESHOLD = 0.92


class QueryEngine:
    """
    Pipeline:
    1. Normalize query
    2. Multi-concept: expand retrieval for multi-topic queries
    3. FAISS + optional keyword boost
    4. Neo4j graph (chunks + entity edges)
    5. Answer generator: graph triples + top 3 chunks -> synthesize
    6. Confidence scoring, reasoning trace, graph trace
    """

    def __init__(self):
        self.settings = get_settings()

    def query(self, user_query: str, document_id: str | None = None) -> dict[str, Any]:
        t0 = time.time()
        timeline: list[dict] = []

        # 1. Normalize
        t = time.time()
        normalized = self._normalize(user_query)
        timeline.append({"step": "normalize", "duration_ms": (time.time() - t) * 1000})

        # 2. Retrieve
        t = time.time()
        from services.vector_store import VectorStoreService
        vs = VectorStoreService()
        retrieval_k = int(getattr(self.settings, "query_retrieval_k", 8))
        is_multi_concept = self._is_multi_concept(normalized)
        if is_multi_concept:
            retrieval_k = min(retrieval_k + 4, 14)
        query_terms = self._extract_query_terms(normalized)
        keyword_hits = []
        if query_terms and document_id:
            keyword_hits = vs.search_by_keywords(query_terms, document_id=document_id, k=retrieval_k)
        faiss_hits = vs.search(normalized, k=retrieval_k, document_id=document_id)
        seen_ids = set()
        merged_hits = []
        for h in keyword_hits:
            cid = h.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                merged_hits.append(h)
        for h in faiss_hits:
            cid = h.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                merged_hits.append(h)
        faiss_hits = merged_hits[:retrieval_k]
        chunk_ids = [h["chunk_id"] for h in faiss_hits if h.get("chunk_id")]
        timeline.append({"step": "faiss_search", "duration_ms": (time.time() - t) * 1000})

        # 3. Graph
        t = time.time()
        graph_result = (
            self._query_neo4j_by_chunk_ids(chunk_ids, document_id=document_id)
            if chunk_ids
            else {"nodes": [], "chunks": [], "edges": []}
        )
        chunks_used = graph_result.get("chunks", [])
        if not chunks_used and faiss_hits:
            chunks_used = [
                {
                    "chunk_id": h.get("chunk_id"),
                    "text": (h.get("text") or "")[:2000],
                    "source": h.get("source", ""),
                    "page": h.get("page", 0),
                }
                for h in faiss_hits
            ]
        else:
            seen = {c.get("chunk_id"): c for c in chunks_used}
            for h in faiss_hits:
                cid = h.get("chunk_id")
                if cid and cid not in seen:
                    seen[cid] = {
                        "chunk_id": cid,
                        "text": (h.get("text") or "")[:2000],
                        "source": h.get("source", ""),
                        "page": h.get("page", 0),
                    }
                    chunks_used.append(seen[cid])
        timeline.append({"step": "neo4j_graph", "duration_ms": (time.time() - t) * 1000})

        chunks_used = self._deduplicate_chunks(chunks_used)

        # 4. Generate answer (graph triples + top chunks)
        t = time.time()
        answer, citations, reasoning_steps = generate_answer(
            normalized,
            chunks_used,
            graph_result.get("edges", []),
            graph_result.get("nodes", []),
            chunk_entities=graph_result.get("chunk_entities", {}),
            chunk_relationships=graph_result.get("chunk_relationships", {}),
        )
        timeline.append({"step": "generate_answer", "duration_ms": (time.time() - t) * 1000})

        # 5. Confidence score
        confidence = self._compute_confidence(
            graph_result, chunks_used, faiss_hits, citations
        )

        # 6. Graph trace (nodes + edges used)
        graph_trace = self._build_graph_trace(graph_result)

        # 7. Reasoning steps (human-readable, with duration)
        reasoning_steps_full: list[dict] = []
        for i, desc in enumerate(reasoning_steps):
            reasoning_steps_full.append({
                "step": str(i + 1),
                "description": desc,
                "duration_ms": 0,
            })
        for t_entry in timeline:
            reasoning_steps_full.append({
                "step": t_entry.get("step", ""),
                "description": self._step_description(t_entry.get("step", "")),
                "duration_ms": round(t_entry.get("duration_ms", 0)),
            })

        retrieval_graph = {
            "nodes": [
                {"id": n.get("id"), "name": (n.get("props") or {}).get("name", n.get("id", ""))}
                for n in graph_result.get("nodes", []) if n.get("id")
            ],
            "links": graph_result.get("edges", []),
        }
        return {
            "answer": answer,
            "reasoning_steps": reasoning_steps_full,
            "citations": citations[:CITATION_MAX],
            "graph_trace": graph_trace,
            "confidence": round(confidence, 2),
            "retrieval_graph": retrieval_graph,
            "graph_nodes": graph_result.get("nodes", []),
            "graph_edges": graph_result.get("edges", []),
            "chunks_used": chunks_used,
            "processing_timeline": timeline,
            "cypher_query": "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE c.id IN $chunk_ids RETURN c, e",
        }

    def _extract_query_terms(self, query: str) -> list[str]:
        """Years + metrics for hybrid keyword search."""
        if not (query or "").strip():
            return []
        terms = []
        for m in re.finditer(r"\b(19\d{2}|20\d{2})\b", query):
            terms.append(m.group(1))
        for m in re.finditer(
            r"\b(revenue|cost|profit|earnings|margin|sales|income|expense)\b",
            query, re.I
        ):
            terms.append(m.group(1).lower())
        return list(dict.fromkeys(terms))

    def _is_multi_concept(self, query: str) -> bool:
        """Detect multi-topic queries (and, together, both, etc.)."""
        if not query:
            return False
        q = query.lower()
        return bool(
            re.search(r"\b(and|together|both|combined|also|additionally)\b", q)
            or "trends" in q and ("ai" in q or "usage" in q)
        )

    def _compute_confidence(
        self,
        graph_result: dict,
        chunks: list,
        faiss_hits: list,
        citations: list,
    ) -> float:
        """
        Real confidence: graph_match*0.4 + chunk_similarity*0.3 + citation_strength*0.3
        """
        graph_match = 0.0
        if graph_result.get("edges"):
            graph_match = min(1.0, len(graph_result["edges"]) / 10.0)
        if graph_result.get("nodes"):
            graph_match = max(graph_match, min(1.0, len(graph_result["nodes"]) / 15.0))

        chunk_sim = 0.0
        if faiss_hits:
            scores = [float(h.get("score", 0)) for h in faiss_hits if "score" in h]
            if scores:
                chunk_sim = min(1.0, max(scores))

        citation_strength = min(1.0, len(citations) / 5.0) if citations else 0.0

        base = graph_match * 0.4 + chunk_sim * 0.3 + citation_strength * 0.3
        return min(1.0, max(0.2, base))  # Floor 0.2, cap 1.0

    def _build_graph_trace(self, graph_result: dict) -> dict:
        """Graph path used in reasoning: nodes + edges."""
        nodes = graph_result.get("nodes", [])
        edges = graph_result.get("edges", [])
        used_ids = {str(n.get("id", "")) for n in nodes if n.get("id")}
        for e in edges:
            used_ids.add(str(e.get("source", "")))
            used_ids.add(str(e.get("target", "")))
        return {
            "nodes": [
                {
                    "id": n.get("id"),
                    "name": (n.get("props") or {}).get("name", n.get("id", "")),
                    "used": True,
                }
                for n in nodes if n.get("id")
            ],
            "edges": [
                {
                    "source": e.get("source"),
                    "target": e.get("target"),
                    "type": e.get("type", "REL"),
                }
                for e in edges
            ],
            "used_node_ids": list(used_ids),
        }

    def _deduplicate_chunks(self, chunks: list[dict]) -> list[dict]:
        """Remove near-duplicate chunks by text similarity."""
        if len(chunks) <= 1:
            return chunks
        seen: set[str] = set()
        unique: list[dict] = []
        for c in chunks:
            text = (c.get("text") or c.get("content") or "").strip()[:400]
            if not text:
                unique.append(c)
                continue
            fp = re.sub(r"\s+", " ", text.lower())
            if fp in seen:
                continue
            seen.add(fp)
            unique.append(c)
        return unique

    def _deduplicate_citations(self, chunks: list[dict]) -> list[dict]:
        """Deduplicate by chunk_id and embedding similarity. Max 5."""
        if not chunks:
            return []
        seen_ids: set[str] = set()
        unique: list[dict] = []
        try:
            from services.embedder import get_embedder
            embedder = get_embedder()
        except Exception:
            embedder = None
        for c in chunks:
            cid = c.get("chunk_id", "")
            if cid and cid in seen_ids:
                continue
            text = (c.get("text") or c.get("content") or "").strip()
            if embedder and text:
                try:
                    emb = embedder.encode([text], normalize_embeddings=True)
                    for u in unique:
                        u_text = (u.get("text") or u.get("content") or "").strip()
                        if not u_text:
                            continue
                        u_emb = embedder.encode([u_text], normalize_embeddings=True)
                        if float(np.dot(emb[0], u_emb[0])) >= CITATION_DEDUP_SIM_THRESHOLD:
                            break
                    else:
                        seen_ids.add(cid or "")
                        unique.append(c)
                        if len(unique) >= CITATION_MAX:
                            break
                    continue
                except Exception:
                    pass
            seen_ids.add(cid or "")
            unique.append(c)
            if len(unique) >= CITATION_MAX:
                break
        return unique

    def _step_description(self, step: str) -> str:
        desc = {
            "normalize": "Query translated/normalized to English",
            "faiss_search": "Semantic + keyword search for relevant chunks",
            "neo4j_graph": "Neo4j graph expansion (chunks + entity edges)",
            "generate_answer": "LLM synthesized from graph triples + passages",
        }
        return desc.get(step, step)

    def _normalize(self, text: str) -> str:
        if not text or not text.strip():
            return text
        t = text.strip()
        t = re.sub(r"\bteh\b", "the", t, flags=re.I)
        try:
            if detect(t) != "en":
                return GoogleTranslator(source="auto", target="en").translate(t)
        except Exception:
            pass
        return t

    def _query_neo4j_by_chunk_ids(self, chunk_ids: list[str], document_id: str | None = None) -> dict:
        if not chunk_ids:
            return {"nodes": [], "chunks": [], "edges": []}
        try:
            from neo4j import GraphDatabase
            s = get_settings()
            if not s.neo4j_uri:
                return {"nodes": [], "chunks": [], "edges": []}
            driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user_resolved, s.neo4j_password))
            nodes: list[dict] = []
            chunks: list[dict] = []
            seen_chunk_ids: set[str] = set()
            entity_keys: set[str] = set()
            chunk_entities: dict[str, set[str]] = {}
            seen_node_ids: set[str] = set()

            with driver.session() as session:
                if document_id:
                    cypher = """
                    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                    WHERE c.id IN $chunk_ids
                      AND (c.document_id = $document_id OR c.doc_id = $document_id)
                      AND (e.document_id = $document_id OR e.doc_id = $document_id)
                    RETURN c, e
                    """
                    result = session.run(cypher, chunk_ids=chunk_ids, document_id=document_id)
                else:
                    result = session.run(
                        """
                        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                        WHERE c.id IN $chunk_ids
                        RETURN c, e
                        """,
                        chunk_ids=chunk_ids,
                    )
                for record in result:
                    c_node = record.get("c")
                    e_node = record.get("e")
                    cid = None
                    if c_node:
                        props = dict(c_node)
                        cid = props.get("id") or props.get("chunk_id")
                        if cid and cid not in seen_chunk_ids:
                            seen_chunk_ids.add(cid)
                            chunks.append({
                                "chunk_id": cid,
                                "text": props.get("text", ""),
                                "source": props.get("source", ""),
                                "page": props.get("page", 0),
                            })
                    if e_node and cid:
                        props = dict(e_node)
                        gk = props.get("graph_key") or props.get("name") or ""
                        if gk:
                            entity_keys.add(str(gk))
                            chunk_entities.setdefault(cid, set()).add(str(gk))
                        nid = str(gk) if gk else getattr(e_node, "element_id", str(id(e_node)))
                        if gk and nid not in seen_node_ids:
                            seen_node_ids.add(nid)
                            nodes.append({
                                "id": nid,
                                "labels": list(e_node.labels) if hasattr(e_node, "labels") else [],
                                "props": props,
                            })

                edges: list[dict] = []
                keys = list(entity_keys)
                if keys:
                    if document_id:
                        er = session.run(
                            """
                            MATCH (a:Entity)-[r]->(b:Entity)
                            WHERE a.graph_key IN $keys AND b.graph_key IN $keys
                              AND (a.document_id = $document_id OR a.doc_id = $document_id)
                              AND (b.document_id = $document_id OR b.doc_id = $document_id)
                            RETURN a.graph_key AS s, b.graph_key AS t, type(r) AS rel
                            """,
                            keys=keys,
                            document_id=document_id,
                        )
                    else:
                        er = session.run(
                            """
                            MATCH (a:Entity)-[r]->(b:Entity)
                            WHERE a.graph_key IN $keys AND b.graph_key IN $keys
                            RETURN a.graph_key AS s, b.graph_key AS t, type(r) AS rel
                            LIMIT 100
                            """,
                            keys=keys,
                        )
                    seen_e: set[tuple[str, str, str]] = set()
                    for row in er:
                        s, t, rel = row.get("s"), row.get("t"), row.get("rel")
                        if s and t:
                            key = (str(s), str(t), str(rel or "REL"))
                            if key not in seen_e:
                                seen_e.add(key)
                                edges.append({"source": str(s), "target": str(t), "type": str(rel or "REL")})
                # Map each chunk to relationships involving its entities
                chunk_relationships: dict[str, list[dict]] = {}
                for cid, ents in chunk_entities.items():
                    ent_set = set(ents)
                    for e in edges:
                        if e.get("source") in ent_set and e.get("target") in ent_set:
                            chunk_relationships.setdefault(cid, []).append(e)
                return {
                    "nodes": nodes,
                    "chunks": chunks,
                    "edges": edges,
                    "chunk_entities": {k: list(v) for k, v in chunk_entities.items()},
                    "chunk_relationships": chunk_relationships,
                }
        except Exception as e:
            logger.warning(f"Neo4j failed: {e}")
            return {"nodes": [], "chunks": [], "edges": [], "chunk_entities": {}, "chunk_relationships": {}}
