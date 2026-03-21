"""
Phase 3: Query Engine - Graph-first RAG.
FAISS returns chunk_ids -> Neo4j queried by chunk_ids.
Production output: answer, unique citations, graph trace, reasoning steps, confidence.
"""
import logging
import re
import time
from typing import Any

import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator

from config import get_settings

logger = logging.getLogger("graph_rag.services.query_engine")

CITATION_MAX = 5
CITATION_DEDUP_SIM_THRESHOLD = 0.92  # cosine sim >= this = duplicate


class QueryEngine:
    """
    Pipeline:
    1. Normalize query (-> English)
    2. FAISS semantic search -> chunk_ids
    3. Neo4j by chunk_ids: MATCH (c:Chunk)-[:MENTIONS]->(e) WHERE c.id IN $chunk_ids
    4. Optional: LLM-generated Cypher for relationship queries
    5. Merge context -> LLM answer + citations
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    def query(self, user_query: str, document_id: str | None = None) -> dict[str, Any]:
        timeline = []
        t0 = time.time()
        
        normalized = self._normalize(user_query)
        timeline.append({"step": "normalize", "duration_ms": (time.time() - t0) * 1000})
        
        t1 = time.time()
        from services.vector_store import VectorStoreService
        vs = VectorStoreService()
        faiss_hits = vs.search(normalized, k=5, document_id=document_id)
        chunk_ids = [h["chunk_id"] for h in faiss_hits if h.get("chunk_id")]
        timeline.append({"step": "faiss_search", "duration_ms": (time.time() - t1) * 1000})
        
        t2 = time.time()
        graph_result = (
            self._query_neo4j_by_chunk_ids(chunk_ids, document_id=document_id)
            if chunk_ids
            else {"nodes": [], "chunks": [], "edges": []}
        )
        if not graph_result.get("chunks") and faiss_hits:
            chunks_used = faiss_hits
        else:
            chunks_used = graph_result.get("chunks", [])
            if faiss_hits:
                seen = {c.get("chunk_id"): c for c in chunks_used}
                for h in faiss_hits:
                    cid = h.get("chunk_id")
                    if cid and cid not in seen:
                        seen[cid] = h
                        chunks_used.append(h)
        timeline.append({"step": "neo4j_by_chunk_ids", "duration_ms": (time.time() - t2) * 1000})
        
        context = self._merge_context(chunks_used)
        
        t3 = time.time()
        answer, citations, confidence = self._generate_answer(normalized, context, chunks_used)
        timeline.append({"step": "generate_answer", "duration_ms": (time.time() - t3) * 1000})

        cypher_preview = (
            "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE c.id IN $chunk_ids"
            + (
                f" AND (c.document_id = '{document_id}' OR c.doc_id = '{document_id}')"
                if document_id
                else ""
            )
            + " RETURN c, e"
        )
        
        # Deduplicate citations: chunk_id + embedding similarity, limit to top 3-5
        citations_deduped = self._deduplicate_citations(chunks_used)[:CITATION_MAX]
        citations_structured = [
            {
                "chunk_id": c.get("chunk_id", ""),
                "page": int(c.get("page", 0)),
                "source": c.get("source", ""),
                "text": (c.get("text") or c.get("content", ""))[:2000],
            }
            for c in citations_deduped
        ]

        # Graph trace: only nodes and edges from retrieval (subgraph used in answering)
        used_node_ids = {str(n.get("id", "")) for n in graph_result.get("nodes", []) if n.get("id")}
        graph_trace_nodes = [
            {
                "id": n.get("id"),
                "name": (n.get("props") or {}).get("name", n.get("id", "")),
                "group": 2,
                "labels": n.get("labels", []),
                "used": True,
            }
            for n in graph_result.get("nodes", [])
            if n.get("id")
        ]
        graph_trace_edges = list(graph_result.get("edges", []))

        retrieval_graph = {
            "nodes": [
                {"id": n.get("id"), "name": (n.get("props") or {}).get("name", n.get("id", "")), "group": 2, "labels": n.get("labels", [])}
                for n in graph_result.get("nodes", []) if n.get("id")
            ],
            "links": graph_result.get("edges", []),
        }

        reasoning_steps = [
            {"step": t.get("step", ""), "duration_ms": round(t.get("duration_ms", 0)), "description": self._step_description(t.get("step", ""))}
            for t in timeline
        ]

        return {
            "answer": answer,
            "citations": citations_structured,
            "graph_trace": {"nodes": graph_trace_nodes, "edges": graph_trace_edges, "used_node_ids": list(used_node_ids)},
            "graph_nodes": graph_result.get("nodes", []),
            "graph_edges": graph_result.get("edges", []),
            "retrieval_graph": retrieval_graph,
            "chunks_used": chunks_used,
            "reasoning_steps": reasoning_steps,
            "confidence": confidence,
            "cypher_query": cypher_preview,
            "processing_timeline": timeline,
        }
    
    def _step_description(self, step: str) -> str:
        """Human-readable step description."""
        desc = {
            "normalize": "Query translated/normalized to English",
            "faiss_search": "Semantic search in FAISS for relevant chunks",
            "neo4j_by_chunk_ids": "Neo4j graph expansion by chunk IDs",
            "generate_answer": "LLM synthesized answer with citations",
        }
        return desc.get(step, step)

    def _deduplicate_citations(self, chunks: list[dict]) -> list[dict]:
        """Deduplicate by chunk_id and embedding similarity. Keep top 3-5 unique chunks."""
        if not chunks:
            return []
        seen_ids: set[str] = set()
        unique: list[dict] = []
        unique_embeddings: list[np.ndarray] = []
        try:
            from services.embedder import get_embedder
            embedder = get_embedder()
        except Exception as e:
            logger.debug(f"Embedder unavailable for citation dedup: {e}")
            embedder = None
        for c in chunks:
            cid = c.get("chunk_id", "")
            if cid and cid in seen_ids:
                continue
            text = (c.get("text") or c.get("content") or "").strip()
            if not text:
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    unique.append(c)
                continue
            if embedder:
                try:
                    emb = embedder.encode([text], normalize_embeddings=True)
                    for u_emb in unique_embeddings:
                        sim = float(np.dot(emb[0], u_emb))
                        if sim >= CITATION_DEDUP_SIM_THRESHOLD:
                            break
                    else:
                        unique_embeddings.append(emb[0].copy())
                        if cid:
                            seen_ids.add(cid)
                        unique.append(c)
                        if len(unique) >= CITATION_MAX:
                            break
                    continue
                except Exception as e:
                    logger.debug(f"Citation embedding dedup skip: {e}")
            if cid:
                seen_ids.add(cid)
            unique.append(c)
            if len(unique) >= CITATION_MAX:
                break
        return unique

    def _normalize(self, text: str) -> str:
        try:
            lang = detect(text)
            if lang != "en":
                return GoogleTranslator(source=lang, target="en").translate(text)
        except Exception:
            pass
        return text
    
    def _query_neo4j_by_chunk_ids(self, chunk_ids: list[str], document_id: str | None = None) -> dict:
        """FAISS -> Graph. Optional document_id keeps results on the active PDF only."""
        if not chunk_ids:
            return {"nodes": [], "chunks": [], "edges": []}
        try:
            from neo4j import GraphDatabase
            s = get_settings()
            if not s.neo4j_uri:
                return {"nodes": [], "chunks": [], "edges": []}
            driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user_resolved, s.neo4j_password))
            nodes = []
            chunks = []
            seen_chunk_ids = set()
            
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
                entity_keys: set[str] = set()
                for record in result:
                    c_node = record.get("c")
                    e_node = record.get("e")
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
                    if e_node:
                        props = dict(e_node)
                        gk = props.get("graph_key") or props.get("name") or ""
                        if gk:
                            entity_keys.add(str(gk))
                        nodes.append({
                            "id": str(gk) if gk else getattr(e_node, "element_id", str(id(e_node))),
                            "labels": list(e_node.labels) if hasattr(e_node, "labels") else [],
                            "props": props,
                        })
                edges: list[dict] = []
                if entity_keys and document_id:
                    er = session.run(
                        """
                        MATCH (a:Entity)-[r]->(b:Entity)
                        WHERE a.graph_key IN $keys AND b.graph_key IN $keys
                          AND (a.document_id = $document_id OR a.doc_id = $document_id)
                          AND (b.document_id = $document_id OR b.doc_id = $document_id)
                        RETURN a.graph_key AS s, b.graph_key AS t, type(r) AS rel
                        """,
                        keys=list(entity_keys),
                        document_id=document_id,
                    )
                    seen_e: set[tuple[str, str, str]] = set()
                    for row in er:
                        s, t, rel = row.get("s"), row.get("t"), row.get("rel")
                        if s and t:
                            key = (str(s), str(t), str(rel or "REL"))
                            if key not in seen_e:
                                seen_e.add(key)
                                edges.append({"source": str(s), "target": str(t), "type": str(rel or "REL")})
                return {"nodes": nodes, "chunks": chunks, "edges": edges}
        except Exception as e:
            logger.warning(f"Neo4j by chunk_ids failed: {e}")
            return {"nodes": [], "chunks": [], "edges": []}
    
    def _merge_context(self, chunks: list) -> str:
        """Merge chunk context for LLM."""
        parts = []
        for c in chunks[:10]:
            text = c.get("text", c.get("content", ""))
            if text:
                parts.append(f"[Source: {c.get('source', '')}]\n{text}")
        return "\n\n---\n\n".join(parts) if parts else "No context found."
    
    def _fallback_answer_from_chunks(self, query: str, chunks: list) -> str:
        """
        When the LLM is unavailable or returns empty, still return useful text from retrieval
        so the UI is not a dead end (Graph RAG already found evidence).
        """
        if not chunks:
            return (
                "No answer could be generated: nothing was retrieved, and the LLM did not respond.\n\n"
                "To enable synthesis: run `ollama serve` and `ollama pull llama3.1:8b`, or set "
                "`GEMINI_API_KEY` / OpenAI keys in backend `.env`."
            )

        q_terms = {w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", (query or "").lower())}
        ranked = list(chunks[:12])
        if q_terms:
            def score(c: dict) -> int:
                t = (c.get("text") or c.get("content") or "").lower()
                return sum(1 for w in q_terms if w in t)

            ranked.sort(key=score, reverse=True)

        lines = [
            "Retrieved evidence (LLM unavailable — showing top passages from your document):",
            "",
        ]
        for c in ranked[:4]:
            text = (c.get("text") or c.get("content") or "").strip()
            if not text:
                continue
            src = c.get("source", "source")
            page = c.get("page", 0)
            excerpt = text[:720].rstrip()
            if len(text) > 720:
                excerpt += "…"
            lines.append(f"• [{src} · page {page}]")
            lines.append(f"  {excerpt}")
            lines.append("")

        lines.append(
            "—\nTo get a synthesized answer, start Ollama or add a cloud LLM key in backend `.env`."
        )
        return "\n".join(lines).strip()

    def _clean_answer_text(self, text: str) -> str:
        """Remove inline citation markup; clean trailing raw table-like data."""
        if not text:
            return text
        text = re.sub(r"\[Source:\s*[^\]]+\]", "", text, flags=re.I)
        text = re.sub(r"Cited source:\s*\[?[^\]]*\]?", "", text, flags=re.I)
        text = re.sub(r"\(Source:\s*[^)]+\)", "", text)
        # Remove trailing raw table rows (e.g. "year Income cost 2021 80 50")
        text = re.sub(r"\n\s*(year|Year)\s+[A-Za-z]+\s+[A-Za-z]+\s+\d{4}\s+\d+\s+\d+.*$", "", text, flags=re.I | re.DOTALL)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _generate_answer(self, query: str, context: str, chunks: list) -> tuple[str, list, float]:
        """LLM generates answer with citations; falls back to chunk excerpts if LLM is down."""
        from services.llm_client import LLMClient
        llm = LLMClient()

        chunk_refs = [{"source": c.get("source", ""), "page": c.get("page", 0), "chunk_id": c.get("chunk_id", "")} for c in chunks]

        prompt = f"""Answer the question using ONLY the provided context. Do not hallucinate.
If the context does not contain the answer, say so.

Context:
{context}

Question: {query}

Provide a clear, well-formatted answer:
1. Start with a direct answer in 1-2 sentences.
2. Add a "Details:" section with bullet points or short paragraphs if needed.
3. Do NOT include inline citations like [Source: page_1] or "Cited source:" in the answer text.
4. Citations will be displayed separately in the UI.
"""
        answer = (llm.generate(prompt, max_tokens=1024) or "").strip()
        answer = self._clean_answer_text(answer)
        has_context = bool(context.strip()) and context.strip() != "No context found."

        if answer:
            confidence = 0.85 if has_context else 0.35
        elif has_context and chunks:
            logger.warning("Query LLM returned empty; using retrieval-only fallback answer")
            answer = self._fallback_answer_from_chunks(query, chunks)
            confidence = 0.48
        else:
            answer = self._fallback_answer_from_chunks(query, chunks)
            confidence = 0.28

        citations = chunk_refs[:5]
        return answer, citations, confidence
