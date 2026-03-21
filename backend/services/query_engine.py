"""
Phase 3: Query Engine - Graph-first RAG.
FAISS returns chunk_ids -> Neo4j queried by chunk_ids.
MATCH (c:Chunk)-[:MENTIONS]->(e) WHERE c.id IN [...] RETURN c, e
"""
import logging
import re
import time
from typing import Any

from langdetect import detect
from deep_translator import GoogleTranslator

from config import get_settings

logger = logging.getLogger("graph_rag.services.query_engine")


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
        
        retrieval_graph = {
            "nodes": [
                {
                    "id": n.get("id"),
                    "name": (n.get("props") or {}).get("name", n.get("id", "")),
                    "group": 2,
                    "labels": n.get("labels", []),
                }
                for n in graph_result.get("nodes", [])
                if n.get("id")
            ],
            "links": graph_result.get("edges", []),
        }

        return {
            "answer": answer,
            "citations": citations,
            "graph_nodes": graph_result.get("nodes", []),
            "graph_edges": graph_result.get("edges", []),
            "retrieval_graph": retrieval_graph,
            "chunks_used": chunks_used,
            "confidence": confidence,
            "cypher_query": cypher_preview,
            "processing_timeline": timeline,
        }
    
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

Provide a clear answer and cite sources (source, page) where relevant.
"""
        answer = (llm.generate(prompt, max_tokens=1024) or "").strip()
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
