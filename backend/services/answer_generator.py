"""
Production-grade answer generator.
Always synthesizes from graph triples + chunks. Never "no direct explanation".
"""
import logging
import re
from typing import Any

from config import get_settings

logger = logging.getLogger("graph_rag.services.answer_generator")


def _format_graph_triples(edges: list[dict], nodes: list[dict]) -> list[str]:
    """Convert graph edges to human-readable triples for LLM context."""
    name_by_id: dict[str, str] = {}
    for n in (nodes or []):
        nid = n.get("id") or n.get("graph_key", "")
        name = (n.get("props") or {}).get("name", nid) or nid
        if nid:
            name_by_id[str(nid)] = str(name).split("::")[-1] if "::" in str(name) else str(name)
    triples: list[str] = []
    for e in (edges or []):
        s = e.get("source", "")
        t = e.get("target", "")
        rel = e.get("type", "REL")
        s_name = name_by_id.get(str(s), s)
        t_name = name_by_id.get(str(t), t)
        if s_name and t_name:
            triples.append(f"({s_name})-[{rel}]->({t_name})")
    return triples[:30]  # Limit for latency


def _compress_chunks(chunks: list[dict], max_chunks: int = 3, max_chars: int = 400) -> list[dict]:
    """Top N chunks, truncated. Remove redundancy."""
    out: list[dict] = []
    seen: set[str] = set()
    for c in chunks[:max_chunks]:
        text = (c.get("text") or c.get("content") or "").strip()
        if not text:
            continue
        fp = re.sub(r"\s+", " ", text[:150].lower())
        if fp in seen:
            continue
        seen.add(fp)
        excerpt = text[:max_chars].rstrip()
        if len(text) > max_chars:
            excerpt += "…"
        out.append({
            "chunk_id": c.get("chunk_id", ""),
            "page": int(c.get("page", 0)),
            "source": c.get("source", ""),
            "text": excerpt,
        })
    return out


def _build_synthesis_prompt(
    query: str,
    graph_triples: list[str],
    chunks: list[dict],
) -> str:
    """Build compressed prompt for fast LLM. Multi-concept: ALWAYS synthesize."""
    lines = [
        "You are a document assistant. Answer using ONLY the context below. NEVER say 'no direct explanation' or 'no direct link'.",
        "",
        "## Graph relationships (use these to reason):",
    ]
    if graph_triples:
        for t in graph_triples:
            lines.append(f"- {t}")
    else:
        lines.append("- (none extracted)")
    lines.extend(["", "## Relevant passages:"])
    for c in chunks:
        src = c.get("source", "source")
        page = c.get("page", 0)
        text = c.get("text", "")
        lines.append(f"[{src} p{page}]")
        lines.append(text)
        lines.append("")
    lines.extend([
        "## Question:",
        query,
        "",
        "## Instructions:",
        "1. ALWAYS synthesize an answer. Combine concepts from graph + passages.",
        "2. If multiple topics (e.g. financial trends AND AI usage), address BOTH and relate them logically.",
        "3. Direct answer first (1-2 sentences), then Details if needed.",
        "4. No inline citations. Write clean prose.",
        "5. Max 150 words.",
        "6. NEVER say 'no direct explanation', 'no direct link', or 'no direct connection'. Always synthesize from available context.",
    ])
    return "\n".join(lines)


def generate_answer(
    query: str,
    chunks: list[dict],
    graph_edges: list[dict],
    graph_nodes: list[dict],
    chunk_entities: dict[str, list[str]] | None = None,
    chunk_relationships: dict[str, list[dict]] | None = None,
) -> tuple[str, list[dict], list[str]]:
    """
    Synthesize answer from graph + chunks.
    Returns (answer, citations, reasoning_steps).
    """
    settings = get_settings()
    max_chunks = getattr(settings, "query_context_max_chunks", 3)
    max_chars = getattr(settings, "query_context_max_chars_per_chunk", 400)

    # Compress context for latency
    compressed = _compress_chunks(chunks, max_chunks=max_chunks, max_chars=max_chars)
    triples = _format_graph_triples(graph_edges, graph_nodes)

    reasoning_steps: list[str] = []
    if triples:
        reasoning_steps.append("Retrieved graph relationships")
    if compressed:
        reasoning_steps.append(f"Retrieved {len(compressed)} relevant passages")
    if len(compressed) >= 2 or triples:
        reasoning_steps.append("Combined graph + passages into unified explanation")

    prompt = _build_synthesis_prompt(query, triples, compressed)

    from services.llm_client import LLMClient
    llm = LLMClient()
    answer = (llm.generate(prompt, max_tokens=512) or "").strip()
    if not answer:
        answer = _fallback_from_chunks(query, compressed or chunks[:3])

    answer = _clean_answer(answer)

    # Citations: full passages + graph nodes/relationships from each source
    citations = _build_citations(
        chunks[:5],
        chunk_entities=chunk_entities or {},
        chunk_relationships=chunk_relationships or {},
        graph_nodes=graph_nodes or [],
        graph_edges=graph_edges or [],
    )

    return answer, citations, reasoning_steps


def _fallback_from_chunks(query: str, chunks: list[dict]) -> str:
    """Retrieval-only fallback. Synthesize from passages, never say 'no direct explanation'."""
    if not chunks:
        return "No relevant context was retrieved. Try refining your query or uploading a document."
    parts = []
    for c in chunks[:3]:
        text = (c.get("text") or c.get("content") or "").strip()
        if text:
            parts.append(text[:350])
    context = "\n\n".join(parts)
    summary = context[:600] + ("…" if len(context) > 600 else "")
    return f"Based on the retrieved passages:\n\n{summary}"


def _clean_answer(text: str) -> str:
    """Remove citation markup."""
    if not text:
        return text
    text = re.sub(r"\[Source:\s*[^\]]+\]", "", text, flags=re.I)
    text = re.sub(r"\[.*?p\d+.*?\]", "", text, flags=re.I)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_citations(
    chunks: list[dict],
    max_n: int = 5,
    chunk_entities: dict[str, list[str]] | None = None,
    chunk_relationships: dict[str, list[dict]] | None = None,
    graph_nodes: list[dict] | None = None,
    graph_edges: list[dict] | None = None,
) -> list[dict]:
    """
    Detailed citations: chunk_id, page, full passage text,
    graph nodes (entities), and relationships used from this source.
    """
    chunk_entities = chunk_entities or {}
    chunk_relationships = chunk_relationships or {}
    name_by_id: dict[str, str] = {}
    for n in (graph_nodes or []):
        nid = n.get("id") or n.get("graph_key", "")
        name = (n.get("props") or {}).get("name", nid) or nid
        if nid:
            name_by_id[str(nid)] = str(name).split("::")[-1] if "::" in str(name) else str(name)

    seen_ids: set[str] = set()
    out: list[dict] = []
    for c in chunks:
        cid = c.get("chunk_id", "")
        if cid and cid in seen_ids:
            continue
        text = (c.get("text") or c.get("content") or "").strip()
        # Full passage (up to 800 chars for depth)
        passage = text[:800].rstrip() + ("…" if len(text) > 800 else "")
        entities = chunk_entities.get(cid, [])
        entity_names = [name_by_id.get(str(e), e) for e in entities if e]
        rels_raw = chunk_relationships.get(cid, [])
        relationships = []
        for r in rels_raw:
            s = r.get("source", "")
            t = r.get("target", "")
            typ = r.get("type", "REL")
            s_name = name_by_id.get(str(s), s)
            t_name = name_by_id.get(str(t), t)
            if s_name and t_name:
                relationships.append(f"({s_name})-[{typ}]->({t_name})")
        seen_ids.add(cid or str(len(out)))
        out.append({
            "chunk_id": cid,
            "page": int(c.get("page", 0)),
            "source": c.get("source", ""),
            "text": passage,
            "graph_entities": entity_names[:20],
            "graph_relationships": relationships[:15],
        })
        if len(out) >= max_n:
            break
    return out
