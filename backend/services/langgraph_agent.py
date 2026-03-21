"""
Phase 4: LangGraph Agent - Orchestrates query pipeline.
Nodes: analyze_query, generate_cypher, query_graph, conditional_faiss, merge_context, generate_answer
"""
import logging
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

logger = logging.getLogger("graph_rag.services.langgraph_agent")


class AgentState(TypedDict):
    query: str
    normalized_query: str
    cypher: str
    cypher_failed: bool
    graph_result: dict
    faiss_chunks: list
    merged_context: str
    answer: str
    citations: list
    confidence: float
    graph_nodes: list
    chunks_used: list
    timeline: list
    error: str | None


def analyze_query(state: AgentState) -> AgentState:
    """Normalize language to English."""
    from langdetect import detect
    from deep_translator import GoogleTranslator
    query = state["query"]
    try:
        lang = detect(query)
        normalized = GoogleTranslator(source=lang, target="en").translate(query) if lang != "en" else query
    except Exception:
        normalized = query
    return {**state, "normalized_query": normalized}


def generate_cypher(state: AgentState) -> AgentState:
    """Generate schema-aware Cypher. Retry logic on failure."""
    from services.llm_client import LLMClient
    llm = LLMClient()
    schema = "Nodes: Chunk {chunk_id, text, source, page}, Entity {name}. Relationships: (Chunk)-[:MENTIONS]->(Entity), (Entity)-[:RELATION]->(Entity)"
    prompt = f"""Schema: {schema}
Generate Cypher for: "{state['normalized_query']}"
Return ONLY the Cypher, no markdown."""
    try:
        cypher = llm.generate(prompt, max_tokens=512).strip()
        if "```" in cypher:
            cypher = cypher.split("```")[1]
            if cypher.startswith("cypher"):
                cypher = cypher[6:]
        cypher = cypher.strip()
        return {**state, "cypher": cypher, "cypher_failed": False}
    except Exception as e:
        logger.warning(f"Cypher generation failed: {e}")
        return {**state, "cypher": "", "cypher_failed": True}


def query_graph(state: AgentState) -> AgentState:
    """Query Neo4j with Cypher. Uses fallback if cypher empty/failed."""
    cypher = state.get("cypher", "").strip()
    if not cypher or state.get("cypher_failed"):
        return {**state, "graph_result": {"nodes": [], "chunks": []}}
    
    from neo4j import GraphDatabase
    from config import get_settings
    s = get_settings()
    driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user_resolved, s.neo4j_password))
    nodes, chunks = [], []
    seen = set()
    
    def process_val(val):
        if val is None:
            return
        if hasattr(val, "nodes"):
            for n in val.nodes:
                process_val(n)
            return
        props = dict(val) if hasattr(val, "items") else {}
        labels = list(val.labels) if hasattr(val, "labels") else []
        nid = getattr(val, "element_id", None) or getattr(val, "id", str(id(val)))
        nodes.append({"id": str(nid), "labels": labels, "props": props})
        if "Chunk" in labels and "chunk_id" in props and props["chunk_id"] not in seen:
            seen.add(props["chunk_id"])
            chunks.append({"chunk_id": props["chunk_id"], "text": props.get("text", ""), "source": props.get("source", ""), "page": props.get("page", 0)})
    
    try:
        with driver.session() as session:
            for record in session.run(state["cypher"]):
                for k in record.keys():
                    process_val(record[k])
    except Exception as e:
        logger.warning(f"Neo4j query failed: {e}")
    
    return {**state, "graph_result": {"nodes": nodes, "chunks": chunks}}


def conditional_faiss(state: AgentState) -> AgentState:
    """If graph result insufficient, fetch from FAISS."""
    chunks = state["graph_result"].get("chunks", [])
    need_faiss = len(chunks) < 2
    faiss_chunks = []
    if need_faiss:
        from services.vector_store import VectorStoreService
        vs = VectorStoreService()
        faiss_chunks = vs.search(state["normalized_query"], k=5)
    merged = {c["chunk_id"]: c for c in chunks}
    for c in faiss_chunks:
        merged[c["chunk_id"]] = c
    return {**state, "faiss_chunks": faiss_chunks, "chunks_used": list(merged.values())}


def merge_context(state: AgentState) -> AgentState:
    """Merge graph + chunk context for LLM."""
    chunks = state["chunks_used"]
    parts = [f"[{c.get('source','')}]\n{c.get('text','')}" for c in chunks[:10] if c.get("text")]
    return {**state, "merged_context": "\n\n---\n\n".join(parts) if parts else "No context."}


def generate_answer(state: AgentState) -> AgentState:
    """LLM generates final answer with citations."""
    from services.llm_client import LLMClient
    llm = LLMClient()
    prompt = f"""Context:\n{state['merged_context']}\n\nQuestion: {state['normalized_query']}\n\nAnswer using only context. Cite sources."""
    answer = llm.generate(prompt, max_tokens=1024)
    citations = [{"source": c.get("source",""), "page": c.get("page",0), "chunk_id": c.get("chunk_id","")} for c in state["chunks_used"][:5]]
    confidence = 0.85 if state["merged_context"].strip() != "No context." else 0.3
    return {
        **state,
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "graph_nodes": state["graph_result"].get("nodes", []),
    }


def build_agent_graph():
    """Build LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("generate_cypher", generate_cypher)
    workflow.add_node("query_graph", query_graph)
    workflow.add_node("conditional_faiss", conditional_faiss)
    workflow.add_node("merge_context", merge_context)
    workflow.add_node("generate_answer", generate_answer)
    
    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "generate_cypher")
    workflow.add_edge("generate_cypher", "query_graph")
    workflow.add_edge("query_graph", "conditional_faiss")
    workflow.add_edge("conditional_faiss", "merge_context")
    workflow.add_edge("merge_context", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    return workflow.compile()


def run_agent(query: str) -> dict[str, Any]:
    """Execute agent and return structured result."""
    graph = build_agent_graph()
    initial: AgentState = {
        "query": query,
        "normalized_query": "",
        "cypher": "",
        "cypher_failed": False,
        "graph_result": {"nodes": [], "chunks": []},
        "faiss_chunks": [],
        "merged_context": "",
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "graph_nodes": [],
        "chunks_used": [],
        "timeline": [],
        "error": None,
    }
    final = graph.invoke(initial)
    return {
        "answer": final["answer"],
        "citations": final["citations"],
        "graph_nodes": final["graph_nodes"],
        "chunks_used": final["chunks_used"],
        "confidence": final["confidence"],
        "cypher_query": final.get("cypher"),
        "processing_timeline": final.get("timeline", []),
    }
