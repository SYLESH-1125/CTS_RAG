"use client";

import { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import { Search, FileText, GitBranch, Brain, Activity, Loader2, ChevronRight } from "lucide-react";
import { queryRag } from "@/lib/api";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

interface Citation {
  chunk_id: string;
  page: number;
  source: string;
  text: string;
  graph_entities?: string[];
  graph_relationships?: string[];
}

interface GraphTraceNode {
  id: string;
  name: string;
  group?: number;
  labels?: string[];
  used?: boolean;
}

interface ReasoningStep {
  step: string;
  duration_ms: number;
  description: string;
}

interface QueryResult {
  answer: string;
  citations: Citation[];
  graph_trace?: { nodes: GraphTraceNode[]; edges: { source: string; target: string; type?: string }[]; used_node_ids?: string[] };
  reasoning_steps: ReasoningStep[];
  confidence: number;
  retrieval_graph?: { nodes: unknown[]; links: unknown[] };
}

const DOC_ID_KEY = "graph_rag_document_id";

function GraphTraceViz({ nodes, edges, usedIds }: { nodes: GraphTraceNode[]; edges: { source: string; target: string }[]; usedIds?: string[] }) {
  const usedSet = useMemo(() => new Set(usedIds || []), [usedIds]);
  const graphData = useMemo(() => {
    const nodeMap = new Map<string, { id: string; name: string; used: boolean }>();
    for (const n of nodes) {
      const id = String(n.id ?? "");
      if (!id) continue;
      nodeMap.set(id, {
        id,
        name: String(n.name ?? n.id ?? "").slice(0, 40),
        used: usedSet.has(id) ?? n.used ?? true,
      });
    }
    const linksOut: { source: string; target: string }[] = [];
    for (const e of edges) {
      const s = typeof e.source === "string" ? e.source : (e.source as { id?: string })?.id ?? "";
      const t = typeof e.target === "string" ? e.target : (e.target as { id?: string })?.id ?? "";
      if (s && t && (nodeMap.has(s) || nodeMap.has(t))) {
        linksOut.push({ source: s, target: t });
      }
    }
    return { nodes: Array.from(nodeMap.values()), links: linksOut };
  }, [nodes, edges, usedSet]);

  if (graphData.nodes.length === 0) {
    return (
      <div className="flex h-72 items-center justify-center rounded-xl bg-slate-50 text-slate-500 text-sm">
        No graph trace for this query
      </div>
    );
  }

  return (
    <ForceGraph2D
      graphData={graphData}
      width={500}
      height={280}
      nodeLabel={(n) => (n as { name?: string }).name ?? ""}
      nodeColor={(n) => {
        const node = n as { used?: boolean; id?: string };
        return (node.used ?? usedSet.has(String(node.id ?? ""))) ? "#0ea5e9" : "#cbd5e1";
      }}
      nodeRelSize={8}
      linkColor="#94a3b8"
      backgroundColor="#f8fafc"
    />
  );
}

export default function QueryPage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [documentId, setDocumentId] = useState<string | null>(null);
  const [searchAll, setSearchAll] = useState(false);
  const [activeCitation, setActiveCitation] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    setDocumentId(window.localStorage.getItem(DOC_ID_KEY));
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setActiveCitation(null);
    try {
      const data = await queryRag(query.trim(), documentId, searchAll);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/95 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4">
          <Link href="/" className="font-semibold text-slate-900">Graph RAG</Link>
          <nav className="flex gap-6">
            <Link href="/dashboard" className="text-sm text-slate-600 hover:text-slate-900">Pipeline</Link>
            <Link href="/upload" className="text-sm text-slate-600 hover:text-slate-900">Upload</Link>
            <Link href="/query" className="text-sm font-medium text-blue-600">Query</Link>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8">
        <h1 className="text-2xl font-bold text-slate-900">Query Interface</h1>
        <p className="mt-1 text-slate-600">
          Graph-first retrieval from Neo4j (cumulative across uploads).
          {documentId && !searchAll ? (
            <span className="ml-2 text-xs text-slate-500">Scoped to last PDF ({documentId.slice(0, 8)}…)</span>
          ) : searchAll ? (
            <span className="ml-2 text-xs text-emerald-600">Searching all documents</span>
          ) : (
            <span className="ml-2 text-xs text-amber-600">No doc — upload first or enable &quot;Search all&quot;</span>
          )}
        </p>

        <label className="mt-3 flex items-center gap-2 text-sm text-slate-600">
          <input type="checkbox" checked={searchAll} onChange={(e) => setSearchAll(e.target.checked)} />
          Search all documents (cumulative Neo4j + FAISS)
        </label>

        <form onSubmit={handleSubmit} className="mt-6">
          <div className="flex gap-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question..."
              className="flex-1 rounded-xl border border-slate-300 px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading}
              className="rounded-xl bg-blue-600 px-6 py-3 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Search className="h-5 w-5" />}
            </button>
          </div>
        </form>

        {error && (
          <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-4 text-red-700">{error}</div>
        )}

        {result && (
          <div className="mt-8 grid gap-6 lg:grid-cols-12">
            {/* 1. Answer panel */}
            <div className="lg:col-span-7 space-y-6">
              <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                <h2 className="flex items-center gap-2 font-semibold text-slate-900">
                  <Brain className="h-5 w-5 text-blue-600" /> Answer
                </h2>
                <div className="mt-4 prose prose-slate max-w-none text-slate-700 leading-relaxed whitespace-pre-wrap">
                  {result.answer}
                </div>
              </div>

              {/* 2. Confidence score */}
              <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                <h2 className="flex items-center gap-2 font-semibold text-slate-900">
                  <Activity className="h-5 w-5 text-emerald-600" /> Confidence
                </h2>
                <div className="mt-4 flex items-center gap-4">
                  <div className="h-4 flex-1 max-w-xs overflow-hidden rounded-full bg-slate-200">
                    <div
                      className="h-full rounded-full bg-emerald-500 transition-all duration-500"
                      style={{ width: `${Math.round((result.confidence ?? 0) * 100)}%` }}
                    />
                  </div>
                  <span className="text-lg font-semibold text-slate-800">
                    {Math.round((result.confidence ?? 0) * 100)}%
                  </span>
                </div>
              </div>
            </div>

            {/* 3. Citation cards (clickable) */}
            <div className="lg:col-span-5">
              <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm sticky top-24">
                <h2 className="flex items-center gap-2 font-semibold text-slate-900">
                  <FileText className="h-5 w-5 text-blue-600" /> Citations
                </h2>
                <p className="mt-1 text-xs text-slate-500">Sources with graph nodes and relationships used</p>
                <div className="mt-4 space-y-3 max-h-80 overflow-y-auto">
                  {result.citations?.length ? (
                    result.citations.map((c, i) => (
                      <button
                        key={c.chunk_id || i}
                        type="button"
                        onClick={() => setActiveCitation(activeCitation === c.chunk_id ? null : c.chunk_id)}
                        className={`w-full rounded-lg border p-3 text-left transition-all ${
                          activeCitation === c.chunk_id
                            ? "border-blue-500 bg-blue-50 ring-2 ring-blue-200"
                            : "border-slate-200 bg-slate-50/50 hover:border-slate-300 hover:bg-slate-100"
                        }`}
                      >
                        <div className="flex items-center gap-2 text-xs font-medium text-slate-500">
                          <span>{c.source}</span>
                          <span>·</span>
                          <span>Page {c.page}</span>
                          {c.chunk_id && (
                            <>
                              <span>·</span>
                              <span className="font-mono truncate">{c.chunk_id.slice(0, 12)}…</span>
                            </>
                          )}
                        </div>
                        <p className="mt-2 text-sm text-slate-700">{c.text}</p>
                        {c.graph_entities && c.graph_entities.length > 0 && (
                          <div className="mt-2 pt-2 border-t border-slate-200">
                            <span className="text-xs font-medium text-violet-600">Graph entities:</span>
                            <span className="ml-2 text-xs text-slate-600">{c.graph_entities.join(", ")}</span>
                          </div>
                        )}
                        {c.graph_relationships && c.graph_relationships.length > 0 && (
                          <div className="mt-1">
                            <span className="text-xs font-medium text-violet-600">Relationships:</span>
                            <ul className="mt-1 space-y-0.5 text-xs text-slate-600">
                              {c.graph_relationships.map((r, j) => (
                                <li key={j} className="font-mono">{r}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </button>
                    ))
                  ) : (
                    <p className="text-sm text-slate-500 italic">No citations</p>
                  )}
                </div>
              </div>
            </div>

            {/* 4. Graph visualization (full width) */}
            <div className="lg:col-span-12">
              <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                <h2 className="flex items-center gap-2 font-semibold text-slate-900">
                  <GitBranch className="h-5 w-5 text-violet-600" /> Graph Trace
                </h2>
                <p className="mt-1 text-xs text-slate-500">Nodes and edges used in answering</p>
                <div className="mt-4 rounded-lg border border-slate-100 bg-slate-50/30">
                  {(() => {
                    const gt = result.graph_trace;
                    const rg = result.retrieval_graph;
                    const nodes = (gt?.nodes as GraphTraceNode[]) ?? [];
                    const edges = gt?.edges ?? [];
                    const fallbackNodes = (rg?.nodes as { id?: string; name?: string }[]) ?? [];
                    const fallbackLinks = (rg?.links as { source: string | { id?: string }; target: string | { id?: string } }[]) ?? [];
                    const hasData = nodes.length > 0 || fallbackNodes.length > 0;
                    if (!hasData) {
                      return <div className="flex h-72 items-center justify-center text-slate-500 text-sm">No graph trace</div>;
                    }
                    return (
                      <GraphTraceViz
                        nodes={nodes.length > 0 ? nodes : fallbackNodes.map((n) => ({ id: n.id ?? "", name: n.name ?? n.id ?? "", used: true }))}
                        edges={edges.length > 0 ? edges : fallbackLinks.map((l) => ({
                          source: typeof l.source === "string" ? l.source : (l.source as { id?: string })?.id ?? "",
                          target: typeof l.target === "string" ? l.target : (l.target as { id?: string })?.id ?? "",
                        }))}
                        usedIds={gt?.used_node_ids}
                      />
                    );
                  })()}
                </div>
              </div>
            </div>

            {/* 5. Reasoning trace */}
            <div className="lg:col-span-12">
              <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                <h2 className="flex items-center gap-2 font-semibold text-slate-900">
                  <Activity className="h-5 w-5 text-amber-600" /> Reasoning Steps
                </h2>
                <p className="mt-1 text-xs text-slate-500">Explainable AI reasoning trace</p>
                <div className="mt-4 flex flex-wrap gap-2">
                  {result.reasoning_steps?.length ? (
                    result.reasoning_steps.map((s, i) => (
                      <div
                        key={i}
                        className="flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-4 py-2"
                      >
                        <span className="text-xs font-medium text-slate-500">{s.step}</span>
                        <ChevronRight className="h-4 w-4 text-slate-400" />
                        <span className="text-sm text-slate-700">{s.description}</span>
                        <span className="text-xs text-slate-400">({s.duration_ms}ms)</span>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-slate-500 italic">No reasoning steps</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
