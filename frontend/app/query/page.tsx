"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { Search, FileText, GitBranch, Database, Loader2 } from "lucide-react";
import { queryRag } from "@/lib/api";
import { GraphVisualization } from "@/components/GraphVisualization";

interface QueryResult {
  answer: string;
  citations: Array<{ source: string; page: number; chunk_id: string }>;
  graph_nodes: Array<{ id: string; labels: string[]; props: Record<string, unknown> }>;
  chunks_used: Array<{ chunk_id: string; text: string; source: string; page: number; score?: number }>;
  confidence: number;
  cypher_query: string | null;
  processing_timeline: Array<{ step: string; duration_ms: number }>;
}

const DOC_ID_KEY = "graph_rag_document_id";

export default function QueryPage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [documentId, setDocumentId] = useState<string | null>(null);
  const [searchAll, setSearchAll] = useState(false);

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
    <div className="min-h-screen bg-surface-50">
      <header className="border-b border-surface-200 bg-white">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4">
          <Link href="/" className="font-semibold text-surface-900">Graph RAG</Link>
          <nav className="flex gap-6">
            <Link href="/dashboard" className="text-sm text-surface-600 hover:text-surface-900">Pipeline</Link>
            <Link href="/upload" className="text-sm text-surface-600 hover:text-surface-900">Upload</Link>
            <Link href="/query" className="text-sm font-medium text-accent">Query</Link>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8">
        <h1 className="text-2xl font-bold text-surface-900">Query Interface</h1>
        <p className="mt-1 text-surface-600">
          Graph-first retrieval from Neo4j (cumulative across uploads).
          {documentId && !searchAll ? (
            <span className="ml-1 block text-xs text-surface-500 sm:inline sm:ml-2">
              Scoped to last ingested PDF ({documentId.slice(0, 8)}…). Use <Link href="/dashboard" className="underline">Pipeline dashboard</Link> to ingest another.
            </span>
          ) : searchAll ? (
            <span className="ml-1 block text-xs text-emerald-700 sm:inline sm:ml-2">
              Searching all documents in Neo4j/FAISS (cumulative).
            </span>
          ) : (
            <span className="ml-1 block text-xs text-amber-700 sm:inline sm:ml-2">
              No document id — queries use last upload. Upload first or enable &quot;Search all&quot;.
            </span>
          )}
        </p>

        <label className="mt-2 flex items-center gap-2 text-sm text-surface-600">
          <input
            type="checkbox"
            checked={searchAll}
            onChange={(e) => setSearchAll(e.target.checked)}
          />
          Search all documents (cumulative Neo4j + FAISS)
        </label>

        {/* Query input */}
        <form onSubmit={handleSubmit} className="mt-6">
          <div className="flex gap-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question..."
              className="flex-1 rounded-lg border border-surface-300 px-4 py-3 text-surface-900 placeholder:text-surface-400 focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading}
              className="rounded-lg bg-accent px-6 py-3 font-medium text-white hover:bg-accent-hover disabled:opacity-50"
            >
              {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Search className="h-5 w-5" />}
            </button>
          </div>
        </form>

        {error && (
          <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-4 text-red-700">{error}</div>
        )}

        {result && (
          <div className="mt-8 space-y-6">
            {/* Processing timeline */}
            <div className="rounded-lg border border-surface-200 bg-white p-4">
              <h2 className="flex items-center gap-2 font-semibold text-surface-900">
                <Database className="h-4 w-4" /> Processing Timeline
              </h2>
              <div className="mt-2 flex flex-wrap gap-2">
                {result.processing_timeline?.map((t, i) => (
                  <span key={i} className="rounded bg-surface-100 px-2 py-1 text-xs">
                    {t.step}: {Math.round(t.duration_ms)}ms
                  </span>
                ))}
              </div>
            </div>

            {/* Cypher query */}
            {result.cypher_query && (
              <div className="rounded-lg border border-surface-200 bg-white p-4">
                <h2 className="flex items-center gap-2 font-semibold text-surface-900">
                  <GitBranch className="h-4 w-4" /> Generated Cypher
                </h2>
                <pre className="mt-2 overflow-x-auto rounded bg-surface-100 p-3 font-mono text-sm text-surface-700">
                  {result.cypher_query}
                </pre>
              </div>
            )}

            {/* Graph + Chunks side by side */}
            <div className="grid gap-6 lg:grid-cols-2">
              <div className="rounded-lg border border-surface-200 bg-white p-4">
                <h2 className="font-semibold text-surface-900">Graph Traversal</h2>
                <div className="mt-2 h-64">
                  <GraphVisualization nodes={result.graph_nodes} />
                </div>
              </div>
              <div className="rounded-lg border border-surface-200 bg-white p-4">
                <h2 className="flex items-center gap-2 font-semibold text-surface-900">
                  <FileText className="h-4 w-4" /> Retrieved Chunks (Graph + FAISS)
                </h2>
                <div className="mt-2 max-h-64 overflow-y-auto space-y-2">
                  {result.chunks_used?.map((c, i) => (
                    <div key={i} className="rounded bg-surface-50 p-2 text-sm">
                      <span className="text-xs text-surface-500">{c.source} · Page {c.page}</span>
                      {c.score != null && <span className="ml-2 text-xs text-accent">score: {c.score.toFixed(2)}</span>}
                      <p className="mt-1 line-clamp-2 text-surface-700">{c.text}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Output panel */}
            <div className="rounded-lg border border-surface-200 bg-white p-6">
              <h2 className="font-semibold text-surface-900">Answer</h2>
              <div className="mt-2 flex items-center gap-2">
                <span className="rounded-full bg-surface-100 px-2 py-0.5 text-xs font-medium">
                  Confidence: {(result.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <div className="mt-4 prose prose-sm max-w-none text-surface-700">
                {result.answer}
              </div>
              {result.citations?.length > 0 && (
                <div className="mt-6">
                  <h3 className="text-sm font-medium text-surface-600">Citations</h3>
                  <ul className="mt-2 space-y-1 text-sm">
                    {result.citations.map((c, i) => (
                      <li key={i} className="text-surface-600">
                        {c.source} — Page {c.page}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
