"use client";

import { useState, useCallback, useEffect } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { Upload, FileText, Table, Image, Loader2, Network } from "lucide-react";
import { uploadPdf, getJobStatusStreamUrl, pushToNeo4j, getDocumentGraph, getNeo4jStats } from "@/lib/api";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

type JobPhase = "idle" | "extraction" | "chunking" | "graph_build" | "vector_store" | "completed" | "failed";

interface ProgressState {
  extraction?: { step?: string; current?: number; total?: number; detail?: string };
  chunking?: { current?: number; total?: number; stage?: string };
  graph?: { current?: number; total?: number; entities?: number; relations?: number };
}

interface JobState {
  job_id: string;
  status: string;
  phase: JobPhase;
  phases?: string[];
  extraction: {
    text_units: Array<{ type: string; original: string; translated: string; page: number; source: string }>;
    table_units: Array<{ type: string; original: string; context_ready: string; page: number }>;
    image_units: Array<{
      type: string;
      original: string;
      ocr_translated?: string;
      vision_context?: string;
      merged_context?: string;
      page: number;
      source: string;
    }>;
    logs: Array<{ step: string; message: string }>;
  } | null;
  chunks: Array<{ chunk_id: string; text: string; source: string; page: number }> | null;
  graph: { chunks_processed: number; entities_created: number; relations_created: number } | null;
  progress?: ProgressState;
  logs: Array<{ phase: string; message: string }>;
  error: string | null;
}

function ProgressBar({ label, current, total, detail }: { label: string; current: number; total: number; detail?: string }) {
  const pct = total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 0;
  const indeterminate = total <= 0 && current <= 0;
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs font-medium text-surface-700">
        <span>{label}</span>
        <span>{total > 0 ? `${current} / ${total}` : detail || "…"}</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-surface-200">
        <div
          className={`h-full rounded-full bg-accent transition-all duration-300 ${indeterminate ? "animate-pulse w-1/3" : ""}`}
          style={indeterminate ? {} : { width: `${pct}%` }}
        />
      </div>
      {detail && total > 0 && <p className="text-xs text-surface-500">{detail}</p>}
    </div>
  );
}

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobState, setJobState] = useState<JobState | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [neo4jPushed, setNeo4jPushed] = useState<boolean | null>(null);
  const [neo4jPushError, setNeo4jPushError] = useState<string | null>(null);
  const [neo4jStats, setNeo4jStats] = useState<{ connected: boolean; chunks: number; entities: number; relationships: number } | null>(null);
  const [graphFromNeo4j, setGraphFromNeo4j] = useState<{ nodes: { id: string; name?: string; group?: number }[]; links: { source: string; target: string; type?: string }[] } | null>(null);

  const refreshNeo4jStats = useCallback(async () => {
    try {
      const s = await getNeo4jStats();
      setNeo4jStats(s.connected ? { connected: true, chunks: s.chunks, entities: s.entities, relationships: s.relationships } : null);
    } catch {
      setNeo4jStats(null);
    }
  }, []);

  const handleForcePushNeo4j = useCallback(async () => {
    if (!jobId || jobState?.status !== "completed") return;
    setNeo4jPushed(null);
    setNeo4jPushError(null);
    try {
      await pushToNeo4j(jobId);
      setNeo4jPushed(true);
      setGraphFromNeo4j(await getDocumentGraph(jobId));
      refreshNeo4jStats();
    } catch (err: unknown) {
      setNeo4jPushed(false);
      setNeo4jPushError(err instanceof Error ? err.message : String(err));
    }
  }, [jobId, jobState?.status, refreshNeo4jStats]);

  const handleUpload = useCallback(async () => {
    if (!file) return;
    setUploading(true);
    setNeo4jPushed(null);
    setNeo4jPushError(null);
    setGraphFromNeo4j(null);
    try {
      const res = await uploadPdf(file);
      setJobId(res.job_id);
      const url = getJobStatusStreamUrl(res.job_id);
      const es = new EventSource(url);
      es.onmessage = (e) => {
        const data = JSON.parse(e.data);
        setJobState(data);
        if (data.status === "completed") {
          if (res.job_id && typeof window !== "undefined") {
            window.localStorage.setItem("graph_rag_document_id", res.job_id);
          }
          // Classic upload: explicitly push graph to Neo4j, then fetch to display
          pushToNeo4j(res.job_id)
            .then(() => {
              setNeo4jPushed(true);
              setNeo4jPushError(null);
              return getDocumentGraph(res.job_id);
            })
            .then((g) => setGraphFromNeo4j(g))
            .then(() => refreshNeo4jStats())
            .catch((err: unknown) => {
              setNeo4jPushed(false);
              setNeo4jPushError(err instanceof Error ? err.message : String(err));
            });
          es.close();
        }
        if (data.status === "failed") es.close();
      };
      es.onerror = () => es.close();
    } catch (err) {
      console.error(err);
      setJobState({ job_id: "", status: "failed", phase: "idle", extraction: null, chunks: null, graph: null, logs: [], error: String(err) } as JobState);
    } finally {
      setUploading(false);
    }
  }, [file, refreshNeo4jStats]);

  const prog = jobState?.progress;
  const isProcessing = jobState?.status === "processing";

  useEffect(() => {
    refreshNeo4jStats();
  }, [refreshNeo4jStats]);

  return (
    <div className="min-h-screen bg-surface-50">
      <header className="border-b border-surface-200 bg-white">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4">
          <Link href="/" className="font-semibold text-surface-900">Graph RAG</Link>
          <nav className="flex gap-6">
            <Link href="/upload" className="text-sm font-medium text-accent">Upload</Link>
            <Link href="/query" className="text-sm text-surface-600 hover:text-surface-900">Query</Link>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8">
        <h1 className="text-2xl font-bold text-surface-900">Upload Dashboard</h1>
        <p className="mt-1 text-surface-600">Live extraction, chunking, and graph build with progress bars.</p>

        {/* Neo4j stats - verify data is in the database */}
        <div className="mt-4 flex items-center gap-4">
          <button
            type="button"
            onClick={refreshNeo4jStats}
            className="rounded-md border border-surface-200 bg-white px-3 py-1.5 text-sm text-surface-600 hover:bg-surface-50"
          >
            Refresh Neo4j counts
          </button>
          {neo4jStats && (
            <span className="text-sm text-surface-600">
              Neo4j: <strong>{neo4jStats.chunks}</strong> Chunks · <strong>{neo4jStats.entities}</strong> Entities · <strong>{neo4jStats.relationships}</strong> Relationships
            </span>
          )}
          {neo4jStats === null && (
            <span className="text-sm text-surface-500">Neo4j not configured or unreachable</span>
          )}
        </div>

        <div
          className={`mt-6 rounded-xl border-2 border-dashed p-12 text-center transition-colors ${
            dragOver ? "border-accent bg-accent/5" : "border-surface-300 bg-white"
          }`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f?.name.endsWith(".pdf")) setFile(f); }}
        >
          <input type="file" accept=".pdf" className="hidden" id="pdf-input" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <label htmlFor="pdf-input" className="cursor-pointer">
            <Upload className="mx-auto h-12 w-12 text-surface-400" />
            <p className="mt-2 font-medium text-surface-700">{file ? file.name : "Drop PDF or click to browse"}</p>
          </label>
          <button
            onClick={handleUpload}
            disabled={!file || uploading || isProcessing}
            className="mt-4 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50 hover:bg-accent-hover"
          >
            {uploading || isProcessing ? <><Loader2 className="mr-2 inline h-4 w-4 animate-spin" /> Processing...</> : "Start Extraction"}
          </button>
        </div>

        {jobState && (
          <div className="mt-8 space-y-6">
            {/* Pipeline + live progress */}
            <div className="rounded-lg border border-surface-200 bg-white p-4">
              <h2 className="font-semibold text-surface-900">Pipeline &amp; live progress</h2>
              <div className="mt-3 flex flex-wrap gap-2">
                {(jobState.phases ?? ["extraction", "chunking", "graph_build", "vector_store", "completed"]).map((p) => (
                  <span
                    key={p}
                    className={`rounded-full px-3 py-1 text-xs font-medium ${
                      jobState.phase === p ? "bg-accent text-white" : "bg-surface-100 text-surface-600"
                    }`}
                  >
                    {p.replace(/_/g, " ")}
                  </span>
                ))}
              </div>

              {isProcessing && prog && (
                <div className="mt-4 space-y-4 border-t border-surface-100 pt-4">
                  {jobState.phase === "extraction" && (
                    <ProgressBar
                      label="PDF extraction"
                      current={prog.extraction?.current ?? 0}
                      total={prog.extraction?.total ?? 0}
                      detail={prog.extraction?.detail || prog.extraction?.step}
                    />
                  )}
                  {jobState.phase === "chunking" && (
                    <ProgressBar
                      label="Chunking"
                      current={prog.chunking?.current ?? 0}
                      total={prog.chunking?.total || Math.max(jobState.chunks?.length || 0, 1)}
                      detail={prog.chunking?.stage}
                    />
                  )}
                  {jobState.phase === "graph_build" && (
                    <div className="space-y-2">
                      <ProgressBar
                        label="Graph build (LLM batches)"
                        current={prog.graph?.current ?? 0}
                        total={prog.graph?.total || Math.max(jobState.chunks?.length || 0, 1)}
                        detail={`Entities so far: ${prog.graph?.entities ?? 0} · Relations: ${prog.graph?.relations ?? 0}`}
                      />
                      <div className="flex items-center gap-2 rounded-lg border border-dashed border-accent/40 bg-accent/5 p-3 text-sm text-surface-700">
                        <Network className="h-5 w-5 shrink-0 text-accent" />
                        <span>Extracting entities &amp; relations from chunks into Neo4j (cumulative — each upload adds to the graph).</span>
                      </div>
                    </div>
                  )}
                  {jobState.phase === "vector_store" && (
                    <div className="flex items-center gap-2 text-sm text-surface-600">
                      <Loader2 className="h-4 w-4 animate-spin text-accent" />
                      Building FAISS embeddings…
                    </div>
                  )}
                </div>
              )}

              {jobState.status === "failed" && (
                <p className="mt-2 text-sm text-red-600">{jobState.error}</p>
              )}
            </div>

            <div className="rounded-lg border border-surface-200 bg-white p-4">
              <h2 className="font-semibold text-surface-900">Logs</h2>
              <div className="mt-2 max-h-40 overflow-y-auto font-mono text-xs text-surface-600">
                {jobState.logs?.length ? (
                  jobState.logs.map((l, i) => (
                    <div key={i} className="py-1">
                      <span className="text-accent">[{l.phase}]</span> {l.message}
                    </div>
                  ))
                ) : (
                  <div className="py-2 text-surface-500 italic">{isProcessing ? "Waiting for logs…" : "No logs yet"}</div>
                )}
              </div>
            </div>

            {jobState.extraction && (
              <div className="grid gap-6 md:grid-cols-3">
                <div className="rounded-lg border border-surface-200 bg-white p-4">
                  <h3 className="flex items-center gap-2 font-semibold text-surface-900">
                    <FileText className="h-4 w-4" /> Text ({jobState.extraction.text_units?.length || 0})
                  </h3>
                  <div className="mt-2 max-h-64 overflow-y-auto space-y-2 text-sm text-surface-600">
                    {jobState.extraction.text_units?.map((u, i) => (
                      <div key={i} className="rounded bg-surface-50 p-2">
                        <span className="text-xs text-surface-500">Page {u.page}</span>
                        <p className="mt-1 line-clamp-3">{u.translated || u.original}</p>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="rounded-lg border border-surface-200 bg-white p-4">
                  <h3 className="flex items-center gap-2 font-semibold text-surface-900">
                    <Table className="h-4 w-4" /> Tables ({jobState.extraction.table_units?.length || 0})
                  </h3>
                  <div className="mt-2 max-h-64 overflow-y-auto space-y-2 text-sm text-surface-600">
                    {jobState.extraction.table_units?.map((u, i) => (
                      <div key={i} className="rounded bg-surface-50 p-2">
                        <span className="text-xs text-surface-500">Page {u.page}</span>
                        <p className="mt-1 line-clamp-3">{u.context_ready || u.original}</p>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="rounded-lg border border-surface-200 bg-white p-4">
                  <h3 className="flex items-center gap-2 font-semibold text-surface-900">
                    <Image className="h-4 w-4" /> Images / Charts ({jobState.extraction.image_units?.length || 0})
                  </h3>
                  <div className="mt-3 max-h-96 overflow-y-auto space-y-4">
                    {jobState.extraction.image_units?.map((u, i) => (
                      <div key={i} className="rounded-lg border border-surface-200 bg-surface-50 p-3">
                        <span className="text-xs font-medium text-surface-500">{u.source} • Page {u.page}</span>
                        <p className="mt-2 text-sm text-surface-700 whitespace-pre-wrap">{u.merged_context || u.original || "(no text)"}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {jobState.chunks && jobState.chunks.length > 0 && (
              <div className="rounded-lg border border-surface-200 bg-white p-4">
                <h2 className="font-semibold text-surface-900">Chunks {isProcessing && jobState.phase === "chunking" ? "(updating live)" : ""}</h2>
                <p className="mt-1 text-sm text-surface-600">
                  {jobState.chunks.length} chunk{jobState.chunks.length !== 1 ? "s" : ""} total
                </p>
                <div className="mt-2 max-h-64 overflow-y-auto space-y-1 text-xs">
                  {jobState.chunks.map((c, i) => (
                    <div key={c.chunk_id || i} className="rounded bg-surface-50 p-2">
                      <span className="text-surface-500 font-medium">{c.source}</span>
                      <span className="text-surface-400 ml-1">({i + 1}/{jobState.chunks!.length})</span>
                      <p className="mt-1 text-surface-700 line-clamp-2">{c.text?.slice(0, 120)}{(c.text?.length || 0) > 120 ? "…" : ""}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {jobState.graph && (
              <div className="rounded-lg border border-surface-200 bg-white p-4">
                <h2 className="flex items-center gap-2 font-semibold text-surface-900">
                  <Network className="h-5 w-5 text-accent" /> Graph built (from Neo4j)
                </h2>
                <p className="mt-1 text-sm text-surface-600">
                  Stored in Neo4j. Nodes and edges below are fetched directly from the database.
                </p>
                <div className="mt-2 flex flex-wrap gap-4 text-sm">
                  <span className="rounded-md bg-surface-100 px-2 py-1">Chunks: {jobState.graph.chunks_processed}</span>
                  <span className="rounded-md bg-surface-100 px-2 py-1">Entities: {jobState.graph.entities_created}</span>
                  <span className="rounded-md bg-surface-100 px-2 py-1">Relations: {jobState.graph.relations_created}</span>
                  {neo4jPushed === true && (
                    <span className="rounded-md bg-emerald-100 px-2 py-1 text-emerald-800">Neo4j ✓</span>
                  )}
                  {neo4jPushed === false && (
                    <>
                      <span className="rounded-md bg-red-100 px-2 py-1 text-red-800">Neo4j push failed</span>
                      {neo4jPushError && <span className="text-xs text-red-600">{neo4jPushError}</span>}
                      <button
                        type="button"
                        onClick={handleForcePushNeo4j}
                        className="rounded-md bg-accent px-2 py-1 text-xs font-medium text-white hover:bg-accent-hover"
                      >
                        Force push again
                      </button>
                    </>
                  )}
                  {jobState.status === "completed" && neo4jPushed === null && (
                    <span className="text-surface-500">Pushing to Neo4j…</span>
                  )}
                </div>
                {graphFromNeo4j && graphFromNeo4j.nodes?.length > 0 && (
                  <div className="mt-4 h-[280px] w-full overflow-hidden rounded-lg border border-surface-200 bg-slate-950">
                    <ForceGraph2D
                      graphData={(() => {
                        const nodeMap = new Map<string, { id: string; name: string; group?: number }>();
                        for (const n of graphFromNeo4j.nodes || []) {
                          const id = String(n?.id ?? "");
                          if (id) nodeMap.set(id, { id, name: String(n?.name ?? n?.id ?? id).slice(0, 40), group: n?.group });
                        }
                        const linksOut: { source: string; target: string }[] = [];
                        for (const l of graphFromNeo4j.links || []) {
                          const s = typeof l.source === "string" ? l.source : (l.source as { id?: string })?.id ?? "";
                          const t = typeof l.target === "string" ? l.target : (l.target as { id?: string })?.id ?? "";
                          if (s && t && (nodeMap.has(s) || nodeMap.has(t))) {
                            if (!nodeMap.has(s)) nodeMap.set(s, { id: s, name: s.split("::").pop() ?? s });
                            if (!nodeMap.has(t)) nodeMap.set(t, { id: t, name: t.split("::").pop() ?? t });
                            linksOut.push({ source: s, target: t });
                          }
                        }
                        return { nodes: Array.from(nodeMap.values()), links: linksOut };
                      })()}
                      nodeLabel="name"
                      nodeColor={(n) => ((n as { group?: number }).group === 1 ? "#94a3b8" : "#60a5fa")}
                      linkColor="#64748b"
                      backgroundColor="#0f172a"
                    />
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
