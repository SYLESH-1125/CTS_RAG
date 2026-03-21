"use client";

import { useCallback, useEffect, useMemo, useState, type ComponentProps } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Upload, LayoutDashboard, Loader2 } from "lucide-react";
import { uploadPdf, getJobStatusStreamUrl, getDocumentGraph, getPipelineEventsStreamUrl } from "@/lib/api";
import { useDashboardStore } from "@/stores/useDashboardStore";
import { ExtractionViewer } from "@/components/pipeline/ExtractionViewer";
import { ChunkingVisualizer } from "@/components/pipeline/ChunkingVisualizer";
import { GraphBuilderVisualizer } from "@/components/pipeline/GraphBuilderVisualizer";
import { QueryVisualizer } from "@/components/pipeline/QueryVisualizer";
import { PipelineStepper } from "@/components/pipeline/PipelineStepper";

type JobState = {
  status?: string;
  phase?: string;
  extraction?: unknown;
  chunks?: unknown[];
  graph?: { chunks_processed: number; entities_created: number; relations_created: number };
  progress?: {
    extraction?: { step?: string; current?: number; total?: number; detail?: string };
    chunking?: { current?: number; total?: number; stage?: string };
    graph?: { current?: number; total?: number; entities?: number; relations?: number; percent?: number };
  };
  live_graph?: {
    nodes: { id: string; name?: string; group?: number; entity_type?: string; community_id?: number; val?: number }[];
    links: { source: string; target: string }[];
  };
  logs?: { phase: string; message: string }[];
  error?: string;
};

type StreamExtraction = {
  text_units: Array<{ original: string; translated: string; page: number; source: string }>;
  table_units: Array<{ original?: string; context_ready?: string; page: number; source?: string }>;
  image_units: Array<{ merged_context?: string; original?: string; page: number; source?: string }>;
};

function upsertBySource<T extends Record<string, unknown>>(arr: T[], item: T, key: keyof T = "source" as keyof T): T[] {
  const src = String(item[key] ?? "");
  const i = arr.findIndex((x) => String(x[key] ?? "") === src);
  if (i >= 0) {
    const n = [...arr];
    n[i] = item;
    return n;
  }
  return [...arr, item];
}

export default function DashboardPage() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [jobState, setJobState] = useState<JobState | null>(null);
  const [rightTab, setRightTab] = useState<"details" | "json">("details");
  const [streamSessionId, setStreamSessionId] = useState<string | null>(null);
  const [streamedExtraction, setStreamedExtraction] = useState<StreamExtraction | null>(null);
  const [streamFeed, setStreamFeed] = useState<Array<{ type: string; summary: string; at: number }>>([]);
  const [lastEntityKey, setLastEntityKey] = useState<string | null>(null);

  const setJobStore = useDashboardStore((s) => s.setJobState);
  const setDocumentId = useDashboardStore((s) => s.setDocumentId);
  const centerStage = useDashboardStore((s) => s.centerStage);
  const setCenterStage = useDashboardStore((s) => s.setCenterStage);
  const documentId = useDashboardStore((s) => s.documentId);

  // Restore last ingested document id (queries stay scoped after refresh if backend data exists)
  useEffect(() => {
    if (typeof window === "undefined") return;
    const saved = window.localStorage.getItem("graph_rag_document_id");
    if (saved && !documentId) setDocumentId(saved);
  }, [documentId, setDocumentId]);

  useEffect(() => {
    if (jobState) setJobStore(jobState as Record<string, unknown>);
  }, [jobState, setJobStore]);

  useEffect(() => {
    if (!jobState?.phase) return;
    const p = jobState.phase;
    if (p === "extraction") setCenterStage("extraction");
    else if (p === "chunking") setCenterStage("chunking");
    else if (p === "graph_build") setCenterStage("graph");
    else if (p === "completed") setCenterStage("graph");
  }, [jobState?.phase, setCenterStage]);

  useEffect(() => {
    if (!streamSessionId) return;
    const es = new EventSource(getPipelineEventsStreamUrl(streamSessionId));
    const pushFeed = (type: string, summary: string) => {
      setStreamFeed((f) => [...f.slice(-60), { type, summary, at: Date.now() }]);
    };

    const on = (type: string, handler: (data: Record<string, unknown>) => void) => {
      es.addEventListener(type, (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data) as Record<string, unknown>;
          handler(data);
        } catch {
          /* ignore */
        }
      });
    };

    on("extraction_text", (d) => {
      setStreamedExtraction((prev) => {
        const base = prev ?? { text_units: [], table_units: [], image_units: [] };
        const u = {
          original: String(d.original_preview ?? ""),
          translated: String(d.text ?? ""),
          page: Number(d.page) || 0,
          source: String(d.source ?? ""),
        };
        return { ...base, text_units: upsertBySource(base.text_units, u) };
      });
      pushFeed("extraction_text", `Text p.${d.page} · ${String(d.text).slice(0, 40)}…`);
    });

    on("extraction_table", (d) => {
      setStreamedExtraction((prev) => {
        const base = prev ?? { text_units: [], table_units: [], image_units: [] };
        const grid = d.grid_preview as string[][] | undefined;
        const flat = grid?.map((row) => row.join(" | ")).join("\n") || String(d.original ?? "");
        const u = {
          original: String(d.original ?? flat),
          translated: String(d.translated ?? d.context_ready ?? flat),
          context_ready: String(d.context_ready ?? d.translated ?? flat),
          page: Number(d.page) || 0,
          source: String(d.source ?? ""),
        };
        return { ...base, table_units: upsertBySource(base.table_units, u) };
      });
      pushFeed("extraction_table", `Table ${d.source}`);
    });

    on("extraction_image", (d) => {
      setStreamedExtraction((prev) => {
        const base = prev ?? { text_units: [], table_units: [], image_units: [] };
        const orig = String(d.original ?? d.ocr_text ?? "");
        const translated = String(d.ocr_translated ?? d.merged_context ?? orig);
        const u = {
          merged_context: translated,
          original: orig,
          ocr_translated: translated,
          page: Number(d.page) || 0,
          source: String(d.source ?? ""),
        };
        return { ...base, image_units: upsertBySource(base.image_units, u) };
      });
      pushFeed("extraction_image", `Image ${d.source}`);
    });

    on("chunk_created", (d) => {
      const ch = d.chunk as Record<string, unknown> | undefined;
      if (ch?.chunk_id) {
        setJobState((prev) => {
          if (!prev) return prev;
          const chunks = [...((prev.chunks as Record<string, unknown>[]) || [])];
          if (!chunks.some((x) => x.chunk_id === ch.chunk_id)) chunks.push(ch as Record<string, unknown>);
          return { ...prev, chunks };
        });
      }
      pushFeed("chunk_created", `Chunk ${d.index}/${d.total}`);
    });

    on("entity_created", (d) => {
      const gk = typeof d.graph_key === "string" ? d.graph_key : null;
      if (gk) setLastEntityKey(gk);
      pushFeed("entity_created", String(d.name ?? ""));
    });

    on("relationship_created", (d) => {
      pushFeed("relationship_created", `${d.subject} → ${d.object} (${d.predicate})`);
    });

    on("graph_update", (d) => {
      const nodes = (Array.isArray(d.nodes) ? d.nodes : []) as NonNullable<JobState["live_graph"]>["nodes"];
      const links = (Array.isArray(d.links) ? d.links : []) as NonNullable<JobState["live_graph"]>["links"];
      setJobState((prev) => (prev ? { ...prev, live_graph: { nodes, links } } : prev));
      pushFeed("graph_update", `Graph ${nodes.length}n / ${links.length}e`);
    });

    on("progress_update", (d) => {
      pushFeed("progress", `${d.phase} · ${d.step ?? d.stage ?? ""}`);
    });

    es.addEventListener("done", () => {
      es.close();
    });

    es.onerror = () => es.close();
    return () => es.close();
  }, [streamSessionId]);

  const startUpload = useCallback(async () => {
    if (!file) return;
    setUploading(true);
    setJobState(null);
    setStreamedExtraction(null);
    setStreamFeed([]);
    try {
      const res = await uploadPdf(file);
      const jid = res.job_id as string;
      setStreamSessionId(jid);
      setDocumentId(jid);
      if (typeof window !== "undefined") {
        window.localStorage.setItem("graph_rag_document_id", jid);
      }
      const es = new EventSource(getJobStatusStreamUrl(jid));
      es.onmessage = (e) => {
        const data = JSON.parse(e.data) as JobState;
        setJobState(data);
        if (data.status === "completed" || data.status === "failed") {
          es.close();
          if (data.status === "completed") {
            getDocumentGraph(jid).then((g) => {
              setJobState((prev) => (prev ? { ...prev, live_graph: g } : prev));
            }).catch(() => {});
          }
        }
      };
      es.onerror = () => es.close();
    } catch (e) {
      console.error(e);
    } finally {
      setUploading(false);
    }
  }, [file, setDocumentId]);

  const extractionMerged = useMemo(() => {
    const fromJob = jobState?.extraction as ComponentProps<typeof ExtractionViewer>["extraction"] | null;
    const ju = fromJob as { text_units?: unknown[] } | null;
    if (ju?.text_units && ju.text_units.length > 0) return fromJob;
    if (
      streamedExtraction &&
      (streamedExtraction.text_units.length > 0 ||
        streamedExtraction.table_units.length > 0 ||
        streamedExtraction.image_units.length > 0)
    ) {
      return streamedExtraction as ComponentProps<typeof ExtractionViewer>["extraction"];
    }
    return fromJob ?? null;
  }, [jobState?.extraction, streamedExtraction]);

  const extraction = extractionMerged;
  const chunks = (jobState?.chunks ?? null) as ComponentProps<typeof ChunkingVisualizer>["chunks"];
  const liveGraph = jobState?.live_graph || null;
  const processing = jobState?.status === "processing";

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900">
      <header className="sticky top-0 z-10 border-b border-slate-200/80 bg-white/90 backdrop-blur-md">
        <div className="mx-auto flex h-14 max-w-[1600px] items-center justify-between px-4">
          <div className="flex items-center gap-2">
            <LayoutDashboard className="h-5 w-5 text-slate-700" />
            <span className="font-semibold tracking-tight">Graph RAG · Pipeline</span>
          </div>
          <nav className="flex gap-4 text-sm text-slate-600">
            <Link href="/" className="hover:text-slate-900">Home</Link>
            <Link href="/upload" className="hover:text-slate-900">Classic upload</Link>
          </nav>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1600px] gap-4 p-4 lg:grid-cols-12 lg:gap-6 lg:p-6">
        {/* LEFT */}
        <motion.aside
          initial={{ opacity: 0, x: -8 }}
          animate={{ opacity: 1, x: 0 }}
          className="lg:col-span-3 space-y-4"
        >
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">Ingest</h2>
            <label className="mt-3 flex cursor-pointer flex-col items-center rounded-lg border-2 border-dashed border-slate-200 bg-slate-50/50 px-4 py-8 text-center text-sm text-slate-600 hover:border-slate-300">
              <Upload className="mb-2 h-8 w-8 text-slate-400" />
              <span>{file?.name || "Choose PDF"}</span>
              <input type="file" accept=".pdf" className="hidden" onChange={(e) => setFile(e.target.files?.[0] || null)} />
            </label>
            <button
              type="button"
              disabled={!file || uploading || processing}
              onClick={startUpload}
              className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg bg-slate-900 py-2.5 text-sm font-medium text-white disabled:opacity-50"
            >
              {uploading || processing ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              Run pipeline
            </button>
            {documentId && (
              <p className="mt-2 break-all text-[10px] text-slate-400">Doc: {documentId.slice(0, 8)}…</p>
            )}
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">Timeline</h2>
            <ul className="mt-2 max-h-64 space-y-1 overflow-y-auto font-mono text-[10px] text-slate-600">
              {(jobState?.logs || []).map((l, i) => (
                <li key={i}>
                  <span className="text-blue-600">[{l.phase}]</span> {l.message}
                </li>
              ))}
            </ul>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">Live stream</h2>
            <p className="mt-1 text-[10px] text-slate-400">Granular SSE: extraction → chunks → entities → graph</p>
            <ul className="mt-2 max-h-48 space-y-0.5 overflow-y-auto font-mono text-[9px] leading-tight text-slate-600">
              {streamFeed
                .slice()
                .reverse()
                .map((e, i) => (
                  <motion.li
                    key={`${e.at}-${i}`}
                    initial={{ opacity: 0, x: -4 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="border-l-2 border-violet-300 pl-1"
                  >
                    <span className="text-violet-600">{e.type}</span> {e.summary}
                  </motion.li>
                ))}
            </ul>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">Center view</h2>
            <div className="mt-2 flex flex-wrap gap-1">
              {(["extraction", "chunking", "graph", "query"] as const).map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setCenterStage(t)}
                  className={`rounded-md px-2 py-1 text-xs ${centerStage === t ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-600"}`}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>
        </motion.aside>

        {/* CENTER */}
        <main className="lg:col-span-6 space-y-4">
          <PipelineStepper
            phase={jobState?.phase}
            centerStage={centerStage}
            progress={jobState?.progress}
            status={jobState?.status}
          />
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm min-h-[480px]">
            {centerStage === "extraction" && (
              <ExtractionViewer
                extraction={extraction}
                active={!!extraction || (jobState?.phase === "extraction" && processing)}
                isStreaming={jobState?.phase === "extraction" && processing && !jobState?.extraction}
              />
            )}
            {centerStage === "chunking" && (
              <ChunkingVisualizer
                chunks={chunks}
                progress={jobState?.progress?.chunking}
                active={jobState?.phase === "chunking" || !!chunks?.length}
              />
            )}
            {centerStage === "graph" && (
              <GraphBuilderVisualizer
                graphData={liveGraph}
                progress={jobState?.progress?.graph}
                active={jobState?.phase === "graph_build" || !!liveGraph?.nodes?.length}
                isBuilding={jobState?.status === "processing" && jobState?.phase === "graph_build"}
                highlightKeys={lastEntityKey ? [lastEntityKey] : undefined}
              />
            )}
            {centerStage === "query" && <QueryVisualizer documentId={documentId} />}
          </div>
        </main>

        {/* RIGHT */}
        <aside className="lg:col-span-3 space-y-4">
          <div className="rounded-xl border border-slate-200 bg-white shadow-sm">
            <div className="flex border-b border-slate-100">
              <button
                type="button"
                onClick={() => setRightTab("details")}
                className={`flex-1 py-2 text-xs font-medium ${rightTab === "details" ? "border-b-2 border-slate-900 text-slate-900" : "text-slate-500"}`}
              >
                Details
              </button>
              <button
                type="button"
                onClick={() => setRightTab("json")}
                className={`flex-1 py-2 text-xs font-medium ${rightTab === "json" ? "border-b-2 border-slate-900 text-slate-900" : "text-slate-500"}`}
              >
                JSON
              </button>
            </div>
            <div className="max-h-[520px] overflow-auto p-4 text-xs">
              {rightTab === "details" ? (
                <div className="space-y-3 text-slate-600">
                  <p><strong className="text-slate-800">Phase</strong> {jobState?.phase || "—"}</p>
                  <p><strong className="text-slate-800">Extraction</strong> {jobState?.progress?.extraction?.detail || jobState?.progress?.extraction?.step || "—"}</p>
                  <p><strong className="text-slate-800">Chunking</strong> {jobState?.progress?.chunking?.stage} {jobState?.progress?.chunking?.current}/{jobState?.progress?.chunking?.total}</p>
                  <p><strong className="text-slate-800">Graph</strong> {jobState?.graph?.chunks_processed} chunks · {jobState?.graph?.entities_created} ent · {jobState?.graph?.relations_created} rel</p>
                  {jobState?.error && <p className="text-red-600">{jobState.error}</p>}

                  <div className="rounded-lg border border-slate-100 bg-slate-50/80 p-2">
                    <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">Entities (live)</p>
                    <ul className="mt-1 max-h-36 space-y-1 overflow-y-auto text-[11px] text-slate-700">
                      {(liveGraph?.nodes || [])
                        .filter((n) => n.id && !String(n.id).startsWith("chunk:"))
                        .slice(0, 40)
                        .map((n) => (
                          <li key={n.id} className="truncate font-mono">
                            <span className="text-violet-600">{n.entity_type || "concept"}</span> · {n.name || n.id}
                          </li>
                        ))}
                      {!(liveGraph?.nodes || []).some((n) => n.id && !String(n.id).startsWith("chunk:")) && (
                        <li className="text-slate-400">—</li>
                      )}
                    </ul>
                  </div>

                  <div className="rounded-lg border border-slate-100 bg-slate-50/80 p-2">
                    <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">Current reasoning</p>
                    <p className="mt-1 text-[11px] leading-relaxed text-slate-600">
                      {streamFeed.length
                        ? streamFeed[streamFeed.length - 1]!.summary
                        : "Pipeline events and graph updates will appear in the live stream (left)."}
                    </p>
                  </div>
                </div>
              ) : (
                <pre className="whitespace-pre-wrap break-all font-mono text-[10px] text-slate-700">
                  {JSON.stringify(jobState, null, 2)}
                </pre>
              )}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
