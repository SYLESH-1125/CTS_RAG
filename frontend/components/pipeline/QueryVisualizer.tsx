"use client";

import { useState, useMemo, useCallback } from "react";
import dynamic from "next/dynamic";
import { motion, AnimatePresence } from "framer-motion";
import { queryRag } from "@/lib/api";
import { Loader2, Send, GitBranch, Sparkles, FileText } from "lucide-react";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

type Citation = {
  chunk_id?: string;
  source?: string;
  page?: number;
  text?: string;
  graph_entities?: string[];
  graph_relationships?: string[];
};
type QueryResult = {
  answer: string;
  citations: Citation[];
  cypher_query?: string;
  confidence?: number;
  chunks_used?: { source?: string; text?: string; chunk_id?: string; score?: number }[];
  processing_timeline?: { step: string; duration_ms: number }[];
  retrieval_graph?: {
    nodes: { id: string; name?: string; group?: number }[];
    links: { source: string; target: string; type?: string }[];
  };
  graph_edges?: { source: string; target: string; type?: string }[];
};

type Node = { id: string; name?: string; group?: number };
function sanitizeQueryGraph(g: QueryResult["retrieval_graph"] | undefined) {
  if (!g?.nodes?.length) return { nodes: [] as Node[], links: [] as { source: string; target: string; type?: string }[] };
  const nodeById = new Map<string, Node>();
  for (const n of g.nodes) {
    if (!n?.id) continue;
    const id = String(n.id);
    nodeById.set(id, { id, name: n.name || id.split("::").pop() || id, group: n.group ?? 2 });
  }
  const linksOut: { source: string; target: string; type?: string }[] = [];
  for (const l of g.links || []) {
    const sRaw = typeof l.source === "object" && l.source != null ? (l.source as { id?: string }).id : l.source;
    const tRaw = typeof l.target === "object" && l.target != null ? (l.target as { id?: string }).id : l.target;
    if (sRaw == null || tRaw == null) continue;
    const s = String(sRaw);
    const t = String(tRaw);
    const short = (x: string) => (x.includes("::") ? x.split("::").pop() || x : x).slice(0, 48);
    if (!nodeById.has(s)) nodeById.set(s, { id: s, name: short(s), group: 1 });
    if (!nodeById.has(t)) nodeById.set(t, { id: t, name: short(t), group: 2 });
    linksOut.push({ source: s, target: t, type: l.type });
  }
  return { nodes: Array.from(nodeById.values()), links: linksOut };
}

export function QueryVisualizer({ documentId }: { documentId: string | null }) {
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResult | null>(null);
  const graphPayload = useMemo(() => sanitizeQueryGraph(result?.retrieval_graph), [result?.retrieval_graph]);

  const paint = useCallback(
    (node: object, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const n = node as Node & { x?: number; y?: number };
      const label = n.name || n.id;
      const r = (n.group === 1 ? 5 : 7) / globalScale;
      ctx.beginPath();
      ctx.arc(n.x!, n.y!, r, 0, 2 * Math.PI);
      ctx.fillStyle = n.group === 1 ? "#94a3b8" : "#60a5fa";
      ctx.fill();
      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 0.6 / globalScale;
      ctx.stroke();
      ctx.font = `${9 / globalScale}px sans-serif`;
      ctx.fillStyle = "#0f172a";
      ctx.fillText(String(label).slice(0, 14), n.x! + r + 2 / globalScale, n.y! + 3 / globalScale);
    },
    []
  );

  async function run() {
    if (!q.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const data = (await queryRag(q.trim(), documentId)) as QueryResult;
      setResult(data);
    } catch (e) {
      setResult({ answer: String(e), citations: [] });
    } finally {
      setLoading(false);
    }
  }

  const timeline = result?.processing_timeline || [];
  const showGraph = graphPayload.nodes.length > 0;

  const formattedAnswer = useMemo(() => {
    const raw = result?.answer ?? "";
    if (!raw) return { summary: "", details: "" };
    const detailsMatch = raw.match(/\bDetails?:\s*([\s\S]*)/i);
    if (detailsMatch) {
      const summary = raw.slice(0, raw.search(/\bDetails?:\s*/i)).trim();
      const details = detailsMatch[1].trim();
      return { summary, details };
    }
    return { summary: raw, details: "" };
  }, [result?.answer]);

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-2">
        <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-blue-600" />
        <div>
          <h3 className="text-sm font-semibold text-slate-800">Phase 4 · Query reasoning</h3>
          <p className="text-xs text-slate-500">
            Scoped to your document. Watch retrieval steps, subgraph, citations, then the answer.
          </p>
          <div className="mt-2 inline-flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50/90 px-2.5 py-1 text-[11px] font-medium text-emerald-900">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500" aria-hidden />
            Querying current document only (FAISS + Neo4j filtered by doc / session id)
          </div>
        </div>
      </div>

      <div className="flex gap-2">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Ask about this document…"
          className="flex-1 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm shadow-sm focus:border-slate-400 focus:outline-none focus:ring-1 focus:ring-slate-400"
          onKeyDown={(e) => e.key === "Enter" && run()}
        />
        <button
          type="button"
          onClick={run}
          disabled={loading || !documentId}
          className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          Run
        </button>
      </div>
      {!documentId && <p className="text-xs text-amber-700">Complete an upload first to attach a document.</p>}

      <AnimatePresence mode="wait">
        {loading && (
          <motion.div
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="space-y-2 rounded-xl border border-slate-200 bg-slate-50 p-4"
          >
            <p className="text-xs font-medium text-slate-600">Retrieving context…</p>
            <motion.div
              className="h-1 overflow-hidden rounded-full bg-slate-200"
              initial={{ opacity: 0.8 }}
              animate={{ opacity: [0.6, 1, 0.6] }}
              transition={{ repeat: Infinity, duration: 1.2 }}
            >
              <motion.div
                className="h-full bg-blue-500"
                initial={{ width: "0%" }}
                animate={{ width: "100%" }}
                transition={{ duration: 2.2, ease: "easeInOut" }}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {result && !loading && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4"
        >
          {/* Step-by-step pipeline */}
          {timeline.length > 0 && (
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Retrieval trace</p>
              <ol className="mt-3 space-y-2">
                {timeline.map((t, i) => (
                  <motion.li
                    key={`${t.step}-${i}`}
                    initial={{ opacity: 0, x: -6 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.08 }}
                    className="flex items-center justify-between rounded-lg border border-blue-200/60 bg-blue-50/40 px-2 py-1.5 text-xs text-slate-800"
                  >
                    <span className="font-mono text-[11px]">{t.step}</span>
                    <span className="text-slate-500">{Math.round(t.duration_ms)} ms</span>
                  </motion.li>
                ))}
              </ol>
            </div>
          )}

          {result.cypher_query && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.15 }}
              className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm"
            >
              <span className="flex items-center gap-1 text-xs font-medium text-slate-500">
                <GitBranch className="h-3.5 w-3.5" /> Graph pattern used
              </span>
              <pre className="mt-2 max-h-28 overflow-auto rounded-lg bg-slate-900 p-3 text-[11px] leading-relaxed text-slate-100">
                {result.cypher_query}
              </pre>
            </motion.div>
          )}

          {/* Subgraph from this query */}
          {showGraph && (
            <motion.div
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2, type: "spring", stiffness: 120 }}
              className="overflow-hidden rounded-xl border border-slate-200 bg-slate-950 shadow-inner"
            >
              <div className="border-b border-slate-800 px-3 py-2 text-[11px] text-slate-400">
                Entities & relationships used for this answer ({graphPayload.nodes.length} nodes · {graphPayload.links.length}{" "}
                edges)
              </div>
              <div className="h-[220px] w-full">
                <ForceGraph2D
                  graphData={graphPayload}
                  nodeCanvasObject={paint}
                  nodePointerAreaPaint={(node, color, ctx) => {
                    const n = node as Node & { x?: number; y?: number };
                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.arc(n.x!, n.y!, 9, 0, 2 * Math.PI);
                    ctx.fill();
                  }}
                  linkColor={() => "#64748b"}
                  linkWidth={0.5}
                  backgroundColor="#0f172a"
                  cooldownTicks={60}
                />
              </div>
            </motion.div>
          )}

          {/* Chunks with animation */}
          {(result.chunks_used?.length ?? 0) > 0 && (
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <span className="text-xs font-medium text-slate-500">Evidence chunks (FAISS + graph)</span>
              <ul className="mt-2 max-h-40 space-y-2 overflow-y-auto">
                {result.chunks_used!.map((c, i) => (
                  <motion.li
                    key={c.chunk_id || i}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.25 + i * 0.06 }}
                    className="rounded-lg border border-slate-100 bg-slate-50/80 p-2 text-xs text-slate-700"
                  >
                    <span className="font-mono text-[10px] text-blue-600">{c.chunk_id?.slice(0, 8)}…</span>{" "}
                    <span className="text-slate-500">{c.source}</span>
                    {c.score != null && (
                      <span className="ml-2 rounded bg-white px-1 text-[10px] text-slate-500">sim {c.score.toFixed(2)}</span>
                    )}
                    <p className="mt-1 line-clamp-3 text-[11px] leading-relaxed">{c.text}</p>
                  </motion.li>
                ))}
              </ul>
            </div>
          )}

          {/* Detailed Answer card - formatted, structured output */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.32 }}
            className="rounded-xl border-2 border-blue-200/80 bg-white p-6 shadow-lg"
          >
            <div className="flex items-center justify-between border-b border-slate-200 pb-3">
              <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                <FileText className="h-4 w-4 text-blue-600" /> Detailed Answer
              </h3>
              <motion.span
                className="rounded-full bg-emerald-100 px-2.5 py-0.5 text-[10px] font-semibold text-emerald-800"
                initial={{ scale: 0.9 }}
                animate={{ scale: 1 }}
              >
                {(Math.round((result.confidence ?? 0) * 100))}% confidence
              </motion.span>
            </div>
            <div className="mt-4 space-y-4">
              {formattedAnswer.summary && (
                <div>
                  <p className="text-sm font-medium text-slate-700 leading-relaxed">
                    {formattedAnswer.summary}
                  </p>
                </div>
              )}
              {formattedAnswer.details && (
                <div className="rounded-lg border border-slate-100 bg-slate-50/50 p-4">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 mb-2">Details</p>
                  <div className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap space-y-2">
                    {formattedAnswer.details.split(/\n+/).filter(Boolean).map((line, i) => (
                      <p key={i} className={line.trim().match(/^[•\-*]\s/) ? "pl-3 border-l-2 border-blue-200" : ""}>
                        {line.trim()}
                      </p>
                    ))}
                  </div>
                </div>
              )}
              {!formattedAnswer.summary && !formattedAnswer.details && (
                <p className="text-sm leading-relaxed text-slate-700">{result.answer}</p>
              )}
            </div>
          </motion.div>

          {(result.citations?.length ?? 0) > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.35 }}
              className="mt-4 rounded-xl border border-slate-200 bg-slate-50/30 p-4"
            >
              <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-600 mb-3">
                Citations — source passages and graph evidence
              </h4>
              <div className="space-y-3 max-h-72 overflow-y-auto">
                {Array.from(
                  new Map(result.citations!.map((c) => [c.chunk_id || `${c.source}-${c.page}`, c])).values()
                ).map((c, i) => (
                  <motion.div
                    key={c.chunk_id || i}
                    initial={{ opacity: 0, x: -4 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 + i * 0.05 }}
                    className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm text-left"
                  >
                    <div className="flex items-center gap-2 text-xs font-medium text-slate-500 mb-2">
                      <span>{c.source ?? "Source"}</span>
                      <span>·</span>
                      <span>Page {c.page ?? "—"}</span>
                    </div>
                    {c.text && (
                      <p className="text-sm text-slate-700 leading-relaxed mb-3">{c.text}</p>
                    )}
                    {c.graph_entities && c.graph_entities.length > 0 && (
                      <div className="mb-2">
                        <span className="text-[10px] font-semibold uppercase text-violet-600">Graph entities:</span>
                        <p className="text-xs text-slate-600 mt-0.5">{c.graph_entities.join(", ")}</p>
                      </div>
                    )}
                    {c.graph_relationships && c.graph_relationships.length > 0 && (
                      <div>
                        <span className="text-[10px] font-semibold uppercase text-violet-600">Relationships:</span>
                        <ul className="mt-0.5 space-y-0.5 text-xs font-mono text-slate-600">
                          {c.graph_relationships.map((r, j) => (
                            <li key={j}>{r}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </motion.div>
      )}
    </div>
  );
}
