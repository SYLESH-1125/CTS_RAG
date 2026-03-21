"use client";

import dynamic from "next/dynamic";
import { useMemo, useCallback } from "react";
import { motion } from "framer-motion";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false });

type GraphNode = {
  id: string;
  name?: string;
  group?: number;
  entity_type?: string;
  community_id?: number;
  val?: number;
};
type RawLink = { source: string | { id?: string }; target: string | { id?: string }; type?: string };

const ENTITY_COLORS: Record<string, string> = {
  chunk: "#94a3b8",
  concept: "#60a5fa",
  metric: "#fbbf24",
  date: "#4ade80",
  number: "#c084fc",
  default: "#93c5fd",
};

function communityStroke(comm: number): string {
  if (comm < 0) return "transparent";
  const h = (comm * 47) % 360;
  return `hsla(${h}, 75%, 58%, 0.55)`;
}

/** Ensure every link endpoint exists as a node (d3-force throws otherwise). */
function sanitizeForceGraph(graphData: { nodes: GraphNode[]; links: RawLink[] } | null): {
  nodes: GraphNode[];
  links: { source: string; target: string; type?: string }[];
} {
  if (!graphData?.nodes?.length) {
    return { nodes: [{ id: "_placeholder", name: "…", entity_type: "chunk", val: 1 }], links: [] };
  }
  const nodeById = new Map<string, GraphNode>();
  for (const n of graphData.nodes) {
    if (n?.id == null) continue;
    const id = String(n.id);
    const et = n.entity_type || (id.startsWith("chunk:") ? "chunk" : "concept");
    nodeById.set(id, {
      ...n,
      id,
      entity_type: et,
      val: typeof n.val === "number" && n.val > 0 ? n.val : 1,
      community_id: typeof n.community_id === "number" ? n.community_id : -1,
    });
  }
  const linksOut: { source: string; target: string; type?: string }[] = [];
  for (const l of graphData.links || []) {
    const sRaw = typeof l.source === "object" && l.source != null ? (l.source as { id?: string }).id : l.source;
    const tRaw = typeof l.target === "object" && l.target != null ? (l.target as { id?: string }).id : l.target;
    if (sRaw == null || tRaw == null) continue;
    const s = String(sRaw);
    const t = String(tRaw);
    const shortName = (full: string) => (full.includes("::") ? full.split("::").pop() || full : full).slice(0, 60);
    if (!nodeById.has(s)) {
      nodeById.set(s, {
        id: s,
        name: s.startsWith("chunk:") ? "chunk" : shortName(s),
        group: s.startsWith("chunk:") ? 1 : 2,
        entity_type: s.startsWith("chunk:") ? "chunk" : "concept",
        val: 1,
        community_id: -1,
      });
    }
    if (!nodeById.has(t)) {
      nodeById.set(t, {
        id: t,
        name: t.startsWith("chunk:") ? "chunk" : shortName(t),
        group: t.startsWith("chunk:") ? 1 : 2,
        entity_type: t.startsWith("chunk:") ? "chunk" : "concept",
        val: 1,
        community_id: -1,
      });
    }
    linksOut.push({ source: s, target: t, type: l.type });
  }
  return { nodes: Array.from(nodeById.values()), links: linksOut };
}

export function GraphBuilderVisualizer({
  graphData,
  progress,
  active,
  isBuilding = false,
  highlightKeys,
}: {
  graphData: { nodes: GraphNode[]; links: RawLink[] } | null;
  progress?: { current?: number; total?: number; entities?: number; relations?: number };
  active: boolean;
  isBuilding?: boolean;
  /** graph_key or id substrings to pulse-highlight during streaming */
  highlightKeys?: string[];
}) {
  const data = useMemo(() => sanitizeForceGraph(graphData), [graphData]);

  const liveNodeCount = graphData?.nodes?.length ?? 0;
  const liveLinkCount = graphData?.links?.length ?? 0;

  const highlightSet = useMemo(() => new Set((highlightKeys || []).filter(Boolean)), [highlightKeys]);

  const paint = useCallback(
    (node: object, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const n = node as GraphNode & { x?: number; y?: number };
      const label = n.name || n.id;
      if (n.x === undefined || n.y === undefined) return;
      const nx = n.x;
      const ny = n.y;
      if (!Number.isFinite(nx) || !Number.isFinite(ny) || !Number.isFinite(globalScale) || globalScale <= 0) {
        return;
      }
      const et = n.entity_type || "concept";
      const fill = ENTITY_COLORS[et] || ENTITY_COLORS.default;
      const deg = typeof n.val === "number" ? n.val : 1;
      const baseR = Math.max(2.5, Math.min(14, 3.5 + Math.sqrt(deg) * 1.4)) / globalScale;
      const comm = typeof n.community_id === "number" ? n.community_id : -1;
      const hl =
        highlightSet.has(String(n.id)) ||
        Array.from(highlightSet).some((k) => String(n.id).includes(k));

      if (comm >= 0) {
        ctx.beginPath();
        ctx.arc(nx, ny, baseR + 2.2 / globalScale, 0, 2 * Math.PI);
        ctx.strokeStyle = communityStroke(comm);
        ctx.lineWidth = 1.25 / globalScale;
        ctx.stroke();
      }

      if (hl) {
        ctx.beginPath();
        ctx.arc(nx, ny, baseR + 4 / globalScale, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(59, 130, 246, 0.2)";
        ctx.fill();
      }

      ctx.beginPath();
      ctx.arc(nx, ny, baseR, 0, 2 * Math.PI);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = hl ? "#38bdf8" : "#334155";
      ctx.lineWidth = (hl ? 1.1 : 0.55) / globalScale;
      ctx.stroke();
      ctx.font = `${10 / globalScale}px ui-sans-serif, system-ui, sans-serif`;
      ctx.fillStyle = "#e2e8f0";
      ctx.fillText(String(label).slice(0, 18), nx + baseR + 2 / globalScale, ny + 3 / globalScale);
    },
    [highlightSet]
  );

  const showGraph = active && graphData?.nodes?.length && data.nodes.some((n) => n.id !== "_placeholder");

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-slate-800">Graph build · live</h3>
        {progress && (
          <motion.span
            key={`${progress.entities}-${progress.relations}`}
            initial={{ opacity: 0.5 }}
            animate={{ opacity: 1 }}
            className="text-xs text-slate-600"
          >
            Entities {progress.entities ?? 0} · rel {progress.relations ?? 0}
          </motion.span>
        )}
      </div>

      {isBuilding && (
        <motion.div
          className="flex flex-wrap items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-700"
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <motion.span
            className="relative flex h-2 w-2 shrink-0"
            animate={{ opacity: [1, 0.45, 1] }}
            transition={{ repeat: Infinity, duration: 1.4, ease: "easeInOut" }}
          >
            <span className="absolute inline-flex h-full w-full rounded-full bg-blue-500 opacity-40" />
            <span className="relative m-0.5 inline-flex h-1.5 w-1.5 rounded-full bg-blue-600" />
          </motion.span>
          <span>
            <strong className="font-medium text-slate-800">Streaming</strong>
            {" · "}
            Chunk {progress?.current ?? 0}/{progress?.total ?? "—"}
            {" · "}
            <motion.span key={liveNodeCount} initial={{ color: "#0f172a" }} animate={{ color: "#334155" }}>
              {liveNodeCount} nodes, {liveLinkCount} edges
            </motion.span>
          </span>
        </motion.div>
      )}

      <motion.div
        layout
        className={`relative h-[380px] w-full overflow-hidden rounded-xl border bg-slate-950 shadow-inner ${
          isBuilding ? "border-blue-200/80 ring-1 ring-blue-500/20" : "border-slate-200"
        }`}
      >
        {showGraph ? (
          <ForceGraph2D
            graphData={data}
            nodeLabel="name"
            nodeVal={(n) => (n as GraphNode).val || 1}
            nodeCanvasObject={paint}
            nodePointerAreaPaint={(node, color, ctx) => {
              const n = node as GraphNode & { x?: number; y?: number };
              if (n.x === undefined || n.y === undefined) return;
              const px = n.x;
              const py = n.y;
              if (!Number.isFinite(px) || !Number.isFinite(py)) return;
              const deg = typeof n.val === "number" ? n.val : 1;
              const r = Math.max(4, Math.min(16, 6 + Math.sqrt(deg) * 1.2));
              ctx.fillStyle = color;
              ctx.beginPath();
              ctx.arc(px, py, r, 0, 2 * Math.PI);
              ctx.fill();
            }}
            linkColor={() => "#64748b"}
            linkWidth={0.65}
            linkDirectionalParticles={isBuilding ? 1 : 0}
            linkDirectionalParticleWidth={1.2}
            linkDirectionalParticleSpeed={0.004}
            backgroundColor="#0f172a"
            cooldownTicks={isBuilding ? 45 : 90}
            d3VelocityDecay={0.38}
          />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
            {isBuilding ? (
              <>
                <motion.div
                  className="h-1.5 w-48 overflow-hidden rounded-full bg-slate-800"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <motion.div
                    className="h-full bg-blue-500/90"
                    initial={{ width: "0%" }}
                    animate={{
                      width: progress?.total ? `${Math.min(100, ((progress.current || 0) / progress.total) * 100)}%` : "35%",
                    }}
                    transition={{ type: "spring", stiffness: 120, damping: 20 }}
                  />
                </motion.div>
                <p className="max-w-sm text-sm text-slate-400">
                  Stage 1: entities stream in; stage 2: relationships batch in the background. Graph updates incrementally —
                  no full reload.
                </p>
              </>
            ) : (
              <p className="text-sm text-slate-500">Upload a PDF to build a document-scoped knowledge graph.</p>
            )}
          </div>
        )}
      </motion.div>

      <div className="flex flex-wrap gap-3 text-[10px] text-slate-500">
        <span className="inline-flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-[#60a5fa]" /> concept
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-[#fbbf24]" /> metric
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-[#4ade80]" /> date
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-[#c084fc]" /> number
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-[#94a3b8]" /> chunk
        </span>
        <span className="text-slate-400">Ring hue = community</span>
      </div>
    </div>
  );
}
