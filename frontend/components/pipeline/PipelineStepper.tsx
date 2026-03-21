"use client";

import { motion } from "framer-motion";

export type StepId = "extraction" | "chunking" | "graph" | "query" | "output";

const STEPS: { id: StepId; label: string }[] = [
  { id: "extraction", label: "Extraction" },
  { id: "chunking", label: "Chunking" },
  { id: "graph", label: "Graph" },
  { id: "query", label: "Query" },
  { id: "output", label: "Output" },
];

function pctForPhase(phase: string | undefined): Partial<Record<StepId, number>> {
  if (!phase) return {};
  if (phase === "extraction") return { extraction: 40 };
  if (phase === "chunking") return { extraction: 100, chunking: 35 };
  if (phase === "graph_build") return { extraction: 100, chunking: 100, graph: 45 };
  if (phase === "vector_store") return { extraction: 100, chunking: 100, graph: 100, output: 50 };
  if (phase === "completed") return { extraction: 100, chunking: 100, graph: 100, query: 100, output: 100 };
  return {};
}

export function PipelineStepper({
  phase,
  centerStage,
  progress,
  status,
}: {
  phase?: string;
  centerStage: "extraction" | "chunking" | "graph" | "query";
  progress?: {
    extraction?: { current?: number; total?: number; step?: string };
    chunking?: { current?: number; total?: number; stage?: string };
    graph?: { current?: number; total?: number; entities?: number; relations?: number; percent?: number };
  };
  status?: string;
}) {
  const inferred = pctForPhase(phase);
  const ex = progress?.extraction;
  const ch = progress?.chunking;
  const gr = progress?.graph;

  const extractionPct =
    inferred.extraction ??
    (ex?.total && ex.total > 0 ? Math.min(100, Math.round((100 * (ex.current ?? 0)) / ex.total)) : phase === "extraction" ? 25 : 0);

  const chunkingPct =
    inferred.chunking ??
    (ch?.total && ch.total > 0 && ch.stage === "chunks"
      ? Math.min(100, Math.round((100 * (ch.current ?? 0)) / ch.total))
      : ch?.stage && ch.stage !== "done"
        ? 15
        : 0);

  const graphPct =
    inferred.graph ??
    (typeof gr?.percent === "number"
      ? Math.min(100, gr.percent)
      : gr?.total && gr.total > 0
        ? Math.min(100, Math.round((100 * (gr.current ?? 0)) / gr.total))
        : 0);

  const queryPct = inferred.query ?? (centerStage === "query" && status === "completed" ? 100 : centerStage === "query" ? 30 : 0);

  const outputPct = inferred.output ?? (status === "completed" ? 100 : 0);

  const pctByStep: Record<StepId, number> = {
    extraction: extractionPct,
    chunking: chunkingPct,
    graph: graphPct,
    query: queryPct,
    output: outputPct,
  };

  const activeIndex = Math.max(
    0,
    STEPS.findIndex((s) => s.id === (centerStage as StepId))
  );
  const processing = status === "processing";

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">Pipeline</h2>
        {processing && (
          <motion.span
            className="text-[10px] font-medium text-blue-600"
            animate={{ opacity: [1, 0.5, 1] }}
            transition={{ repeat: Infinity, duration: 1.5 }}
          >
            Live
          </motion.span>
        )}
      </div>
      <ol className="flex flex-wrap gap-2 lg:flex-nowrap lg:gap-0">
        {STEPS.map((s, i) => {
          const pct = pctByStep[s.id];
          const isActive = i === activeIndex || (s.id === "graph" && centerStage === "graph");
          const isDone = pct >= 100;
          return (
            <li key={s.id} className="relative flex min-w-[100px] flex-1 flex-col gap-1">
              {i > 0 && (
                <div
                  className="absolute left-0 top-[10px] hidden h-px w-2 bg-slate-200 lg:block"
                  style={{ transform: "translateX(-8px)" }}
                  aria-hidden
                />
              )}
              <div className="flex items-center gap-2">
                <span
                  className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-bold ${
                    isDone ? "bg-emerald-500 text-white" : isActive ? "bg-slate-900 text-white" : "bg-slate-200 text-slate-600"
                  }`}
                >
                  {i + 1}
                </span>
                <span className={`text-[11px] font-medium ${isActive ? "text-slate-900" : "text-slate-500"}`}>{s.label}</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-slate-100">
                <motion.div
                  className={`h-full ${isDone ? "bg-emerald-500" : "bg-blue-500"}`}
                  initial={false}
                  animate={{ width: `${Math.min(100, pct)}%` }}
                  transition={{ type: "spring", stiffness: 200, damping: 28 }}
                />
              </div>
              <span className="text-[9px] tabular-nums text-slate-400">{Math.round(pct)}%</span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
