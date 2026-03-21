"use client";

import { motion } from "framer-motion";

type Chunk = { chunk_id: string; text: string; source: string; page?: number };

export function ChunkingVisualizer({
  chunks,
  progress,
  active,
}: {
  chunks: Chunk[] | null;
  progress?: { current?: number; total?: number; stage?: string };
  active: boolean;
}) {
  const stage = progress?.stage || "";
  const structural = stage === "structural" || stage === "semantic" || stage === "sizing";
  const showCards = (chunks?.length ?? 0) > 0;

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-slate-800">Phase 2 · Chunking</h3>
      <div className="flex gap-2 text-xs text-slate-600">
        <Step label="1 · Structural" on={structural && stage === "structural"} done={!!chunks?.length} />
        <Step label="2 · Semantic" on={stage === "semantic"} done={stage === "sizing" || stage === "chunks" || stage === "done"} />
        <Step label="3 · Final" on={stage === "chunks" || stage === "done"} done={stage === "done"} />
      </div>
      {active && progress?.total ? (
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
          <motion.div
            className="h-full bg-slate-800"
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(100, ((progress.current || 0) / progress.total) * 100)}%` }}
          />
        </div>
      ) : null}
      <div className="grid max-h-72 gap-2 overflow-y-auto sm:grid-cols-2">
        {showCards &&
          chunks!.map((c, i) => (
            <motion.div
              key={c.chunk_id || i}
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: Math.min(i * 0.04, 0.5) }}
              className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm"
            >
              <div className="flex items-center justify-between text-[10px] text-slate-500">
                <span className="font-mono truncate">
                  {(c.chunk_id || `chunk-${i}`).slice(0, 12)}
                  {(c.chunk_id || "").length > 12 ? "…" : ""}
                </span>
                <span>{c.source}</span>
              </div>
              <p className="mt-1 line-clamp-3 text-xs text-slate-700">{c.text}</p>
            </motion.div>
          ))}
      </div>
    </div>
  );
}

function Step({ label, on, done }: { label: string; on: boolean; done: boolean }) {
  return (
    <span
      className={`rounded-full px-2 py-1 ${
        done ? "bg-slate-900 text-white" : on ? "bg-slate-200 text-slate-900" : "bg-slate-100 text-slate-500"
      }`}
    >
      {label}
    </span>
  );
}
