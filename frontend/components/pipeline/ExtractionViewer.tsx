"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useDashboardStore } from "@/stores/useDashboardStore";

type Extraction = {
  text_units?: { original: string; translated: string; page: number }[];
  table_units?: { original?: string; translated?: string; context_ready?: string; page: number }[];
  image_units?: { merged_context?: string; original?: string; ocr_translated?: string; page: number; source?: string }[];
};

function Typewriter({ text, className }: { text: string; className?: string }) {
  const [n, setN] = useState(0);
  useEffect(() => {
    if (!text) return;
    setN(0);
    const id = setInterval(() => {
      setN((x) => {
        if (x >= text.length) {
          clearInterval(id);
          return x;
        }
        return x + Math.max(1, Math.floor(text.length / 200));
      });
    }, 12);
    return () => clearInterval(id);
  }, [text]);
  return <span className={className}>{text.slice(0, n)}</span>;
}

export function ExtractionViewer({
  extraction,
  active,
  isStreaming,
}: {
  extraction: Extraction | null;
  active: boolean;
  /** True while SSE is delivering extraction_* events */
  isStreaming?: boolean;
}) {
  const translationMode = useDashboardStore((s) => s.translationMode);
  const setTranslationMode = useDashboardStore((s) => s.setTranslationMode);
  const textUnits = extraction?.text_units ?? [];
  const text = textUnits.length ? textUnits[textUnits.length - 1] : undefined;
  const displayText =
    translationMode === "en" ? text?.translated || text?.original || "" : text?.original || text?.translated || "";

  if (!extraction && isStreaming) {
    return (
      <div className="flex h-64 flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-blue-200 bg-blue-50/30 text-sm text-slate-600">
        <motion.span
          className="h-2 w-2 rounded-full bg-blue-500"
          animate={{ opacity: [1, 0.3, 1], scale: [1, 1.2, 1] }}
          transition={{ repeat: Infinity, duration: 1.2 }}
        />
        Streaming text from PDF…
      </div>
    );
  }

  if (!extraction) {
    return (
      <div className="flex h-64 items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50/50 text-sm text-slate-500">
        Upload a PDF to see live extraction
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-slate-800">Phase 1 · Extraction</h3>
          <p className="text-[10px] text-slate-500">
            {textUnits.length} text block(s)
            {(extraction.table_units?.length ?? 0) > 0 && ` · ${extraction.table_units!.length} table(s)`}
            {(extraction.image_units?.length ?? 0) > 0 && ` · ${extraction.image_units!.length} image(s)`}
          </p>
        </div>
        <div className="flex rounded-lg border border-slate-200 bg-white p-0.5 text-xs">
          <button
            type="button"
            onClick={() => setTranslationMode("en")}
            className={`rounded-md px-2 py-1 ${translationMode === "en" ? "bg-slate-900 text-white" : "text-slate-600"}`}
          >
            English
          </button>
          <button
            type="button"
            onClick={() => setTranslationMode("original")}
            className={`rounded-md px-2 py-1 ${translationMode === "original" ? "bg-slate-900 text-white" : "text-slate-600"}`}
          >
            Original
          </button>
        </div>
      </div>

      <AnimatePresence mode="wait">
        {active && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="space-y-3"
          >
            <motion.div
              className="rounded-xl border border-blue-200/80 bg-blue-50/40 p-4 shadow-sm"
              layout
            >
              <span className="text-xs font-medium uppercase tracking-wide text-blue-700">Text</span>
              <p className="mt-2 max-h-40 overflow-y-auto text-sm leading-relaxed text-slate-800">
                {active ? <Typewriter text={displayText.slice(0, 1200)} /> : displayText.slice(0, 1200)}
              </p>
            </motion.div>

            {(extraction.table_units?.length ?? 0) > 0 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
                className="rounded-xl border border-emerald-200/80 bg-emerald-50/40 p-4 shadow-sm"
              >
                <span className="text-xs font-medium uppercase tracking-wide text-emerald-800">Table (latest)</span>
                <div className="mt-2 overflow-x-auto text-sm">
                  <table className="w-full border-collapse text-left">
                    <tbody>
                      {(() => {
                        const tbl = extraction.table_units![extraction.table_units!.length - 1];
                        const tableContent =
                          translationMode === "en"
                            ? tbl?.translated || tbl?.context_ready || tbl?.original || ""
                            : tbl?.original || tbl?.context_ready || "";
                        return tableContent.split("\n").slice(0, 8);
                      })().map((row, i) => (
                        <tr key={i} className="border-b border-emerald-100">
                          {row.split("|").map((cell, j) => (
                            <td key={j} className="px-2 py-1 text-slate-700">
                              {cell.trim()}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </motion.div>
            )}

            {(extraction.image_units?.length ?? 0) > 0 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.35 }}
                className="rounded-xl border border-violet-200/80 bg-violet-50/40 p-4 shadow-sm"
              >
                <span className="text-xs font-medium uppercase tracking-wide text-violet-800">Image / chart (latest)</span>
                <p className="mt-2 whitespace-pre-wrap text-sm text-slate-800">
                  {(() => {
                    const u = extraction.image_units![extraction.image_units!.length - 1];
                    return translationMode === "en"
                      ? u?.ocr_translated || u?.merged_context || u?.original || ""
                      : u?.original || u?.merged_context || "";
                  })()}
                </p>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
