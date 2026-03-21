import { create } from "zustand";

export type PipelinePhase = "idle" | "extraction" | "chunking" | "graph_build" | "vector_store" | "completed" | "failed";

export interface JobProgress {
  extraction?: { step?: string; current?: number; total?: number; detail?: string };
  chunking?: { current?: number; total?: number; stage?: string };
  graph?: { current?: number; total?: number; entities?: number; relations?: number };
}

interface DashboardState {
  documentId: string | null;
  phase: PipelinePhase;
  jobState: Record<string, unknown> | null;
  liveGraph: { nodes: unknown[]; links: unknown[] } | null;
  centerStage: "extraction" | "chunking" | "graph" | "query";
  translationMode: "en" | "original";
  setDocumentId: (id: string | null) => void;
  setJobState: (s: Record<string, unknown> | null) => void;
  setCenterStage: (s: "extraction" | "chunking" | "graph" | "query") => void;
  setTranslationMode: (m: "en" | "original") => void;
}

export const useDashboardStore = create<DashboardState>((set) => ({
  documentId: null,
  phase: "idle",
  jobState: null,
  liveGraph: null,
  centerStage: "extraction",
  translationMode: "en",
  setDocumentId: (id) => set({ documentId: id }),
  setJobState: (s) =>
    set({
      jobState: s,
      liveGraph: (s?.live_graph as DashboardState["liveGraph"]) ?? null,
      phase: (s?.phase as PipelinePhase) || "idle",
    }),
  setCenterStage: (centerStage) => set({ centerStage }),
  setTranslationMode: (translationMode) => set({ translationMode }),
}));
