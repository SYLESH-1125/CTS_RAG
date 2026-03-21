import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export const api = axios.create({
  baseURL: API_BASE,
  headers: { "Content-Type": "application/json" },
});

export async function uploadPdf(file: File) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post("/upload/pdf", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function getJobStatus(jobId: string) {
  const { data } = await api.get(`/upload/status/${jobId}`);
  return data;
}

export function getJobStatusStreamUrl(jobId: string) {
  return `${API_BASE.replace("/api", "")}/api/upload/status/${jobId}/stream`;
}

/** Explicitly push graph to Neo4j (used by classic upload after pipeline completes). */
export async function pushToNeo4j(jobId: string) {
  const { data } = await api.post(`/upload/push-neo4j/${jobId}`);
  return data;
}

/** Fetch Neo4j total counts (chunks, entities, relationships) for verification. */
export async function getNeo4jStats() {
  const { data } = await api.get("/upload/neo4j-stats");
  return data as {
    connected: boolean;
    chunks: number;
    entities: number;
    relationships: number;
    by_document?: { document_id: string; chunks: number }[];
    error?: string;
  };
}

/** Granular SSE: extraction_text, chunk_created, entity_created, graph_update, … */
export function getPipelineEventsStreamUrl(sessionId: string) {
  return `${API_BASE.replace("/api", "")}/api/stream/${sessionId}`;
}

export async function queryRag(
  query: string,
  documentId?: string | null,
  searchAll?: boolean
) {
  const { data } = await api.post("/query/", {
    query,
    document_id: documentId || undefined,
    search_all: searchAll ?? false,
  });
  return data;
}

export async function getDocumentGraph(documentId: string) {
  const { data } = await api.get(`/graph/document/${documentId}`);
  return data as {
    nodes: { id: string; name?: string; group?: number; entity_type?: string }[];
    links: { source: string; target: string; type?: string }[];
  };
}
