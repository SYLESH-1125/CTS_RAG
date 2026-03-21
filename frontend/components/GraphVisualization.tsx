"use client";

import { useMemo } from "react";
import dynamic from "next/dynamic";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d").then((mod) => mod.default), {
  ssr: false,
  loading: () => <div className="flex h-full items-center justify-center text-surface-500">Loading graph...</div>,
});

interface GraphNode {
  id: string;
  labels: string[];
  props: Record<string, unknown>;
}

interface GraphVisualizationProps {
  nodes: GraphNode[];
}

export function GraphVisualization({ nodes }: GraphVisualizationProps) {
  const graphData = useMemo(() => {
    const gNodes = nodes.map((n) => ({
      id: n.id,
      name: (n.props?.name as string) || (n.props?.chunk_id as string)?.slice(0, 8) || n.id,
      labels: n.labels,
    }));
    return { nodes: gNodes, links: [] as Array<{ source: string; target: string }> };
  }, [nodes]);

  if (nodes.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center rounded bg-surface-100 text-surface-500 text-sm">
        No graph nodes for this query
      </div>
    );
  }

  return (
    <ForceGraph2D
      graphData={graphData}
      width={400}
      height={256}
      nodeLabel={(n) => {
        const node = n as { name?: string; id?: string | number };
        return String(node.name ?? node.id ?? "");
      }}
      nodeColor={(n) => {
        const node = n as { labels?: string[] };
        return node.labels?.includes("Entity") ? "#0ea5e9" : "#94a3b8";
      }}
    />
  );
}
