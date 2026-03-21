import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Graph RAG | Production",
  description: "Graph-first RAG with multimodal ingestion",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-surface-50 text-surface-900 antialiased" style={{ backgroundColor: "#fafafa" }}>
        {children}
      </body>
    </html>
  );
}
