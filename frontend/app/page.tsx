"use client";

import { useState } from "react";
import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-surface-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4">
          <span className="font-semibold text-surface-900">Graph RAG</span>
          <nav className="flex gap-6">
            <Link href="/upload" className="text-sm text-surface-600 hover:text-surface-900">
              Upload
            </Link>
            <Link href="/query" className="text-sm text-surface-600 hover:text-surface-900">
              Query
            </Link>
            <Link href="/dashboard" className="text-sm font-medium text-accent hover:text-accent-hover">
              Live pipeline
            </Link>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-16">
        <div className="text-center">
          <h1 className="text-3xl font-bold tracking-tight text-surface-900">
            Graph RAG System
          </h1>
          <p className="mt-2 text-surface-600">
            Upload PDFs, extract text, tables, and images. Query with graph-first retrieval.
          </p>
          <div className="mt-8 flex flex-wrap justify-center gap-4">
            <Link
              href="/dashboard"
              className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover"
            >
              Pipeline dashboard
            </Link>
            <Link
              href="/upload"
              className="rounded-lg border border-surface-300 px-4 py-2 text-sm font-medium text-surface-700 hover:bg-surface-100"
            >
              Classic upload
            </Link>
            <Link
              href="/query"
              className="rounded-lg border border-surface-300 px-4 py-2 text-sm font-medium text-surface-700 hover:bg-surface-100"
            >
              Query
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
