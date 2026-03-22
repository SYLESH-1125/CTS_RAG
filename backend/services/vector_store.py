"""
Phase 2: FAISS Vector Store.
FAISS is ONLY for semantic retrieval. chunk_id bridges to Neo4j.
Index -> chunk_id mapping. No graph data stored.
"""
import logging
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from config import get_settings
from services.embedder import get_embedder

logger = logging.getLogger("graph_rag.services.vector_store")


class VectorStoreService:
    """
    FAISS: embedding -> index, index -> chunk_id mapping.
    add_chunk(chunk_id, text) / search(query) -> top chunk_ids + metadata.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._index = None
        self._metadata: list[dict] = []  # [{index: i, chunk_id: "c1", ...}]
        self._load_or_init()
    
    def _index_path(self) -> Path:
        return Path(self.settings.faiss_index_dir)
    
    def _load_or_init(self):
        idx_path = self._index_path() / "index.faiss"
        meta_path = self._index_path() / "metadata.json"
        self._index_path().mkdir(parents=True, exist_ok=True)
        
        if idx_path.exists() and meta_path.exists():
            try:
                self._index = faiss.read_index(str(idx_path))
                with open(meta_path) as f:
                    self._metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load FAISS: {e}")
                self._init_index()
        else:
            self._init_index()
    
    def _init_index(self):
        embedder = get_embedder()
        dim = embedder.get_sentence_embedding_dimension()
        self._index = faiss.IndexFlatIP(dim)
        self._metadata = []

    def replace_chunks(self, chunks: list[dict], document_id: str = "") -> None:
        """Replace entire index (resets all prior data). Use add_chunks for cumulative."""
        self._init_index()
        if chunks:
            self.upsert_chunks(chunks, document_id=document_id)
        self._save()
        logger.info(f"FAISS replaced with {len(chunks)} chunks (doc={document_id[:8] if document_id else 'n/a'})")

    def add_chunks(self, chunks: list[dict], document_id: str = "") -> None:
        """Append chunks to existing index (cumulative). Does not clear prior uploads."""
        if not chunks:
            return
        self._load_or_init()  # Ensure we have existing index (or create empty)
        self.upsert_chunks(chunks, document_id=document_id)
        logger.info(
            f"FAISS added {len(chunks)} chunks (doc={document_id[:8] if document_id else 'n/a'}), "
            f"total index size={len(self._metadata)}"
        )

    def add_chunk(self, chunk_id: str, text: str, source: str = "", page: int = 0) -> None:
        """Add a single chunk to FAISS. chunk_id is the bridge to Neo4j."""
        if not chunk_id or not text.strip():
            logger.debug("add_chunk: skipping empty chunk_id or text")
            return
        embedder = get_embedder()
        emb = embedder.encode([text], normalize_embeddings=True)
        idx = len(self._metadata)
        self._index.add(emb)
        self._metadata.append({
            "index": idx,
            "chunk_id": chunk_id,
            "text": text[:2000],
            "source": source,
            "page": page,
        })
        self._save()
    
    def upsert_chunks(self, chunks: list[dict], document_id: str = "") -> None:
        """Add multiple chunks. Tags rows with document_id for per-PDF isolation."""
        if not chunks:
            return
        embedder = get_embedder()
        texts = [c.get("text", "") for c in chunks]
        embeddings = embedder.encode(texts, normalize_embeddings=True)
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id", "")
            if not chunk_id:
                continue
            doc = document_id or chunk.get("document_id") or ""
            self._index.add(embeddings[i : i + 1])
            self._metadata.append({
                "index": len(self._metadata),
                "chunk_id": chunk_id,
                "document_id": doc,
                "text": chunk.get("text", "")[:2000],
                "source": chunk.get("source", ""),
                "page": chunk.get("page", 0),
            })
        self._save()
    
    def search_by_keywords(
        self,
        keywords: list[str],
        document_id: str | None = None,
        k: int = 8,
    ) -> list[dict]:
        """Chunks containing ALL keywords. For exact lookups (e.g. cost in 2040)."""
        if not keywords:
            return []
        kw_lower = [kw.strip().lower() for kw in keywords if kw and len(kw.strip()) >= 2]
        if not kw_lower:
            return []
        results = []
        for meta in self._metadata:
            if document_id and meta.get("document_id") != document_id:
                continue
            text = (meta.get("text") or "").lower()
            if all(kw in text for kw in kw_lower):
                results.append({**meta.copy(), "score": 1.0})
        return results[:k]

    def search(
        self,
        query: str,
        k: int = 5,
        keyword_filter: str | None = None,
        document_id: str | None = None,
    ) -> list[dict]:
        """
        Semantic search. document_id scopes hits to the active PDF only.
        """
        if not self._metadata:
            return []
        embedder = get_embedder()
        q_emb = embedder.encode([query], normalize_embeddings=True)
        n = len(self._metadata)
        scan = min(n, max(k * 15, k * 3) if document_id else (k * 2 if keyword_filter else k))
        scores, indices = self._index.search(q_emb, scan)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx].copy()
            if document_id:
                if meta.get("document_id") != document_id:
                    continue
            if keyword_filter and keyword_filter.lower() not in meta.get("text", "").lower():
                continue
            meta["score"] = float(score)
            results.append(meta)
            if len(results) >= k:
                break
        return results[:k]
    
    def search_chunk_ids(self, query: str, k: int = 5) -> list[str]:
        """Return only chunk_ids for Neo4j lookup."""
        hits = self.search(query, k=k)
        return [h["chunk_id"] for h in hits if h.get("chunk_id")]
    
    def _save(self):
        self._index_path().mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path() / "index.faiss"))
        with open(self._index_path() / "metadata.json", "w") as f:
            json.dump(self._metadata, f, indent=2)
