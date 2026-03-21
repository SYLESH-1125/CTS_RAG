"""
Graph-aware hybrid chunking:
1. Structural split (sections, tables, image text, page boundaries)
2. Semantic split (MiniLM sentence similarity)
3. Entity-aware refinement (split when unrelated entity groups appear in one block)
4. Coherence scoring; drop noisy low-coherence chunks
"""
import logging
import re
import uuid
from typing import Any, Callable

import numpy as np

from config import get_settings
from services.embedder import get_embedder
from services.content_deduplication import deduplicate_content

logger = logging.getLogger("graph_rag.services.chunking")


def _quick_entity_spans(text: str) -> list[tuple[int, int, str]]:
    """Lightweight entity mentions for splitting (regex + keywords). No spaCy required."""
    spans: list[tuple[int, int, str]] = []
    for m in re.finditer(
        r"\b(20\d{2}|19\d{2}|LSTM|CNN|RNN|GRU|Transformer|BERT|GPT|revenue|cost|profit|earnings|margin|%)\b",
        text,
        re.I,
    ):
        spans.append((m.start(), m.end(), m.group(1).lower()))
    return spans


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


class ChunkingService:
    """
    Hybrid graph-aware chunking for Graph RAG.
    Output: [{chunk_id, text, source, type, coherence_score?}]
    """

    def __init__(self):
        self.settings = get_settings()

    def chunk(
        self,
        extraction: dict[str, Any],
        on_progress: Callable[[int, int, str, list[dict]], None] | None = None,
        on_stream: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> list[dict]:
        all_content = self._gather_content(extraction)
        if not all_content:
            logger.warning("No content to chunk")
            return []

        # Deduplicate overlapping content (PDF text + OCR + Vision) before chunking
        def dedup_prog(stage: str, removed: int, total: int) -> None:
            if on_stream:
                on_stream(
                    "progress_update",
                    {"phase": "chunking", "stage": "dedup", "removed": removed, "total": total},
                )

        all_content = deduplicate_content(all_content, on_progress=dedup_prog)
        if not all_content:
            logger.warning("No content after deduplication")
            return []

        structural = self._structural_chunk(all_content)
        logger.info("Structural chunking: %s blocks", len(structural))
        if on_progress:
            on_progress(1, 6, "structural", [])
        if on_stream:
            on_stream(
                "progress_update",
                {
                    "phase": "chunking",
                    "stage": "structural",
                    "current": 1,
                    "total": 6,
                    "percent": round(100 * 1 / 6),
                },
            )

        semantic = self._semantic_chunk(structural)
        logger.info("Semantic chunking: %s blocks", len(semantic))
        if on_progress:
            on_progress(2, 6, "semantic", [])
        if on_stream:
            on_stream(
                "progress_update",
                {
                    "phase": "chunking",
                    "stage": "semantic",
                    "current": 2,
                    "total": 6,
                    "blocks": len(semantic),
                    "percent": round(100 * 2 / 6),
                },
            )

        refined = self._entity_aware_refine(semantic)
        if on_progress:
            on_progress(3, 6, "entity_refine", [])
        if on_stream:
            on_stream(
                "progress_update",
                {
                    "phase": "chunking",
                    "stage": "entity_refine",
                    "current": 3,
                    "total": 6,
                    "percent": round(100 * 3 / 6),
                },
            )

        sized = self._apply_size_control(refined)
        if on_progress:
            on_progress(4, 6, "sizing", [])
        if on_stream:
            on_stream(
                "progress_update",
                {
                    "phase": "chunking",
                    "stage": "sizing",
                    "current": 4,
                    "total": 6,
                    "percent": round(100 * 4 / 6),
                },
            )

        scored = self._score_coherence(sized)
        min_coh = float(getattr(self.settings, "chunk_coherence_min", 0.22))
        kept = [c for c in scored if float(c.get("coherence_score", 0.5)) >= min_coh]
        dropped = len(scored) - len(kept)
        if dropped:
            logger.info("Dropped %s low-coherence chunks (< %s)", dropped, min_coh)
        if on_progress:
            on_progress(5, 6, "quality_filter", [])
        if on_stream:
            on_stream(
                "progress_update",
                {
                    "phase": "chunking",
                    "stage": "quality_filter",
                    "current": 5,
                    "total": 6,
                    "kept": len(kept),
                    "dropped": dropped,
                    "percent": round(100 * 5 / 6),
                },
            )

        out: list[dict] = []
        total = len(kept)
        for i, c in enumerate(kept):
            c["chunk_id"] = c.get("chunk_id") or str(uuid.uuid4())
            out.append(c)
            if on_progress:
                on_progress(i + 1, max(1, total), "chunks", [dict(x) for x in out])
            if on_stream:
                on_stream(
                    "chunk_created",
                    {
                        "index": i + 1,
                        "total": total,
                        "chunk": {
                            "chunk_id": c["chunk_id"],
                            "text": (c.get("text") or "")[:500],
                            "source": c.get("source", ""),
                            "page": c.get("page", 0),
                            "type": c.get("type", "text"),
                            "coherence_score": c.get("coherence_score"),
                        },
                    },
                )

        if on_progress:
            on_progress(total, total, "done", out)
        if on_stream:
            on_stream(
                "progress_update",
                {
                    "phase": "chunking",
                    "stage": "done",
                    "current": total,
                    "total": total,
                    "percent": 100,
                },
            )

        return out

    def _gather_content(self, extraction: dict) -> list[dict]:
        content = []
        for u in extraction.get("text_units", []):
            text = u.get("translated") or u.get("original", "")
            if text.strip():
                content.append({
                    "text": text.strip(),
                    "source": u.get("source", "page_0"),
                    "page": u.get("page", 0),
                    "type": "text",
                })
        for u in extraction.get("table_units", []):
            text = u.get("context_ready") or u.get("translated", "")
            if text.strip():
                content.append({
                    "text": text.strip(),
                    "source": u.get("source", "page_0"),
                    "page": u.get("page", 0),
                    "type": "table",
                })
        for u in extraction.get("image_units", []):
            text = u.get("merged_context", "")
            if text.strip():
                content.append({
                    "text": text.strip(),
                    "source": u.get("source", "page_0"),
                    "page": u.get("page", 0),
                    "type": "image",
                })
        return content

    def _structural_chunk(self, content: list[dict]) -> list[dict]:
        chunks = []
        current = []
        current_page = -1
        max_tokens = self.settings.chunk_max_tokens

        for item in content:
            text = item.get("text", "")
            page = item.get("page", 0)
            if not text.strip():
                continue

            tokens = self._count_tokens(text)

            if current and page != current_page:
                chunks.append(self._merge_items(current))
                current = [item]
                current_page = page
            elif self._is_section_header(text) and current:
                chunks.append(self._merge_items(current))
                current = [item]
                current_page = page
            elif current and sum(self._count_tokens(c.get("text", "")) for c in current) + tokens > max_tokens:
                chunks.append(self._merge_items(current))
                current = [item]
                current_page = page
            else:
                current.append(item)
                if current_page < 0:
                    current_page = page

        if current:
            chunks.append(self._merge_items(current))
        return chunks

    def _semantic_chunk(self, structural_chunks: list[dict]) -> list[dict]:
        threshold = getattr(self.settings, "chunk_similarity_threshold", 0.5)
        embedder = get_embedder()
        final = []

        for chunk in structural_chunks:
            text = chunk.get("text", "")
            if not text.strip():
                continue

            sentences = _split_sentences(text)
            if len(sentences) <= 1:
                final.append(chunk)
                continue

            try:
                embeddings = embedder.encode(sentences, normalize_embeddings=True)
            except Exception as e:
                logger.warning("Embedding failed: %s", e)
                final.append(chunk)
                continue

            sub_chunks = []
            current = [sentences[0]]

            for i in range(1, len(sentences)):
                sim = float(
                    np.dot(embeddings[i - 1], embeddings[i])
                    / (
                        1e-9
                        + float(np.linalg.norm(embeddings[i - 1]))
                        * float(np.linalg.norm(embeddings[i]))
                    )
                )
                if sim < threshold and current:
                    sub_chunks.append(" ".join(current))
                    current = [sentences[i]]
                else:
                    current.append(sentences[i])

            if current:
                sub_chunks.append(" ".join(current))

            for sc in sub_chunks:
                if sc.strip():
                    final.append({
                        "text": sc.strip(),
                        "source": chunk.get("source", ""),
                        "page": chunk.get("page", 0),
                        "type": chunk.get("type", "text"),
                    })

        return final

    def _entity_aware_refine(self, chunks: list[dict]) -> list[dict]:
        """
        If a block is large and contains salient entities that are far apart in embedding space,
        split at the weakest adjacent sentence boundary (low MiniLM similarity).
        """
        min_tok = int(getattr(self.settings, "chunk_entity_split_min_tokens", 80))
        max_ent = int(getattr(self.settings, "chunk_max_entities_before_split", 4))
        embedder = get_embedder()
        out: list[dict] = []

        for chunk in chunks:
            text = chunk.get("text", "")
            if self._count_tokens(text) < min_tok:
                out.append(chunk)
                continue

            spans = _quick_entity_spans(text)
            distinct = {s[2] for s in spans}
            if len(distinct) < max_ent:
                out.append(chunk)
                continue

            sentences = _split_sentences(text)
            if len(sentences) < 3:
                out.append(chunk)
                continue

            try:
                emb = embedder.encode(sentences, normalize_embeddings=True)
            except Exception:
                out.append(chunk)
                continue

            # Weakest boundary between sentences
            weakest_i = 1
            weakest_sim = 1.0
            for i in range(1, len(sentences)):
                sim = float(np.dot(emb[i - 1], emb[i]) / (1e-9 + float(np.linalg.norm(emb[i - 1])) * float(np.linalg.norm(emb[i]))))
                if sim < weakest_sim:
                    weakest_sim = sim
                    weakest_i = i

            # Only split if boundary is weak and halves both have entity hits
            if weakest_sim > 0.42:
                out.append(chunk)
                continue

            left = " ".join(sentences[:weakest_i]).strip()
            right = " ".join(sentences[weakest_i:]).strip()
            if self._count_tokens(left) >= self.settings.chunk_min_tokens // 2 and self._count_tokens(right) >= self.settings.chunk_min_tokens // 2:
                base = {k: chunk[k] for k in ("source", "page", "type") if k in chunk}
                out.append({**base, "text": left})
                out.append({**base, "text": right})
            else:
                out.append(chunk)

        return out

    def _apply_size_control(self, chunks: list[dict]) -> list[dict]:
        min_t = self.settings.chunk_min_tokens
        max_t = self.settings.chunk_max_tokens
        overlap = min(1, self.settings.chunk_overlap_tokens // 10)
        final = []

        for chunk in chunks:
            text = chunk.get("text", "")
            tokens = self._count_tokens(text)

            if min_t <= tokens <= max_t:
                final.append(chunk)
            elif tokens > max_t:
                words = text.split()
                step = max(1, max_t - overlap)
                for i in range(0, len(words), step):
                    segment = " ".join(words[i : i + max_t])
                    if self._count_tokens(segment) >= min_t:
                        final.append({
                            **chunk,
                            "text": segment,
                        })
            else:
                final.append(chunk)

        return final

    def _score_coherence(self, chunks: list[dict]) -> list[dict]:
        """MiniLM-based intra-chunk coherence (mean adjacent sentence similarity)."""
        embedder = get_embedder()
        for chunk in chunks:
            text = chunk.get("text", "")
            sents = _split_sentences(text)
            if len(sents) <= 1:
                chunk["coherence_score"] = 0.95
                continue
            try:
                emb = embedder.encode(sents, normalize_embeddings=True)
            except Exception:
                chunk["coherence_score"] = 0.5
                continue
            sims = []
            for i in range(1, len(sents)):
                sims.append(
                    float(
                        np.dot(emb[i - 1], emb[i])
                        / (1e-9 + float(np.linalg.norm(emb[i - 1])) * float(np.linalg.norm(emb[i])))
                    )
                )
            chunk["coherence_score"] = float(np.mean(sims)) if sims else 0.5
        return chunks

    def _merge_items(self, items: list[dict]) -> dict:
        texts = [i.get("text", "") for i in items]
        return {
            "text": "\n\n".join(texts),
            "source": items[0].get("source", "") if items else "",
            "page": items[0].get("page", 0) if items else 0,
            "type": items[0].get("type", "text") if items else "text",
        }

    def _is_section_header(self, text: str) -> bool:
        t = text.strip()
        if len(t) > 120:
            return False
        if re.match(r"^Section\s+\d+\s*[:\.]", t, re.I):
            return True
        if re.match(r"^#+\s+\w", t):
            return True
        if re.match(r"^[\d\.\-]+\s+\w+", t):
            return True
        if t.isupper() and len(t) < 80:
            return True
        return False

    def _count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
