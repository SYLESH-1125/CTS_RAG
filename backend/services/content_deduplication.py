"""
Content deduplication across extraction sources.
Prevents duplicate chunks from PDF text + OCR + Vision overlap.
Pipeline: normalize -> hash check -> embedding similarity -> merge best version.
"""
import hashlib
import logging
import re
from typing import Any, Callable

import numpy as np

from config import get_settings
from services.embedder import get_embedder

logger = logging.getLogger("graph_rag.services.content_deduplication")

# Source priority: higher = prefer when merging duplicates
# Vision LLM > PDF text > OCR
PRIORITY_VISION = 3
PRIORITY_TABLE = 2
PRIORITY_TEXT = 2
PRIORITY_OCR = 1


def normalize_text(text: str) -> str:
    """
    Normalize for comparison: lowercase, remove punctuation, collapse spaces, standardize numbers.
    Example: "Revenue increased to 150 in 2023" -> "revenue increased to 150 in 2023"
    """
    if not text or not isinstance(text, str):
        return ""
    s = text.strip().lower()
    s = re.sub(r"[^\w\s\d\.\-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    # Standardize common number patterns (optional - keeps semantic similarity)
    s = re.sub(r"\b(\d+)\s*(?:million|mn|m)\b", r"\1m", s, flags=re.I)
    s = re.sub(r"\b(\d+)\s*(?:billion|bn|b)\b", r"\1b", s, flags=re.I)
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*%\b", r"\1 percent", s)
    s = s.strip()
    return s


def content_hash(text: str) -> str:
    """Hash of normalized text for fast exact-duplicate detection."""
    return hashlib.sha256(normalize_text(text).encode("utf-8", errors="ignore")).hexdigest()


def merge_content(
    unit_a: dict[str, Any],
    unit_b: dict[str, Any],
    priority_a: int,
    priority_b: int,
) -> dict[str, Any]:
    """
    Merge two overlapping units into one. Keep most complete sentence; preserve numbers.
    If priorities equal, prefer longer text.
    """
    text_a = (unit_a.get("text") or unit_a.get("merged_context") or unit_a.get("translated") or unit_a.get("original") or "").strip()
    text_b = (unit_b.get("text") or unit_b.get("merged_context") or unit_b.get("translated") or unit_b.get("original") or "").strip()

    if not text_a:
        return dict(unit_b)
    if not text_b:
        return dict(unit_a)

    preferred = unit_a if priority_a >= priority_b else unit_b
    other = unit_b if priority_a >= priority_b else unit_a
    preferred_text = text_a if priority_a >= priority_b else text_b
    other_text = text_b if priority_a >= priority_b else text_a

    # Preserve numbers from OCR if missing in preferred (OCR often has ground-truth digits)
    pref_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", preferred_text))
    other_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", other_text))
    missing_nums = other_nums - pref_nums
    merged_text = preferred_text
    if missing_nums and len(preferred_text) < len(other_text) * 1.2:
        # Append missing numeric facts as short phrases
        extra = " ".join(sorted(missing_nums)[:5])
        if extra and extra not in merged_text:
            merged_text = f"{merged_text} {extra}".strip()

    result = dict(preferred)
    if "text" in result:
        result["text"] = merged_text
    if "merged_context" in result:
        result["merged_context"] = merged_text
    if "translated" in result:
        result["translated"] = merged_text
    return result


def _source_priority(unit: dict[str, Any]) -> int:
    """Assign priority for merge: vision > table/text > ocr (Vision LLM > PDF text > OCR)."""
    utype = str(unit.get("type", "")).lower()
    if utype == "table":
        return PRIORITY_TABLE
    if utype == "text":
        return PRIORITY_TEXT
    if utype == "image":
        if unit.get("vision_context") and str(unit.get("vision_context", "")).strip():
            return PRIORITY_VISION
        return PRIORITY_OCR
    return PRIORITY_TEXT


def _get_text(unit: dict[str, Any]) -> str:
    """Extract primary text from unit."""
    for key in ("text", "merged_context", "context_ready", "translated", "original"):
        v = unit.get(key)
        if v and str(v).strip():
            return str(v).strip()
    return ""


def deduplicate_content(
    units: list[dict[str, Any]],
    similarity_threshold: float | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """
    Remove overlapping content across text, table, image units.
    Uses hash (exact) + embedding similarity (semantic).
    Returns deduplicated list with merged best versions.
    """
    settings = get_settings()
    threshold = similarity_threshold or float(getattr(settings, "content_dedup_similarity_threshold", 0.85))

    # Filter empty
    with_text = [u for u in units if _get_text(u)]
    if not with_text:
        return []

    seen_hashes: set[str] = set()
    kept: list[dict[str, Any]] = []
    kept_normalized: list[str] = []

    # Phase 1: exact hash dedup
    hash_candidates: list[dict[str, Any]] = []
    for u in with_text:
        t = _get_text(u)
        h = content_hash(t)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        hash_candidates.append(u)

    if on_progress:
        on_progress("hash_dedup", len(with_text) - len(hash_candidates), len(with_text))

    if len(hash_candidates) <= 1:
        return hash_candidates

    # Phase 2: semantic similarity dedup (batch embeddings)
    embedder = get_embedder()
    texts = [_get_text(u) for u in hash_candidates]
    try:
        embeddings = embedder.encode(texts, normalize_embeddings=True)
    except Exception as e:
        logger.warning("Embedding for dedup failed: %s; skipping semantic dedup", e)
        return hash_candidates

    # kept: list of (unit, embedding) so we can compare without re-indexing
    kept_with_emb: list[tuple[dict[str, Any], np.ndarray]] = []

    for i, unit in enumerate(hash_candidates):
        emb = np.asarray(embeddings[i])
        norm = normalize_text(_get_text(unit))
        if len(norm) < 15:
            kept_with_emb.append((unit, emb))
            continue

        merged_into_idx: int | None = None
        sim_max = 0.0

        for j, (k_unit, k_emb) in enumerate(kept_with_emb):
            k_norm = normalize_text(_get_text(k_unit))
            if norm == k_norm:
                merged_into_idx = j
                sim_max = 1.0
                break
            cos_sim = float(np.dot(emb, k_emb))
            cos_sim = min(1.0, max(-1.0, cos_sim))
            if cos_sim > sim_max:
                sim_max = cos_sim
                merged_into_idx = j

        if merged_into_idx is not None and sim_max >= threshold:
            k_unit, k_emb = kept_with_emb[merged_into_idx]
            pa = _source_priority(unit)
            pb = _source_priority(k_unit)
            merged = merge_content(unit, k_unit, pa, pb)
            kept_with_emb[merged_into_idx] = (merged, emb if pa >= pb else k_emb)
        else:
            kept_with_emb.append((unit, emb))

    kept = [u for u, _ in kept_with_emb]

    if on_progress:
        on_progress("semantic_dedup", len(hash_candidates) - len(kept), len(hash_candidates))

    logger.info(
        "Content dedup: %s units -> %s unique (threshold=%.2f)",
        len(with_text),
        len(kept),
        threshold,
    )
    return kept
