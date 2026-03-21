"""
Hybrid per-chunk graph extraction: NER (optional spaCy) + rule-based relations,
optional LLM augment when confidence is low. Strict schema, normalization, dedup.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from config import get_settings

logger = logging.getLogger("graph_rag.services.chunk_processor")

# Strict schema — only these relationship types are persisted.
ALLOWED_RELATIONSHIP_TYPES: frozenset[str] = frozenset({
    "HAS_VALUE",
    "INCREASED_FROM",
    "DECREASED_FROM",
    "USED_FOR",
    "RELATED_TO",
})

ENTITY_TYPES: frozenset[str] = frozenset({"concept", "number", "date", "metric"})

_ENTITY_STOP = frozenset({
    "significant", "figure", "growth", "strong", "past", "performance",
    "the", "this", "that", "these", "those", "data", "table", "chart",
    "document", "section", "page", "item", "value", "total", "following",
})

_SYNONYM_CANONICAL: dict[str, str] = {
    "sales": "revenue",
    "turnover": "revenue",
    "income": "revenue",
    "profit": "earnings",
    "net income": "earnings",
    "costs": "cost",
    "expenses": "cost",
    "yr": "year",
    "fy": "fiscal year",
}

_SPACY_NLP = None


def _get_spacy():
    global _SPACY_NLP
    s = get_settings()
    if not s.graph_use_spacy_ner:
        return None
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    try:
        import spacy

        _SPACY_NLP = spacy.load("en_core_web_sm")
        return _SPACY_NLP
    except Exception as e:
        logger.warning("spaCy NER unavailable (%s); using regex/heuristics only", e)
        return None


@dataclass
class MentionRow:
    chunk_id: str
    text: str
    source: str
    page: int
    graph_key: str
    display_name: str
    entity_type: str = "concept"


@dataclass
class RelRow:
    from_key: str
    to_key: str
    from_name: str
    to_name: str
    rel_type: str


@dataclass
class ProcessedChunk:
    chunk_index: int
    chunk_id: str
    text: str
    source: str
    page: int
    mentions: list[MentionRow] = field(default_factory=list)
    relationships: list[RelRow] = field(default_factory=list)
    from_cache: bool = False
    extraction_confidence: float = 0.0
    used_llm: bool = False


def _hash_chunk_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_entity_label(raw: str) -> str:
    s = str(raw).strip().lower()
    s = re.sub(r"\s+", " ", s)
    if not s or len(s) > 64:
        return ""
    if s in _ENTITY_STOP:
        return ""
    if re.match(r"^(in|the|a|an)\s+", s) and len(s) > 24:
        return ""
    if "," in s and len(s) > 32:
        return ""
    if s.endswith("…") or s.endswith("..."):
        s = s.rstrip("….").strip()
    if len(s) < 2:
        return ""
    canon = _SYNONYM_CANONICAL.get(s, s)
    return canon[:200]


def graph_key_for(document_id: str, normalized_label: str) -> str:
    return f"{document_id}::{normalized_label}"


def _coerce_entity_type(raw: str | None) -> str:
    t = (raw or "concept").strip().lower()
    return t if t in ENTITY_TYPES else "concept"


def _coerce_rel_type(raw: str) -> str | None:
    if not raw:
        return None
    t = str(raw).strip().upper()
    t = re.sub(r"[^A-Z_]", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    if t in ALLOWED_RELATIONSHIP_TYPES:
        return t
    aliases = {
        "VALUE": "HAS_VALUE",
        "HAS": "HAS_VALUE",
        "CONTAINS": "RELATED_TO",
        "PART": "RELATED_TO",
        "PART_OF": "RELATED_TO",
        "INCREASE": "INCREASED_FROM",
        "INCREASES": "INCREASED_FROM",
        "DECREASE": "DECREASED_FROM",
        "DECREASES": "DECREASED_FROM",
        "USE": "USED_FOR",
        "USES": "USED_FOR",
        "FOR": "USED_FOR",
        "RELATES": "RELATED_TO",
        "RELATE": "RELATED_TO",
        "SIMILAR": "RELATED_TO",
    }
    if t in aliases:
        return aliases[t]
    for allowed in ALLOWED_RELATIONSHIP_TYPES:
        if allowed in t or t in allowed:
            return allowed
    return None


def _parse_strict_json_object(text: str) -> dict[str, Any] | None:
    if not text or not text.strip():
        return None
    raw = text.strip()
    json_str = None
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            json_str = m.group(1).strip()
    if not json_str:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            json_str = m.group(0)
    if not json_str:
        return None
    try:
        obj = json.loads(json_str)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


# --- Regex / heuristic entities & relations ---

_ML_TERMS = re.compile(
    r"\b(LSTM|CNN|RNN|GRU|Transformer|BERT|GPT|GAN|VAE|attention|neural network|deep learning|machine learning)\b",
    re.I,
)
_FIN_TERMS = re.compile(
    r"\b(revenue|cost|profit|earnings|margin|ebitda|sales|growth|forecast|budget)\b",
    re.I,
)


def _regex_entities(text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(label: str, etype: str) -> None:
        n = normalize_entity_label(label)
        if not n or n in seen:
            return
        seen.add(n)
        out.append({"label": n, "type": etype})

    for m in re.finditer(r"\b(20\d{2}|19\d{2})\b", text):
        add(m.group(1), "date")
    for m in re.finditer(
        r"\b(\d+(?:\.\d+)?)\s*(?:%|percent|million|billion|M|B|USD|usd|bn|mn)\b",
        text,
        re.I,
    ):
        add(m.group(0).strip().lower().replace("  ", " "), "number")
    for m in _ML_TERMS.finditer(text):
        add(m.group(1), "concept")
    for m in _FIN_TERMS.finditer(text):
        add(m.group(1), "metric")
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text):
        if len(m.group(1)) < 28:
            add(m.group(1), "concept")
    return out[:24]


def _spacy_entities(text: str) -> list[dict[str, str]]:
    nlp = _get_spacy()
    if not nlp:
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    try:
        doc = nlp(text[:8000])
        for ent in doc.ents:
            lab = ent.text.strip()
            if len(lab) > 64:
                continue
            et = "concept"
            if ent.label_ in ("DATE", "TIME"):
                et = "date"
            elif ent.label_ in ("MONEY", "QUANTITY", "PERCENT", "CARDINAL"):
                et = "number"
            elif ent.label_ in ("ORG", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"):
                et = "concept"
            n = normalize_entity_label(lab)
            if not n or n in seen:
                continue
            seen.add(n)
            out.append({"label": n, "type": et})
    except Exception as e:
        logger.debug("spaCy ent failed: %s", e)
    return out[:24]


def _rule_relationships(entities: list[dict[str, str]], text: str) -> list[dict[str, str]]:
    """Deterministic patterns; endpoints must match normalized labels present in entities."""
    labels = {e["label"] for e in entities}
    rels: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def add_rel(a: str, b: str, rt: str) -> None:
        fa, tb = normalize_entity_label(a), normalize_entity_label(b)
        if not fa or not tb:
            return
        if fa not in labels or tb not in labels:
            return
        coerced = _coerce_rel_type(rt)
        if not coerced:
            return
        k = (fa, tb, coerced)
        if k in seen:
            return
        seen.add(k)
        rels.append({"from": fa, "to": tb, "type": coerced})

    low = text.lower()
    # increased from X to Y / rose from … to …
    for m in re.finditer(
        r"(increased|rose|grew|up)\s+from\s+([\w\s\.%]+?)\s+to\s+([\w\s\.%]+?)(?:[.\n,;]|$)",
        low,
        re.I,
    ):
        add_rel(m.group(2).strip(), m.group(3).strip(), "INCREASED_FROM")
    for m in re.finditer(
        r"(decreased|fell|dropped|down)\s+from\s+([\w\s\.%]+?)\s+to\s+([\w\s\.%]+?)(?:[.\n,;]|$)",
        low,
        re.I,
    ):
        add_rel(m.group(2).strip(), m.group(3).strip(), "DECREASED_FROM")
    # used for / used to
    for m in re.finditer(r"([\w][\w\s]{2,40}?)\s+is\s+used\s+for\s+([\w][\w\s]{2,40}?)(?:[.\n,;]|$)", low, re.I):
        add_rel(m.group(1).strip(), m.group(2).strip(), "USED_FOR")
    for m in re.finditer(r"([\w][\w\s]{2,40}?)\s+used\s+for\s+([\w][\w\s]{2,40}?)(?:[.\n,;]|$)", low, re.I):
        add_rel(m.group(1).strip(), m.group(2).strip(), "USED_FOR")

    # Pair metric-like with nearest number in same sentence windows
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        nums = [normalize_entity_label(x) for x in re.findall(r"\b\d+(?:\.\d+)?%?\b", sent)]
        metrics = [normalize_entity_label(m.group(1)) for m in _FIN_TERMS.finditer(sent)]
        for met in metrics:
            for num in nums:
                if met and num and met in labels and num in labels:
                    add_rel(met, num, "HAS_VALUE")

    # Weak RELATED_TO: two distinct concepts in one short sentence
    for sent in sentences:
        if len(sent) > 160:
            continue
        caps = [normalize_entity_label(m.group(1)) for m in _ML_TERMS.finditer(sent)]
        fins = [normalize_entity_label(m.group(1)) for m in _FIN_TERMS.finditer(sent)]
        for a in caps:
            for b in fins:
                if a and b and a != b and a in labels and b in labels:
                    add_rel(a, b, "RELATED_TO")

    return rels[:16]


def _extraction_confidence(entities: list[dict], relationships: list[dict], text: str) -> float:
    if not text.strip():
        return 0.0
    cov = min(1.0, (len(entities) * 0.12) + (len(relationships) * 0.18))
    len_bonus = min(0.25, len(text) / 2000.0)
    rel_bonus = 0.2 if relationships else 0.0
    return max(0.0, min(1.0, cov + len_bonus * 0.5 + rel_bonus))


def hybrid_extract_sync(chunk_text: str) -> tuple[dict[str, Any], float, bool]:
    """
    Returns (raw_graph_dict, confidence, used_llm).
    raw_graph: {"entities": [{"label","type"},...] or strings, "relationships": [...]}
    """
    text = (chunk_text or "").strip()
    if not text:
        return {"entities": [], "relationships": []}, 0.0, False

    ent_map: dict[str, str] = {}
    for e in _regex_entities(text):
        ent_map[e["label"]] = e["type"]
    for e in _spacy_entities(text):
        if e["label"] not in ent_map:
            ent_map[e["label"]] = e["type"]

    entities = [{"label": k, "type": v} for k, v in ent_map.items()]
    relationships = _rule_relationships(entities, text)
    conf = _extraction_confidence(entities, relationships, text)

    settings = get_settings()
    need_llm = conf < settings.graph_llm_fallback_threshold or (
        len(entities) >= 3 and len(relationships) == 0
    )

    used_llm = False
    if need_llm:
        llm_raw = _llm_extract_sync(text)
        used_llm = True
        # Merge: LLM entities/relations augment hybrid
        for e in llm_raw.get("entities") or []:
            if isinstance(e, dict):
                lab = normalize_entity_label(str(e.get("label", e.get("text", ""))))
                if lab:
                    ent_map.setdefault(lab, _coerce_entity_type(e.get("type")))
            else:
                lab = normalize_entity_label(str(e))
                if lab:
                    ent_map.setdefault(lab, "concept")
        entities = [{"label": k, "type": v} for k, v in ent_map.items()]
        for r in llm_raw.get("relationships") or []:
            if isinstance(r, dict):
                relationships.append(
                    {
                        "from": str(r.get("from", r.get("subject", ""))),
                        "to": str(r.get("to", r.get("object", ""))),
                        "type": str(r.get("type", r.get("predicate", ""))),
                    }
                )
        # Dedupe relationships
        seen_r: set[tuple[str, str, str]] = set()
        deduped: list[dict[str, str]] = []
        for r in relationships:
            fa = normalize_entity_label(r.get("from", ""))
            tb = normalize_entity_label(r.get("to", ""))
            rt = _coerce_rel_type(r.get("type", ""))
            if not fa or not tb or not rt:
                continue
            k = (fa, tb, rt)
            if k in seen_r:
                continue
            seen_r.add(k)
            deduped.append({"from": fa, "to": tb, "type": rt})
        relationships = deduped[:20]
        conf = max(conf, _extraction_confidence(entities, relationships, text))

    return (
        {"entities": entities, "relationships": relationships},
        conf,
        used_llm,
    )


EXTRACTION_PROMPT_TEMPLATE = """You extract a small knowledge graph from ONE text chunk.

Return ONLY a JSON object (no markdown, no explanation) with this exact shape:
{{"entities": [{{"label":"short name","type":"concept|number|date|metric"}}], "relationships": [{{"from":"label1","to":"label2","type":"HAS_VALUE"}}]}}

Rules:
- entities: 0–20 items; type MUST be one of: concept, number, date, metric
- relationships: each has "from", "to", "type".
- type MUST be exactly one of: HAS_VALUE, INCREASED_FROM, DECREASED_FROM, USED_FOR, RELATED_TO
  - HAS_VALUE: value attached to a metric/concept
  - INCREASED_FROM / DECREASED_FROM: comparative change
  - USED_FOR: purpose or use
  - RELATED_TO: general association
- "from" and "to" MUST match entity labels (normalized short phrases).

Text chunk:
---
{chunk_text}
---
JSON only:"""


def _llm_extract_sync(chunk_text: str, max_chars: int = 2800) -> dict[str, Any]:
    from services.llm_client import LLMClient

    text = (chunk_text or "").strip()[:max_chars]
    if not text:
        return {"entities": [], "relationships": []}
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=text)
    llm = LLMClient()
    resp = llm.generate(prompt, max_tokens=768)
    parsed = _parse_strict_json_object(resp or "")
    if not parsed:
        logger.debug("LLM extraction returned no JSON; hybrid-only")
        return {"entities": [], "relationships": []}
    return parsed


def build_processed_chunk(
    chunk_index: int,
    chunk: dict[str, Any],
    document_id: str,
    raw: dict[str, Any],
    extraction_confidence: float = 0.0,
    used_llm: bool = False,
) -> ProcessedChunk:
    chunk_id = str(chunk.get("chunk_id") or uuid.uuid4())
    text = (chunk.get("text") or chunk.get("content") or "").strip()
    source = str(chunk.get("source") or "")
    page = int(chunk.get("page") or 0)

    norm_to_display: dict[str, str] = {}
    norm_to_type: dict[str, str] = {}
    for e in raw.get("entities") or []:
        if isinstance(e, dict):
            n = normalize_entity_label(str(e.get("label", e.get("text", ""))))
            if not n:
                continue
            if n not in norm_to_display:
                norm_to_display[n] = n
                norm_to_type[n] = _coerce_entity_type(e.get("type"))
        else:
            n = normalize_entity_label(str(e))
            if n and n not in norm_to_display:
                norm_to_display[n] = n
                norm_to_type[n] = "concept"

    mentions: list[MentionRow] = []
    for norm, disp in norm_to_display.items():
        gk = graph_key_for(document_id, norm)
        mentions.append(
            MentionRow(
                chunk_id=chunk_id,
                text=text[:10000],
                source=source,
                page=page,
                graph_key=gk,
                display_name=disp,
                entity_type=norm_to_type.get(norm, "concept"),
            )
        )

    seen_rel: set[tuple[str, str, str]] = set()
    relationships: list[RelRow] = []

    for r in raw.get("relationships") or []:
        if not isinstance(r, dict):
            continue
        fa = normalize_entity_label(str(r.get("from", r.get("subject", ""))))
        tb = normalize_entity_label(str(r.get("to", r.get("object", ""))))
        rt = _coerce_rel_type(str(r.get("type", r.get("predicate", ""))))
        if not fa or not tb or not rt:
            continue
        fk = graph_key_for(document_id, fa)
        tk = graph_key_for(document_id, tb)
        key = (fk, tk, rt)
        if key in seen_rel:
            continue
        seen_rel.add(key)
        relationships.append(
            RelRow(
                from_key=fk,
                to_key=tk,
                from_name=fa,
                to_name=tb,
                rel_type=rt,
            )
        )

    return ProcessedChunk(
        chunk_index=chunk_index,
        chunk_id=chunk_id,
        text=text,
        source=source,
        page=page,
        mentions=mentions,
        relationships=relationships,
        extraction_confidence=extraction_confidence,
        used_llm=used_llm,
    )


def extract_chunk_sync(
    chunk_index: int,
    chunk: dict[str, Any],
    document_id: str,
    cache: dict[str, dict[str, Any]],
    cache_max: int,
) -> ProcessedChunk:
    """Hybrid extract (+ optional LLM) with hash cache. Sync — run in executor."""
    text = (chunk.get("text") or chunk.get("content") or "").strip()
    if not chunk.get("chunk_id"):
        chunk = {**chunk, "chunk_id": str(uuid.uuid4())}

    if not text:
        return ProcessedChunk(
            chunk_index=chunk_index,
            chunk_id=str(chunk["chunk_id"]),
            text="",
            source=str(chunk.get("source") or ""),
            page=int(chunk.get("page") or 0),
            mentions=[],
            relationships=[],
            extraction_confidence=0.0,
            used_llm=False,
        )

    h = _hash_chunk_text(text)
    if h in cache:
        entry = cache[h]
        raw = entry.get("raw") or entry
        conf = float(entry.get("confidence", 0))
        used = bool(entry.get("used_llm", False))
        pc = build_processed_chunk(chunk_index, chunk, document_id, raw, conf, used)
        pc.from_cache = True
        return pc

    raw, conf, used_llm = hybrid_extract_sync(text)
    cache[h] = {"raw": raw, "confidence": conf, "used_llm": used_llm}
    if len(cache) > cache_max:
        keys = list(cache.keys())
        for k in keys[: cache_max // 2]:
            del cache[k]

    return build_processed_chunk(chunk_index, chunk, document_id, raw, conf, used_llm)
