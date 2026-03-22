"""
Graph selection: entity clustering + ranking (≤150), relationship scoring (≤300).
"Compress and prioritize — never randomly drop information"
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from services.chunk_processor import (
    ALLOWED_RELATIONSHIP_TYPES,
    MentionRow,
    ProcessedChunk,
    RelRow,
)

logger = logging.getLogger("graph_rag.services.graph_selection")

# Entity selection limits (overridden by settings)
MAX_ENTITIES = 250
MAX_RELATIONSHIPS = 500

# Relation type weights (higher = more important)
REL_TYPE_WEIGHTS: dict[str, float] = {
    "HAS_VALUE": 3.0,
    "INCREASED_FROM": 2.5,
    "DECREASED_FROM": 2.5,
    "USED_FOR": 2.0,
    "RELATED_TO": 1.0,
}

# Numeric/metric/date entities get bonus
ENTITY_IMPORTANCE_TYPES = frozenset({"number", "metric", "date"})


@dataclass
class EntityStats:
    graph_key: str
    display_name: str
    entity_type: str
    frequency: int = 0
    chunk_ids: set[str] = field(default_factory=set)
    connection_count: int = 0
    score: float = 0.0


def _select_entities_cluster_rank(
    entity_stats: dict[str, EntityStats],
    max_entities: int,
) -> set[str]:
    """
    Cluster entities by embedding similarity, pick best per cluster, then global rank.
    """
    if len(entity_stats) <= max_entities:
        return set(entity_stats.keys())

    try:
        from services.embedder import get_embedder
        from sklearn.cluster import KMeans
        import numpy as np
    except ImportError:
        return _select_entities_global_only(entity_stats, max_entities)

    embedder = get_embedder()
    keys = list(entity_stats.keys())
    labels = [entity_stats[k].display_name or k.split("::")[-1] for k in keys]
    try:
        emb = embedder.encode(labels, normalize_embeddings=True)
    except Exception as e:
        logger.warning("Entity embedding failed: %s", e)
        return _select_entities_global_only(entity_stats, max_entities)

    k = min(max_entities, len(keys))
    try:
        km = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=100)
        cluster_labels = km.fit_predict(emb)
    except Exception as e:
        logger.warning("Entity clustering failed: %s", e)
        return _select_entities_global_only(entity_stats, max_entities)

    clusters: dict[int, list[str]] = defaultdict(list)
    for i, lab in enumerate(cluster_labels):
        clusters[int(lab)].append(keys[i])

    selected: set[str] = set()
    for lab, ents in clusters.items():
        best = max(ents, key=lambda e: entity_stats[e].score)
        selected.add(best)

    if len(selected) <= max_entities:
        return selected
    sorted_all = sorted(
        entity_stats.keys(),
        key=lambda e: entity_stats[e].score,
        reverse=True,
    )
    return set(sorted_all[:max_entities])


def _select_entities_global_only(entity_stats: dict[str, EntityStats], max_entities: int) -> set[str]:
    """Fallback: sort by score, take top N."""
    sorted_keys = sorted(
        entity_stats.keys(),
        key=lambda e: entity_stats[e].score,
        reverse=True,
    )
    return set(sorted_keys[:max_entities])


def aggregate_and_select_entities(
    processed_chunks: list[ProcessedChunk],
    document_id: str,
    max_entities: int = MAX_ENTITIES,
) -> set[str]:
    """
    Aggregate entity stats, score, cluster+rank, return selected graph_keys.
    score = frequency*2 + connections*3 + multi_chunk*2 + numeric_or_metric_bonus
    """
    entity_stats: dict[str, EntityStats] = {}
    rel_degree: dict[str, int] = defaultdict(int)

    for pc in processed_chunks:
        for m in pc.mentions:
            gk = m.graph_key
            if gk not in entity_stats:
                entity_stats[gk] = EntityStats(
                    graph_key=gk,
                    display_name=m.display_name,
                    entity_type=m.entity_type or "concept",
                )
            es = entity_stats[gk]
            es.frequency += 1
            es.chunk_ids.add(m.chunk_id)

        for r in pc.relationships:
            if r.rel_type not in ALLOWED_RELATIONSHIP_TYPES:
                continue
            rel_degree[r.from_key] += 1
            rel_degree[r.to_key] += 1

    for gk, es in entity_stats.items():
        es.connection_count = rel_degree.get(gk, 0)
        multi_chunk = 2 if len(es.chunk_ids) > 1 else 0
        type_bonus = 2.0 if es.entity_type in ENTITY_IMPORTANCE_TYPES else 0.0
        es.score = (
            es.frequency * 2.0
            + es.connection_count * 3.0
            + multi_chunk * 2.0
            + type_bonus
        )

    return _select_entities_cluster_rank(entity_stats, max_entities)


def filter_and_select_relationships(
    processed_chunks: list[ProcessedChunk],
    selected_entities: set[str],
    max_relationships: int = MAX_RELATIONSHIPS,
) -> list[RelRow]:
    """
    Keep only rels where both entities selected.
    Score: entity_importance*2 + frequency*2 + relation_type_weight
    Return top max_relationships.
    """
    selected = selected_entities
    rel_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    rel_rows: dict[tuple[str, str, str], RelRow] = {}
    entity_scores: dict[str, float] = {}

    for pc in processed_chunks:
        for m in pc.mentions:
            entity_scores[m.graph_key] = entity_scores.get(m.graph_key, 0) + 1.0
        for r in pc.relationships:
            if r.rel_type not in ALLOWED_RELATIONSHIP_TYPES:
                continue
            if r.from_key not in selected or r.to_key not in selected:
                continue
            k = (r.from_key, r.to_key, r.rel_type)
            rel_counts[k] += 1
            rel_rows[k] = r

    scored: list[tuple[float, RelRow]] = []
    for k, r in rel_rows.items():
        from_key, to_key, rel_type = k
        freq = rel_counts[k]
        ent_imp = entity_scores.get(from_key, 0) + entity_scores.get(to_key, 0)
        type_w = REL_TYPE_WEIGHTS.get(rel_type, 1.0)
        score = ent_imp * 2.0 + freq * 2.0 + type_w
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:max_relationships]]


def filter_mentions_by_entities(
    processed_chunks: list[ProcessedChunk],
    selected_entities: set[str],
) -> list[tuple[str, MentionRow]]:
    """Return (chunk_id, mention) for mentions whose graph_key is in selected_entities."""
    out: list[tuple[str, MentionRow]] = []
    for pc in processed_chunks:
        for m in pc.mentions:
            if m.graph_key in selected_entities:
                out.append((pc.chunk_id, m))
    return out
