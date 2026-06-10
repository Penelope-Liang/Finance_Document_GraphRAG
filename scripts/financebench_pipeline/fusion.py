from __future__ import annotations

import math

from neo4j_graphrag.types import RetrieverResultItem

from financebench_pipeline.filing_constraints import _extract_question_constraints
from financebench_pipeline.utils import _normalize_text, _safe_float, _sanitize_for_json


def _candidate_bonus(question: str, item: RetrieverResultItem) -> float:
    constraints = _extract_question_constraints(question)
    content = _normalize_text(str(item.content))
    metadata = item.metadata or {}
    section_name = _normalize_text(str(metadata.get("section_name") or ""))
    haystack = " ".join(part for part in (content, section_name) if part)

    score = 0.0
    for hint in constraints["section_hints"]:
        if hint in haystack:
            score += 2.0
        elif any(token in haystack for token in hint.split()):
            score += 0.6

    for year in constraints["years"]:
        if year in haystack:
            score += 1.5

    for quarter in constraints["quarters"]:
        q = f"q{quarter}"
        if q in haystack:
            score += 1.5
        if quarter == "1" and "three months ended" in haystack:
            score += 0.6
        if quarter == "2" and "six months ended" in haystack:
            score += 0.6

    for month in constraints["months"]:
        if month in haystack:
            score += 0.8

    for period_hint in constraints["period_hints"]:
        if period_hint in haystack:
            score += 1.0

    if metadata.get("source") == "table" and (
        constraints["section_hints"] or constraints["years"] or constraints["period_hints"]
    ):
        score += 0.4

    return score


def _table_base_score(item: RetrieverResultItem) -> float:
    metadata = item.metadata or {}
    score = 0.0
    if metadata.get("row_header"):
        score += 1.0
    if metadata.get("col_header"):
        score += 0.8
    if metadata.get("section_name"):
        score += 0.5

    content = _normalize_text(str(item.content or ""))
    if "numeric_value=" in content:
        score += 1.2
    if "table evidence:" in content:
        score += 0.5
    return score


def _item_base_score(item: RetrieverResultItem) -> float:
    metadata = item.metadata or {}
    source = metadata.get("source")
    if source == "chunk":
        return _safe_float(metadata.get("similarity_score"))
    if source == "table":
        return _table_base_score(item)
    if source == "metrics_table":
        return (
            _table_base_score(item)
            + 1.5 * int(metadata.get("section_hit") or 0)
            + 1.0 * int(metadata.get("row_hit") or 0)
            + 0.8 * int(metadata.get("year_hit") or 0)
        )
    if source == "primary_statement":
        # Primary statements are deterministically matched, all are equally relevant
        # to the question's statement-type hint, so give them a uniform high base.
        return 1.0
    return 0.0


def _softmax_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    scores = [_safe_float(score) for score in scores]
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps)
    if total <= 0 or math.isnan(total) or math.isinf(total):
        return [0.0 for _ in scores]
    return [value / total for value in exps]


def _attach_fusion_scores(
    question: str,
    items: list[RetrieverResultItem],
    chunk_weight: float = 0.5,
    table_weight: float = 0.5,
    bonus_weight: float = 0.15,
) -> list[RetrieverResultItem]:
    by_source: dict[str, list[int]] = {}
    base_scores: list[float] = []
    bonuses: list[float] = []

    for idx, item in enumerate(items):
        source = str((item.metadata or {}).get("source") or "unknown")
        by_source.setdefault(source, []).append(idx)
        base_scores.append(_item_base_score(item))
        bonuses.append(_candidate_bonus(question, item))

    normalized_by_idx = [0.0] * len(items)
    for source, indices in by_source.items():
        source_scores = [base_scores[i] for i in indices]
        normalized = _softmax_normalize(source_scores)
        for idx, norm_score in zip(indices, normalized):
            normalized_by_idx[idx] = norm_score

    fused_items: list[RetrieverResultItem] = []
    for idx, item in enumerate(items):
        metadata = dict(item.metadata or {})
        source = str(metadata.get("source") or "unknown")
        if source == "chunk":
            source_weight = chunk_weight
        elif source == "table":
            source_weight = table_weight
        elif source == "metrics_table":
            source_weight = 2.2
        elif source == "primary_statement":
            # Differentiated weight by match_priority:
            #   3 = both statement_type and section_name match (strongest signal)
            #   2 = only statement_type matches (PepsiCo-style: section_name lost during ingest)
            #   1 = only section_name matches (footnote / supplementary table)
            priority = int((item.metadata or {}).get("match_priority") or 2)
            source_weight = {3: 1.6, 2: 1.3, 1: 1.0}.get(priority, 1.3)
        else:
            source_weight = 1.0
        fused_score = source_weight * normalized_by_idx[idx] + bonus_weight * bonuses[idx]
        fused_score = _safe_float(fused_score)
        metadata.update(
            {
                "raw_score": _safe_float(base_scores[idx]),
                "normalized_score": _safe_float(normalized_by_idx[idx]),
                "constraint_bonus": _safe_float(bonuses[idx]),
                "source_weight": _safe_float(source_weight),
                "fused_score": fused_score,
            }
        )
        fused_items.append(
            RetrieverResultItem(
                content=item.content,
                metadata=_sanitize_for_json(metadata),
            )
        )
    return fused_items
