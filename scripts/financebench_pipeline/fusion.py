from __future__ import annotations

import math
import re

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
    if source == "primary_statement_cell":
        return _table_base_score(item) + 0.4
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


_BUCKET_PRIORITY = {
    "exact_cell": 5,
    "metric_cell": 4,
    "statement": 3,
    "document": 2,
    "weak_or_mismatch": 1,
}


def _has_numeric_value(item: RetrieverResultItem) -> bool:
    metadata = item.metadata or {}
    if metadata.get("numeric_value") is not None:
        return True
    content = str(item.content or "")
    return "numeric_value=" in content and "numeric_value=None" not in content


def _year_status(item: RetrieverResultItem, requested_years: list[str] | None = None) -> str:
    """Return matched/unknown/conflict.

    Unknown is neutral: if no requested year or no explicit candidate year is
    available, the item is not demoted. Conflict is used only when the candidate
    exposes explicit years and none match the requested years.
    """
    requested = {str(year) for year in (requested_years or []) if str(year).strip()}
    if not requested:
        return "unknown"

    metadata = item.metadata or {}
    if int(metadata.get("year_hit") or 0) > 0:
        return "matched"

    col_header = _normalize_text(str(metadata.get("col_header") or ""))
    content = _normalize_text(str(item.content or ""))
    haystack = " ".join(part for part in (col_header, content) if part)
    if any(year in haystack for year in requested):
        return "matched"

    candidate_years = set(re.findall(r"\b(?:19|20)\d{2}\b", haystack))
    if candidate_years:
        return "conflict"
    return "unknown"


def _source_priority(item: RetrieverResultItem) -> int:
    source = (item.metadata or {}).get("source")
    return {
        "metrics_table": 4,
        "table": 3,
        "primary_statement_cell": 2,
        "primary_statement": 2,
        "chunk": 1,
    }.get(str(source), 0)


def _evidence_bucket(
    item: RetrieverResultItem,
    target_doc_ids: list[str] | None = None,
    requested_years: list[str] | None = None,
) -> tuple[int, str, dict[str, object]]:
    """Assign a deterministic constraint bucket from metadata only.

    Hard financial constraints (document / metric / period / numeric cell)
    decide the bucket and therefore the rank tier. A per-source base score is
    only used by the caller as a within-bucket tie-break. There are no tunable
    fusion weights here. Items are demoted, never dropped, so recall is kept.
    """
    metadata = item.metadata or {}
    source = metadata.get("source")
    doc_id = metadata.get("doc_id")

    allowed = set(target_doc_ids or [])
    # With no doc constraint (or an item without a doc_id) we cannot prove a
    # mismatch, so we do not penalise it.
    doc_mismatch = bool(allowed) and bool(doc_id) and doc_id not in allowed
    doc_match = bool(allowed) and bool(doc_id) and doc_id in allowed
    doc_unknown = not allowed or not doc_id

    content = _normalize_text(str(item.content or ""))
    row_header = _normalize_text(str(metadata.get("row_header") or ""))
    col_header = _normalize_text(str(metadata.get("col_header") or ""))
    numeric_present = _has_numeric_value(item)
    row_hit = int(metadata.get("row_hit") or 0)
    section_hit = int(metadata.get("section_hit") or 0)
    metric_match = (
        (source == "metrics_table" and (row_hit > 0 or bool(metadata.get("target_name"))))
        or (source in {"table", "primary_statement_cell"} and bool(row_header))
    )
    statement_match = source in {"primary_statement", "primary_statement_cell"} or section_hit > 0
    status = _year_status(item, requested_years)

    features = {
        "doc_match": doc_match,
        "doc_unknown": doc_unknown,
        "doc_mismatch": doc_mismatch,
        "metric_match": metric_match,
        "year_status": status,
        "numeric_present": numeric_present,
        "statement_match": statement_match,
        "source_priority": _source_priority(item),
    }

    if doc_mismatch:
        return _BUCKET_PRIORITY["weak_or_mismatch"], "weak_or_mismatch", features
    if metric_match and numeric_present and status == "matched":
        return _BUCKET_PRIORITY["exact_cell"], "exact_cell", features
    if metric_match and numeric_present and status != "conflict":
        return _BUCKET_PRIORITY["metric_cell"], "metric_cell", features
    if statement_match and status != "conflict":
        return _BUCKET_PRIORITY["statement"], "statement", features
    if doc_match or doc_unknown:
        if row_header or col_header or content:
            return _BUCKET_PRIORITY["document"], "document", features
    return _BUCKET_PRIORITY["weak_or_mismatch"], "weak_or_mismatch", features


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
        elif source == "primary_statement_cell":
            source_weight = 0.9
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
