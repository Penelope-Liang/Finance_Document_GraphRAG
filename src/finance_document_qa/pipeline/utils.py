from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def _load_questions(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            doc_name = row.get("doc_name") or str(row.get("doc_id") or "").removesuffix(
                ".pdf"
            )
            row["doc_name"] = doc_name
            row["doc_id"] = row.get("doc_id") or (
                f"{doc_name}.pdf" if doc_name else None
            )
            row["question_id"] = (
                row.get("question_id") or row.get("financebench_id") or row.get("id")
            )
            row["gold_answer"] = row.get("gold_answer") or row.get("answer")
            row["gold_justification"] = row.get("gold_justification") or row.get(
                "justification"
            )
            rows.append(row)
    return rows


def _sanitize_value(value: Any) -> Any:
    """Convert non-JSON-safe missing numeric values to None.

    Missing financial values must not be coerced to 0 because that changes the
    meaning of later ratios/margins. None keeps them explicitly missing.
    """
    if value is None:
        return None
    if pd is not None and value is pd.NA:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if np is not None and isinstance(value, np.floating):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if np is not None and isinstance(value, np.integer):
        return int(value)
    return value


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_for_json(v) for v in obj)
    return _sanitize_value(obj)


def _assert_valid_json_payload(payload: Any) -> None:
    json.dumps(payload, allow_nan=False)


def _sanitize_retriever_item(item: RetrieverResultItem) -> RetrieverResultItem:
    return RetrieverResultItem(
        content=str(item.content or ""),
        metadata=_sanitize_for_json(dict(item.metadata or {})),
    )


def _sanitize_retriever_result(result: RetrieverResult) -> RetrieverResult:
    return RetrieverResult(
        items=[_sanitize_retriever_item(item) for item in (result.items or [])],
        metadata=_sanitize_for_json(dict(result.metadata or {})),
    )


def _save_results(output_path: Path, results: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_results = _sanitize_for_json(results)
    _assert_valid_json_payload(safe_results)
    output_path.write_text(
        json.dumps(safe_results, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _strip_think(text: str) -> str:
    """Remove reasoning traces emitted by local reasoning models."""
    cleaned = re.sub(
        r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE
    )
    cleaned = re.sub(
        r"^\s*Thinking\.\.\..*?\.\.\.done thinking\.\s*",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return cleaned.strip()


def _parse_answer_payload(raw_answer: str) -> dict[str, Any] | None:
    cleaned = _strip_think(raw_answer)
    # First try parsing the cleaned text directly
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to extracting the first {...} block
        json_str = _extract_json_object(cleaned)
        if json_str:
            try:
                payload = json.loads(json_str)
            except json.JSONDecodeError:
                return None
        else:
            return None
    return payload if isinstance(payload, dict) else None


def _extract_json_object(text: str) -> str | None:
    # Strip think blocks first so the greedy .* doesn't span across them
    cleaned = _strip_think(text)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    return match.group(0) if match else None
